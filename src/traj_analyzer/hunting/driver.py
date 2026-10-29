from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from aifn import Access, Refused, Returned
from pydantic import BaseModel

from traj_analyzer.calls import WorkspaceCalls
from traj_analyzer.extract import SubprocessWorkers, WorkerRunner, extract
from traj_analyzer.hunting.draw import Draw, blind_draws, draw, hits_of
from traj_analyzer.hunting.functions import (
    FACETS,
    FINDINGS,
    LEDGER,
    NEED,
    NEW,
    PROBES,
    PROPOSE,
    SCENARIO,
    VERIFY,
    VERIFY_INSTRUCTION,
    Case,
    CaseVerdict,
    Proposal,
    ProposeRequest,
    VerifyRequest,
    function_name,
    hunt_functions,
    propose_instruction,
)
from traj_analyzer.hunting.kept import Kept, KeptProbe, write_kept
from traj_analyzer.hunting.ledger import (
    HuntState,
    ProbeState,
    Reading,
    RoundRecord,
    calibration,
    coverage,
    ledger,
    new_findings,
    probe_ledger,
    unproductive,
)
from traj_analyzer.hunting.marks import enable_marks, marks_instance
from traj_analyzer.hunting.probes import HITS_SCHEMA, probe_path, run_probes
from traj_analyzer.hunting.run import (
    FINAL,
    PENDING,
    REPORT,
    Hunt,
    note_step,
    open_hunt,
    over,
    resume,
    save_state,
    start,
)
from traj_analyzer.hunting.verdicts import write_verdicts
from traj_analyzer.hunting.workspace import prepare, write_case, write_file
from traj_analyzer.layers.tables import read_table
from traj_analyzer.operators.catalog import edit_operators, load_groups
from traj_analyzer.project import ConfigError, Project
from traj_analyzer.runtime import known_calls, mailbox


class Reads(BaseModel):
    """The reads of a round or of the closing check: the drawn trajectories and their cases, in the groups one reader
    reads, and the trajectories a probe hit that other readers read once more without the hits."""

    draws: list[Draw]
    groups: list[list[Case]]
    blind: list[Draw]
    blind_groups: list[list[Case]]
    files: dict[str, str]
    """The case file of every drawn trajectory, by key."""


Answers = list[tuple[list[Case], dict[str, CaseVerdict], str]]
"""The verdicts per group of cases, each group with the reason a case of it without a verdict has none."""


def hunt(
    project: Project, name: str, *, max_rounds: int | None = None, workers: int | None = None,
    runner: WorkerRunner | None = None, new_run: bool = False,
) -> dict[str, Any]:
    """Search the trajectories for what the hunt asks for with the aifn functions of the hunt, round after round,
    until `patience` rounds in a row find nothing new or `max_rounds` is reached; then read what the active probes
    hit and no round read, write the verdicts, keep the probes, mark every trajectory with them, and return the
    report.

    A run that stopped midway continues from the state of its last finished round, and the calls it had made are
    answered from the mailbox.
    """
    runner = runner or SubprocessWorkers()
    runner.check_credentials(project)
    found = open_hunt(project, name, create=True, max_rounds=max_rounds, new_run=new_run)
    if (found.directory / PENDING).is_file():
        raise ConfigError(f"Round of {found.directory} waits for the verdicts of an agent session; hand them in "
                          f"with `traj hunt verdicts {name}`")
    box = mailbox(project)
    calls = WorkspaceCalls(project=project, runner=runner, workers=workers or found.spec.workers, box=box,
                           known=known_calls(box), functions=hunt_functions(project, name, found.spec),
                           workspace=found.workspace)
    calls.placed.update(prepare(project, found.workspace, found.spec, found.keys))
    state = resume(found.directory) or start(project, name)
    while not over(found.spec, state):
        state = run_round(found, calls, state)
        save_state(found, state)
    draws = closing_draws(found, state)
    if not draws:
        return finish(found, state, [], [], [], [])
    note_step(found, state.round + 1, "close", "open")
    reads = write_reads(found, draws, f"rounds/{FINAL}", state.round + 1)
    readings, blind, unanswered, blind_unanswered = ask_verdicts(found, calls, state, reads, state.round + 1)
    note_step(found, state.round + 1, "close", "done")
    return finish(found, state, readings, blind, unanswered, blind_unanswered)


def propose_request(hunt: Hunt, number: int) -> ProposeRequest:
    """What the proposal of a round builds on: the digests of the files the state writes, read after it wrote them."""
    return ProposeRequest(hunt=hunt.name, round=number, model=hunt.model, files=digests(hunt.workspace, [
        NEED, SCENARIO, FACETS, PROBES, LEDGER, FINDINGS, f"rounds/{number - 1}/verdicts.json"]))


def run_round(hunt: Hunt, calls: WorkspaceCalls, state: HuntState) -> HuntState:
    """One round with the aifn functions: the proposal of probes, their run over every trajectory, the reads, and
    the verdicts."""
    number = state.round + 1
    state = state.model_copy(deep=True)
    write_state(hunt, state)
    request = propose_request(hunt, number)
    leftovers = [hunt.workspace / "notes" / f"{PROPOSE}-{number}.md", *calls.unplaced("operators")]
    note_step(hunt, number, "propose", "open")
    proposal = calls.alone(function_name(PROPOSE, hunt.name), propose_instruction(hunt.spec, number), request,
                           leftovers)
    write_file(hunt.workspace, f"rounds/{number}/proposal.json", proposal.model_dump_json(indent=1))
    note_step(hunt, number, "propose", "done")
    failed, retired, draws = begin_round(hunt, state, proposal, number)
    note_step(hunt, number, "read", "open")
    reads = write_reads(hunt, draws, f"rounds/{number}", number)
    readings, blind, unanswered, blind_unanswered = ask_verdicts(hunt, calls, state, reads, number)
    note_step(hunt, number, "read", "done")
    return end_round(hunt, state, number, readings, blind, unanswered, blind_unanswered, failed, sorted(retired))


def begin_round(
    hunt: Hunt, state: HuntState, proposal: Proposal, number: int
) -> tuple[dict[str, str], dict[str, int], list[Draw]]:
    """Take the proposal, run the probes over every trajectory, retire the unproductive ones and draw the reads of
    the round; return the probes that failed with their errors, those retired with their reads, and the draws."""
    apply(state, proposal, number)
    write_state(hunt, state)
    hits, failed = run_probes(hunt.project, hunt.workspace, state.active(), set(hunt.keys))
    for probe in state.active():
        if probe.name in failed:
            probe.status, probe.reason = "failed", failed[probe.name]
    state.hits = {probe.name: sorted(set(hits.loc[hits["probe"] == probe.name, "key"])) for probe in state.active()}
    retired = unproductive(state, hunt.spec)
    for probe in state.active():
        if probe.name in retired:
            probe.status = "unproductive"
            probe.reason = f"none of the {retired[probe.name]} of its hits read shows the need"
    state.hits = {name: found for name, found in state.hits.items() if name not in retired}
    hits = hits[hits["probe"].isin(list(state.hits))].reset_index(drop=True)
    (hunt.workspace / "rounds" / str(number)).mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(hits, schema=HITS_SCHEMA, preserve_index=False),
                   hunt.workspace / "rounds" / str(number) / "hits.parquet")
    write_state(hunt, state)
    return failed, retired, draw(hunt.spec, state, hits, hunt.keys, hunt.matrix, hunt.clusters, number)


def write_reads(hunt: Hunt, draws: list[Draw], directory: str, number: int) -> Reads:
    """Write every drawn trajectory whole as a case file under `directory`, and put the cases in the groups readers
    read: the drawn ones with the positions the probes hit, then the blind ones, drawn from those a probe hit, with
    none, in groups of their own."""
    cases = [Case(file=write_case(hunt.project, hunt.workspace, f"{directory}/cases", drawn.key), hits=drawn.hits)
             for drawn in draws]
    files = {drawn.key: case.file for drawn, case in zip(draws, cases, strict=True)}
    blind = blind_draws(hunt.spec, draws, number)
    return Reads(draws=draws, groups=groups(hunt, cases), blind=blind,
                 blind_groups=groups(hunt, [Case(file=files[drawn.key]) for drawn in blind]), files=files)


def groups(hunt: Hunt, cases: list[Case]) -> list[list[Case]]:
    """The cases in the groups one reader reads, `cases_per_call` each."""
    size = hunt.spec.cases_per_call
    return [cases[i:i + size] for i in range(0, len(cases), size)]


def ask_verdicts(
    hunt: Hunt, calls: WorkspaceCalls, state: HuntState, reads: Reads, number: int
) -> tuple[list[Reading], list[Reading], list[str], list[str]]:
    """Read the cases with the aifn function of the hunt, one call per group, all calls side by side."""
    files = digests(hunt.workspace, [NEED, FACETS, SCENARIO])
    chunks = [*reads.groups, *reads.blind_groups]
    handles = [calls.issue(function_name(VERIFY, hunt.name), VERIFY_INSTRUCTION,
                           VerifyRequest(hunt=hunt.name, round=number, files=files, cases=chunk, model=hunt.model),
                           Access.READ_ONLY) for chunk in chunks]
    answers: Answers = []
    for chunk, result in zip(chunks, calls.settle(handles), strict=True):
        outcome = result.outcome
        if isinstance(outcome, Returned):
            answers.append((chunk, {verdict.file: verdict for verdict in outcome.value.verdicts},
                            "the call gave no verdict for it"))
        else:
            answers.append((chunk, {}, f"refused: {outcome.reason}" if isinstance(outcome, Refused) else
                            f"failed: {outcome.fault}: {outcome.detail}"))
    shown = len(reads.groups)
    return record(state, reads, answers[:shown], answers[shown:], number)


def record(
    state: HuntState, reads: Reads, answers: Answers, blind_answers: Answers, number: int
) -> tuple[list[Reading], list[Reading], list[str], list[str]]:
    """What every case showed, the blind reads apart, with the trajectories that got no verdict, theirs apart too.

    A facet a verdict with the hits reports as new joins the facets of `state`; one a blind verdict reports stays
    out, so a blind verdict naming it gets no verdict of its own.
    """
    unanswered: list[str] = []
    blind_unanswered: list[str] = []
    readings = _readings(state, reads.draws, reads.files, answers, number, unanswered, adds=True)
    blind = _readings(state, reads.blind, reads.files, blind_answers, number, blind_unanswered, adds=False)
    return readings, blind, unanswered, blind_unanswered


def _readings(
    state: HuntState, draws: list[Draw], files: dict[str, str], answers: Answers, number: int, unanswered: list[str],
    adds: bool,
) -> list[Reading]:
    drawn_by_file = {files[drawn.key]: drawn for drawn in draws}
    readings: list[Reading] = []
    facets = {facet.name for facet in state.facets}
    for chunk, verdicts, missing in answers:
        for case in chunk:
            drawn = drawn_by_file[case.file]
            verdict = verdicts.get(case.file)
            if verdict is None:
                unanswered.append(f"{drawn.key}: {missing}")
                continue
            facet = verdict.facet
            if verdict.found and facet == NEW:
                assert verdict.new_facet is not None
                facet = verdict.new_facet.name
                if adds and facet not in facets:
                    state.facets.append(verdict.new_facet)
                    facets.add(facet)
            elif verdict.found and facet not in facets:
                unanswered.append(f"{drawn.key}: the verdict names the facet {facet}, which {FACETS} lacks")
                continue
            readings.append(Reading(key=drawn.key, round=number, source=drawn.source, hits=drawn.hits,
                                    found=verdict.found, facet=facet, steps=verdict.steps, why=verdict.why))
    return readings


def end_round(
    hunt: Hunt, state: HuntState, number: int, readings: list[Reading], blind: list[Reading], unanswered: list[str],
    blind_unanswered: list[str], failed: dict[str, str], retired: list[str],
) -> HuntState:
    """Add the readings of a round to the state, with its record and whether it brought a new finding."""
    new = new_findings(state, readings)
    state.quiet = 0 if new else state.quiet + 1
    state.round = number
    state.readings.update({reading.key: reading for reading in readings})
    state.blind.extend(blind)
    state.rounds.append(RoundRecord(
        round=number, reads=dict(Counter(reading.source for reading in readings)), blind=len(blind),
        unanswered=unanswered, blind_unanswered=blind_unanswered, failed=sorted(failed), retired=retired,
        found=sum(reading.found for reading in readings), new=new, quiet=state.quiet))
    write_file(hunt.workspace, f"rounds/{number}/verdicts.json",
               _json([reading.model_dump(mode="json") for reading in readings]))
    write_file(hunt.workspace, f"rounds/{number}/blind.json", _json([reading.model_dump(mode="json")
                                                                      for reading in blind]))
    return state


def apply(state: HuntState, proposal: Proposal, number: int) -> None:
    """Take the facets and probes of a proposal.

    A probe returned with the text it had keeps its counts, unless the ledger found it unproductive; a probe with a
    new text starts in this round; an active probe left out is retired. A facet with confirmed trajectories stays
    even when the proposal leaves it out.
    """
    previous = {probe.name: probe for probe in state.probes}
    reasons = {retirement.probe: retirement.reason for retirement in proposal.retired}
    probes = []
    for probe in proposal.probes:
        old = previous.get(probe.name)
        if old is not None and (old.kind, old.text) == (probe.kind, probe.text):
            if old.status != "unproductive":
                old = old.model_copy(update={"facet": probe.facet, "why": probe.why, "status": "active", "reason": ""})
            probes.append(old)
        else:
            probes.append(ProbeState(name=probe.name, facet=probe.facet, kind=probe.kind, text=probe.text,
                                     why=probe.why, since=number))
    proposed = {probe.name for probe in proposal.probes}
    for old in state.probes:
        if old.name in proposed:
            continue
        if old.status == "active":
            old = old.model_copy(update={"status": "retired", "reason": reasons.get(
                old.name, f"left out by the proposal of round {number}")})
        probes.append(old)
    names = {facet.name for facet in proposal.facets}
    found = state.found_facets()
    state.facets = [*proposal.facets, *(facet for facet in state.facets if facet.name not in names and
                                        facet.name in found)]
    state.probes = probes


def write_state(hunt: Hunt, state: HuntState) -> None:
    """Write what the agents read of the state: the facets, the probes with their text, the ledger and the findings."""
    workspace = hunt.workspace
    write_file(workspace, FACETS, _json([facet.model_dump(mode="json") for facet in state.facets]))
    write_file(workspace, PROBES, _json([
        {**probe.model_dump(mode="json", exclude={"text"}),
         "file": str(probe_path(workspace, probe).relative_to(workspace))} for probe in state.probes]))
    active = {probe_path(workspace, probe).name: probe for probe in state.active()}
    for path in [*(workspace / "probes").glob("*.sql"), *(workspace / "probes").glob("*.py")]:
        if path.name not in active:
            path.unlink()
    for file, probe in active.items():
        write_file(workspace, f"probes/{file}", probe.text)
    write_file(workspace, LEDGER, _json(ledger(state, hunt.spec, hunt.keys, hunt.clusters)))
    write_file(workspace, FINDINGS, _json([reading.model_dump(mode="json") for reading in state.readings.values()
                                           if reading.found]))


def closing_draws(hunt: Hunt, state: HuntState) -> list[Draw] | None:
    """At most `final_reads` of the trajectories the active probes hit and no round read, those of the most precise
    probes first, or every one of them; None when the hunt reads none at its close."""
    if hunt.spec.final_reads == 0:
        return None
    precision = {row["name"]: row["precision"] for row in probe_ledger(state, hunt.spec)}
    best: dict[str, float] = {}
    count: dict[str, int] = {}
    for probe, found in state.hits.items():
        for key in found:
            if key not in state.readings:
                value = precision[probe] if precision[probe] is not None else -1.0
                best[key] = max(best.get(key, -1.0), value)
                count[key] = count.get(key, 0) + 1
    chosen = sorted(best, key=lambda key: (-best[key], -count[key], key))
    if hunt.spec.final_reads != "all":
        chosen = chosen[:hunt.spec.final_reads]
    hits = pq.read_table(hunt.workspace / "rounds" / str(state.round) / "hits.parquet").to_pandas()
    return [Draw(key=key, source="final", hits=hits_of(hits, key)) for key in chosen]


def finish(
    hunt: Hunt, state: HuntState, final: list[Reading], blind: list[Reading], unanswered: list[str],
    blind_unanswered: list[str],
) -> dict[str, Any]:
    """Write the verdicts table and the report, keep the active probes for the next run, mark every trajectory with
    them, and return the report."""
    project = hunt.project
    state = state.model_copy(update={"readings": {**state.readings, **{r.key: r for r in final}},
                                     "blind": [*state.blind, *blind]})
    write_state(hunt, state)
    if final or blind:
        write_file(hunt.workspace, f"rounds/{FINAL}/verdicts.json", _json([r.model_dump(mode="json") for r in final]))
        write_file(hunt.workspace, f"rounds/{FINAL}/blind.json", _json([r.model_dump(mode="json") for r in blind]))
    write_verdicts(project, hunt.name, list(state.readings.values()))
    probes = probe_ledger(state, hunt.spec)
    keep_probes(project, hunt.name, state, probes)
    found = [reading for reading in state.readings.values() if reading.found]
    report = {
        "hunt": hunt.name, "run": int(hunt.directory.name),
        "directory": str(hunt.directory.relative_to(project.root)),
        "rounds": state.round, "converged": state.quiet >= hunt.spec.patience, "quiet": state.quiet,
        "patience": hunt.spec.patience, "max_rounds": hunt.spec.max_rounds,
        "found": len(found),
        "facets": [{**facet.model_dump(mode="json"), "found": sum(r.facet == facet.name for r in found)}
                   for facet in state.facets],
        "probes": probes,
        "coverage": coverage(state, hunt.keys, hunt.clusters),
        "calibration": calibration(state),
        "batches": batch_counts(hunt, state),
        "unverified": {probe: [key for key in keys if key not in state.readings]
                       for probe, keys in state.hits.items()},
        "final": {"read": len(final), "found": sum(reading.found for reading in final), "blind": len(blind),
                  "unanswered": unanswered, "blind_unanswered": blind_unanswered},
        "per_round": [record.model_dump(mode="json") for record in state.rounds],
        "marks": mark_trajectories(hunt, state),
    }
    (hunt.directory / REPORT).write_text(_json(report), encoding="utf-8")
    return report


def batch_counts(hunt: Hunt, state: HuntState) -> list[dict[str, Any]]:
    """Per batch and stratum of traj.yaml: the trajectories, how many were read and confirmed, by facet, and how many
    the active probes hit that no read covered."""
    members: dict[tuple[str, str], list[str]] = {}
    for key in hunt.keys:
        members.setdefault(hunt.batches[key], []).append(key)
    hit = state.hit_keys()
    rows = []
    for (batch, stratum), keys in sorted(members.items()):
        read = [state.readings[key] for key in keys if key in state.readings]
        found = [reading for reading in read if reading.found]
        rows.append({"batch": batch, "stratum": stratum, "trajectories": len(keys), "read": len(read),
                     "found": len(found), "facets": dict(sorted(Counter(r.facet for r in found).items())),
                     "unread_hits": sum(key in hit and key not in state.readings for key in keys)})
    return rows


def keep_probes(project: Project, name: str, state: HuntState, probes: list[dict[str, Any]]) -> None:
    """Write the facets and the active probes to `hunts/<name>/`, where the next run of the hunt starts from them and
    the marks read them."""
    counts = {row["name"]: row for row in probes}
    kept = [(KeptProbe(name=probe.name, facet=probe.facet, kind=probe.kind,
                       file=probe_path(Path(), probe).name, why=probe.why,
                       **{field: counts[probe.name][field] for field in ("hits", "read", "found", "precision")}),
             probe.text) for probe in state.active()]
    write_kept(project.hunts_dir / name, Kept(facets=state.facets, probes=kept))


def mark_trajectories(hunt: Hunt, state: HuntState) -> dict[str, Any]:
    """Enable the marks of the hunt in traj.yaml, compute them on every trajectory, and check that they are the
    positions the last round found; return the group, the facet of every feature, and per feature the marked
    trajectories.

    A probe finds its positions over every trajectory of the hunt at once in a round, and over one trajectory at a
    time in the marks, so a probe whose result depends on the other trajectories, such as one comparing a step with
    an average over the batch, marks something the round never read. The comparison is what catches it; a run whose
    marks differ keeps no probes enabled.
    """
    project, name = hunt.project, hunt.name
    group = marks_instance(name)

    edited = enable_marks(project, name)
    extract(edited, groups=[group])
    ((instance,),) = [found.instances for found in load_groups(edited, [group])]
    events = read_table(edited, "events").to_pandas()
    marked = events[events["rule"].isin(list(instance.original))]
    differing = _differing_probes(hunt, state, marked)
    if differing:
        edit_operators(edited, lambda operators: operators.__setitem__(
            slice(None), [use for use in operators if use.get("as") != group]))
        raise ConfigError(f"The probes {differing} of hunt {name} mark the trajectories of the project at other "
                          f"positions than they found in round {state.round}, so what they find in a trajectory "
                          f"depends on the other trajectories; the marks of the hunt are not enabled. Revise those "
                          f"probes in `hunts/{name}/` so that each reads one trajectory at a time, and run "
                          f"`traj hunt run {name}` again.")
    counts = marked.groupby("rule")["key"].nunique()
    return {"group": group, "features": dict(instance.original),
            "marked": {feature: int(counts.get(feature, 0)) for feature in instance.original}}


def _differing_probes(hunt: Hunt, state: HuntState, marked: pd.DataFrame) -> list[str]:
    """The probes whose marks on the trajectories of the hunt are not the positions the last round found."""
    hits = pq.read_table(hunt.workspace / "rounds" / str(state.round) / "hits.parquet").to_pandas()
    found = {(row.probe, row.key, int(row.step), row.note) for row in hits.itertuples()}
    inside = marked[marked["key"].isin(hunt.keys)]
    now = set()
    for row in inside.itertuples():
        probe, _, note = str(row.note).partition(": ")
        now.add((probe, row.key, int(row.step), note))
    return sorted({probe for probe, *_ in found ^ now})


def digests(workspace: Path, files: list[str]) -> dict[str, str]:
    return {file: hashlib.sha256((workspace / file).read_bytes()).hexdigest()
            for file in files if (workspace / file).is_file()}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=1)
