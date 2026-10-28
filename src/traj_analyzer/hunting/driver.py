from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from aifn import Access, Refused, Returned
from pydantic import BaseModel

from traj_analyzer.batches import assign
from traj_analyzer.calls import WorkspaceCalls
from traj_analyzer.extract import SubprocessWorkers, WorkerRunner, extract
from traj_analyzer.hunting.draw import Draw, blind_draws, clusters_of, draw, hits_of
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
from traj_analyzer.hunting.kept import OPERATOR, Kept, KeptProbe, instance_name, read_kept, write_kept
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
from traj_analyzer.hunting.probes import HITS_SCHEMA, probe_path, run_probes
from traj_analyzer.hunting.spec import HuntSpec, hunt_model, load_hunt
from traj_analyzer.hunting.workspace import prepare, write_case, write_file
from traj_analyzer.ingest import load_index
from traj_analyzer.layers.tables import column, read_table, table_path, with_key
from traj_analyzer.operators.catalog import edit_operators, load_groups
from traj_analyzer.project import ConfigError, Project
from traj_analyzer.runtime import known_calls, mailbox
from traj_analyzer.vectorize import Matrix, build_frame

WORKSPACE = "workspace"
STATE = "state"
REPORT = "report.json"
CLUSTERS = "clusters.json"
TIMELINE = "timeline.jsonl"
"""When every step of a run was opened and done, one JSON line each."""
PENDING = "pending.json"
"""A round of an agent session that waits for its verdicts (section 18.9 of the design)."""
FINAL = "final"
"""Directory of the closing check under `rounds/` of the workspace."""
VERDICTS_TABLE = "hunt_verdicts"
VERDICTS_SCHEMA = pa.schema([
    column("hunt", pa.string(), "The hunt that read the trajectory, `hunts/<hunt>.yaml`."),
    column("found", pa.bool_(), "Whether the trajectory shows what the hunt's need describes."),
    column("facet", pa.string(), "When found, the kind of behaviour of the need it shows."),
    column("step", pa.int64(), "When found, the first step where it shows."),
    column("steps", pa.list_(pa.int64()), "When found, every step where it shows, as the verdict cites them."),
    column("why", pa.string(), "What the verdict saw, naming the steps."),
    column("round", pa.int64(), "The round that read the trajectory; the round after the last for the closing "
                                "check."),
    column("source", pa.string(), "Why it was read: hit, a probe hit it; coverage, drawn from the cluster read "
                                  "least; random, drawn at random from the trajectories no probe hits; final, read by "
                                  "the closing check."),
    column("probes", pa.list_(pa.string()), "The probes that hit the trajectory when it was read."),
])
Step = Literal["propose", "read", "close"]


@dataclass(frozen=True)
class Hunt:
    """What every round of one run of a hunt shares."""

    name: str
    spec: HuntSpec
    project: Project
    directory: Path
    keys: list[str]
    matrix: Matrix
    clusters: dict[str, int]
    batches: dict[str, tuple[str, str]]
    """Per trajectory, its batch and stratum under `batches` and `strata` of traj.yaml."""

    @property
    def workspace(self) -> Path:
        return self.directory / WORKSPACE

    @property
    def model(self) -> str:
        return hunt_model(self.project, self.spec)


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


def open_hunt(project: Project, name: str, *, create: bool, max_rounds: int | None = None) -> Hunt:
    """The run of a hunt with what its rounds share; `create` starts a new run when the last one is finished."""
    spec = load_hunt(project, name)
    if max_rounds is not None:
        spec = HuntSpec.model_validate({**spec.model_dump(), "max_rounds": max_rounds})
    rows = sorted(load_index(project, None if spec.datasets == "all" else spec.datasets), key=lambda row: row.key)
    if not rows:
        raise ConfigError(f"Hunt {name} has no trajectories; run `traj ingest`")
    if not table_path(project, "steps").is_file():
        raise ConfigError("No tables yet; run `traj extract`")
    frame = build_frame(project, load_groups(project), rows, project.config.sampling.vector_length)
    if not frame.columns:
        raise ConfigError("No extracted features tell the trajectories apart; run `traj extract`")
    keys = [row.key for row in rows]
    directory = run_directory(project, name) if create else open_run(project, name)
    matrix = frame.matrix(dict.fromkeys(frame.columns, 1.0))
    batch = assign(project, frame, rows, project.config.batches)
    stratum = assign(project, frame, rows, project.config.strata)
    return Hunt(name=name, spec=spec, project=project, directory=directory, keys=keys, matrix=matrix,
                clusters=_clusters(directory, keys, matrix, spec),
                batches={key: (str(batch[key]), str(stratum[key])) for key in keys})


def hunt(
    project: Project, name: str, *, max_rounds: int | None = None, workers: int | None = None,
    runner: WorkerRunner | None = None,
) -> dict[str, Any]:
    """Search the trajectories for what the hunt asks for with the aifn functions of the hunt, round after round,
    until `patience` rounds in a row find nothing new or `max_rounds` is reached; then read what the active probes
    hit and no round read, write the verdicts, keep the probes, mark every trajectory with them, and return the
    report.

    A run that stopped midway continues from the state of its last finished round, and the calls it had made are
    answered from the mailbox.
    """
    runner = runner or SubprocessWorkers()
    runner.check(project)
    found = open_hunt(project, name, create=True, max_rounds=max_rounds)
    if (found.directory / PENDING).is_file():
        raise ConfigError(f"Round of {found.directory} waits for the verdicts of an agent session; hand them in "
                          f"with `traj hunt verdicts {name}`")
    box = mailbox(project)
    calls = WorkspaceCalls(project=project, runner=runner, workers=workers or found.spec.workers, box=box,
                           known=known_calls(box), functions=hunt_functions(project, name, found.spec),
                           workspace=found.workspace)
    calls.placed.update(prepare(project, found.workspace, name, found.spec, found.keys))
    state = resume(found.directory) or start(project, name)
    while not over(found.spec, state):
        state = run_round(found, calls, state)
        save_state(found, state)
    draws = closing_draws(found, state)
    if not draws:
        return finish(found, state, [], [], [])
    note(found, state.round + 1, "close", "open")
    reads = write_reads(found, draws, f"rounds/{FINAL}", state.round + 1)
    readings, blind, unanswered = ask_verdicts(found, calls, state, reads, state.round + 1)
    note(found, state.round + 1, "close", "done")
    return finish(found, state, readings, blind, unanswered)


def over(spec: HuntSpec, state: HuntState) -> bool:
    """Whether the search has ended: `patience` rounds in a row without a new finding, or `max_rounds` rounds."""
    return state.quiet >= spec.patience or state.round >= spec.max_rounds


def run_directory(project: Project, name: str) -> Path:
    """The directory of the run to continue: the last one when it has no report yet, a new one otherwise."""
    root = project.hunting_dir / name
    numbers = sorted(int(p.name) for p in root.glob("[0-9]*") if p.is_dir() and p.name.isdigit())
    number = numbers[-1] if numbers else 1
    if (root / str(number) / REPORT).is_file():
        number += 1
    directory = root / str(number)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def last_run(project: Project, name: str) -> Path | None:
    root = project.hunting_dir / name
    numbers = sorted(int(p.name) for p in root.glob("[0-9]*") if p.is_dir() and p.name.isdigit())
    return root / str(numbers[-1]) if numbers else None


def open_run(project: Project, name: str) -> Path:
    """The run of the hunt in progress."""
    directory = last_run(project, name)
    if directory is None or (directory / REPORT).is_file():
        raise ConfigError(f"Hunt {name} has no run in progress; start one with `traj hunt start {name}`")
    return directory


def _clusters(directory: Path, keys: list[str], matrix: Matrix, spec: HuntSpec) -> dict[str, int]:
    """The clusters of the run, computed when it starts and kept for every round of it."""
    path = directory / CLUSTERS
    if path.is_file():
        clusters: dict[str, int] = json.loads(path.read_text(encoding="utf-8"))
        if sorted(clusters) != keys:
            raise ConfigError(f"The trajectories of the hunt changed since its run in {directory} started; remove "
                              f"that directory to start the run again")
        return clusters
    clusters = clusters_of(keys, matrix, spec.clusters, spec.seed)
    path.write_text(json.dumps(clusters, indent=1, sort_keys=True), encoding="utf-8")
    return clusters


def resume(directory: Path) -> HuntState | None:
    """The state after the last finished round of a run; None before its first round finishes."""
    states = sorted((directory / STATE).glob("*.json"), key=lambda path: int(path.stem))
    if not states:
        return None
    return HuntState.model_validate_json(states[-1].read_text(encoding="utf-8"))


def save_state(hunt: Hunt, state: HuntState) -> None:
    (hunt.directory / STATE).mkdir(exist_ok=True)
    (hunt.directory / STATE / f"{state.round}.json").write_text(state.model_dump_json(indent=1), encoding="utf-8")


def note(hunt: Hunt, number: int, step: Step, event: Literal["open", "done"]) -> None:
    """Record in the timeline of the run that a step of a round was opened or done."""
    line = {"round": number, "step": step, "event": event, "at": datetime.now(UTC).isoformat(timespec="seconds")}
    with (hunt.directory / TIMELINE).open("a", encoding="utf-8") as out:
        out.write(json.dumps(line) + "\n")


def start(project: Project, name: str) -> HuntState:
    """The state a first round starts from: the facets and probes the hunt kept from its last run, when it has."""
    kept = read_kept(project.hunts_dir / name)
    if kept is None:
        return HuntState()
    probes = [ProbeState(name=entry.name, facet=entry.facet, kind=entry.kind, why=entry.why, since=0, text=text)
              for entry, text in kept.probes]
    return HuntState(facets=kept.facets, probes=probes)


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
    note(hunt, number, "propose", "open")
    proposal = calls.alone(function_name(PROPOSE, hunt.name), propose_instruction(hunt.spec, number), request,
                           leftovers)
    write_file(hunt.workspace, f"rounds/{number}/proposal.json", proposal.model_dump_json(indent=1))
    note(hunt, number, "propose", "done")
    failed, retired, draws = begin_round(hunt, state, proposal, number)
    note(hunt, number, "read", "open")
    reads = write_reads(hunt, draws, f"rounds/{number}", number)
    readings, blind, unanswered = ask_verdicts(hunt, calls, state, reads, number)
    note(hunt, number, "read", "done")
    return end_round(hunt, state, number, readings, blind, unanswered, failed, sorted(retired))


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
) -> tuple[list[Reading], list[Reading], list[str]]:
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
) -> tuple[list[Reading], list[Reading], list[str]]:
    """What every case showed, the blind reads apart, with the trajectories that got no verdict. A facet a verdict
    with the hits reports as new joins the facets of `state`; one a blind verdict reports stays out."""
    unanswered: list[str] = []
    readings = _readings(state, reads.draws, reads.files, answers, number, unanswered, adds=True)
    blind = _readings(state, reads.blind, reads.files, blind_answers, number, unanswered, adds=False)
    return readings, blind, unanswered


def _readings(
    state: HuntState, draws: list[Draw], files: dict[str, str], answers: Answers, number: int, unanswered: list[str],
    adds: bool,
) -> list[Reading]:
    drawn_by_file = {files[drawn.key]: drawn for drawn in draws}
    readings: list[Reading] = []
    facets = {facet.name for facet in state.facets}
    label = "" if adds else "blind read of "
    for chunk, verdicts, missing in answers:
        for case in chunk:
            drawn = drawn_by_file[case.file]
            verdict = verdicts.get(case.file)
            if verdict is None:
                unanswered.append(f"{label}{drawn.key}: {missing}")
                continue
            facet = verdict.facet
            if verdict.found and facet == NEW:
                assert verdict.new_facet is not None
                facet = verdict.new_facet.name
                if adds and facet not in facets:
                    state.facets.append(verdict.new_facet)
                    facets.add(facet)
            elif verdict.found and facet not in facets:
                unanswered.append(f"{label}{drawn.key}: the verdict names the facet {facet}, which {FACETS} lacks")
                continue
            readings.append(Reading(key=drawn.key, round=number, source=drawn.source, hits=drawn.hits,
                                    found=verdict.found, facet=facet, steps=verdict.steps, why=verdict.why))
    return readings


def end_round(
    hunt: Hunt, state: HuntState, number: int, readings: list[Reading], blind: list[Reading], unanswered: list[str],
    failed: dict[str, str], retired: list[str],
) -> HuntState:
    """Add the readings of a round to the state, with its record and whether it brought a new finding."""
    new = new_findings(state, readings)
    state.quiet = 0 if new else state.quiet + 1
    state.round = number
    state.readings.update({reading.key: reading for reading in readings})
    state.blind.extend(blind)
    state.rounds.append(RoundRecord(
        round=number, reads=dict(Counter(reading.source for reading in readings)), blind=len(blind),
        unanswered=unanswered, failed=sorted(failed), retired=retired, found=sum(reading.found for reading in readings),
        new=new, quiet=state.quiet))
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
    for path in (workspace / "probes").iterdir():
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
    hunt: Hunt, state: HuntState, final: list[Reading], blind: list[Reading], unanswered: list[str]
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
        "batches": by_batch(hunt, state),
        "unverified": {probe: [key for key in keys if key not in state.readings]
                       for probe, keys in state.hits.items()},
        "final": {"read": len(final), "found": sum(reading.found for reading in final), "blind": len(blind),
                  "unanswered": unanswered},
        "per_round": [record.model_dump(mode="json") for record in state.rounds],
        "marks": mark_trajectories(project, hunt.name),
    }
    (hunt.directory / REPORT).write_text(_json(report), encoding="utf-8")
    return report


def by_batch(hunt: Hunt, state: HuntState) -> list[dict[str, Any]]:
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


def write_verdicts(project: Project, name: str, readings: list[Reading]) -> None:
    """Replace the hunt's rows of the verdicts table with every trajectory it read."""
    rows = [{"key": reading.key, "hunt": name, "found": reading.found, "facet": reading.facet,
             "step": min(reading.steps) if reading.steps else None, "steps": reading.steps, "why": reading.why,
             "round": reading.round, "source": reading.source,
             "probes": sorted({hit.probe for hit in reading.hits})}
            for reading in sorted(readings, key=lambda r: r.key)]
    schema = with_key(VERDICTS_SCHEMA)
    table = pa.Table.from_pylist(rows, schema=schema)
    path = table_path(project, VERDICTS_TABLE)
    if path.is_file():
        kept = pq.read_table(path, schema=schema)
        table = pa.concat_tables([kept.filter(pc.not_equal(kept["hunt"], name)), table])
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def keep_probes(project: Project, name: str, state: HuntState, probes: list[dict[str, Any]]) -> None:
    """Write the facets and the active probes to `hunts/<name>/`, where the next run of the hunt starts from them and
    the marks read them."""
    counts = {row["name"]: row for row in probes}
    kept = [(KeptProbe(name=probe.name, facet=probe.facet, kind=probe.kind,
                       file=probe_path(Path(), probe).name, why=probe.why,
                       **{field: counts[probe.name][field] for field in ("hits", "read", "found", "precision")}),
             probe.text) for probe in state.active()]
    write_kept(project.hunts_dir / name, Kept(facets=state.facets, probes=kept))


def mark_trajectories(project: Project, name: str) -> dict[str, Any]:
    """Enable the marks of the hunt in traj.yaml, unless they are enabled, and compute them on every trajectory;
    return the group, the facet of every feature, and per feature the marked trajectories."""
    group = instance_name(name)

    def enable(operators: list[Any]) -> None:
        if not any(operator.get("as") == group for operator in operators):
            operators.append({"use": OPERATOR, "as": group, "params": {"hunt": name}})

    edited = edit_operators(project, enable)
    extract(edited, groups=[group])
    ((instance,),) = [found.instances for found in load_groups(edited, [group])]
    events = read_table(edited, "events").to_pandas()
    marked = events[events["rule"].isin(list(instance.original))].groupby("rule")["key"].nunique()
    return {"group": group, "features": dict(instance.original),
            "marked": {feature: int(marked.get(feature, 0)) for feature in instance.original}}


def digests(workspace: Path, files: list[str]) -> dict[str, str]:
    return {file: hashlib.sha256((workspace / file).read_bytes()).hexdigest()
            for file in files if (workspace / file).is_file()}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=1)
