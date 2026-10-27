from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from aifn import Access, Refused, Returned
from ruamel.yaml import YAML

from traj_analyzer.calls import WorkspaceCalls
from traj_analyzer.extract import SubprocessWorkers, WorkerRunner, known_calls
from traj_analyzer.files import read_yaml
from traj_analyzer.hunting.draw import Draw, clusters_of, draw, hits_of
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
    Facet,
    Proposal,
    ProposeRequest,
    VerifyRequest,
    function_name,
    hunt_functions,
    propose_instruction,
)
from traj_analyzer.hunting.ledger import (
    HuntState,
    ProbeState,
    Reading,
    RoundRecord,
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
from traj_analyzer.layers.tables import column, table_path, with_key
from traj_analyzer.operators.catalog import load_groups
from traj_analyzer.project import ConfigError, Project
from traj_analyzer.runtime import mailbox
from traj_analyzer.vectorize import Matrix, build_frame

WORKSPACE = "workspace"
STATE = "state"
REPORT = "report.json"
CLUSTERS = "clusters.json"
PENDING = "pending.json"
"""A round of an agent session that waits for its verdicts (section 18.8 of the design)."""
FINAL = "final"
"""Directory of the closing check under `rounds/` of the workspace."""
FINDINGS_TABLE = "hunt_findings"
FINDINGS_SCHEMA = pa.schema([
    column("hunt", pa.string(), "The hunt that found the trajectory, `hunts/<hunt>.yaml`."),
    column("facet", pa.string(), "The kind of behaviour of the hunt's need the trajectory shows."),
    column("step", pa.int64(), "The first step where it shows."),
    column("steps", pa.list_(pa.int64()), "Every step where it shows, as the verdict cites them."),
    column("why", pa.string(), "What the verdict saw at those steps."),
    column("round", pa.int64(), "The round that read the trajectory; the round after the last for the closing "
                                "check."),
    column("source", pa.string(), "Why it was read: hit, a probe hit it; coverage, drawn from the cluster read "
                                  "least; random, drawn at random from the trajectories no probe hits; final, read by "
                                  "the closing check."),
    column("probes", pa.list_(pa.string()), "The probes that hit the trajectory when it was read."),
])


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

    @property
    def workspace(self) -> Path:
        return self.directory / WORKSPACE

    @property
    def model(self) -> str:
        return hunt_model(self.project, self.spec)


Answers = list[tuple[list[Case], dict[str, CaseVerdict], str]]
"""The verdicts of a round per group of cases, each group with the reason a case of it without a verdict has none."""


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
    return Hunt(name=name, spec=spec, project=project, directory=directory, keys=keys, matrix=matrix,
                clusters=_clusters(directory, keys, matrix, spec))


def hunt(
    project: Project, name: str, *, max_rounds: int | None = None, workers: int | None = None,
    runner: WorkerRunner | None = None,
) -> dict[str, Any]:
    """Search the trajectories for what the hunt asks for with the aifn functions of the hunt, round after round,
    until `patience` rounds in a row find nothing new or `max_rounds` is reached; then read what the active probes
    hit and no round read, write the findings, keep the probes, and return the report.

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
    calls = WorkspaceCalls(project=project, runner=runner, workers=workers, box=box, known=known_calls(box),
                           functions=hunt_functions(project, name, found.spec), workspace=found.workspace)
    calls.placed.update(prepare(project, found.workspace, found.spec, found.keys))
    state = resume(found.directory) or start(project, name)
    while not over(found.spec, state):
        state = run_round(found, calls, state)
        save_state(found, state)
    draws = closing_draws(found, state)
    if draws is None:
        return finish(found, state, [], [])
    cases = write_cases(found, draws, f"rounds/{FINAL}")
    return finish(found, state, *ask_verdicts(found, calls, state, draws, cases, state.round + 1))


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


def start(project: Project, name: str) -> HuntState:
    """The state a first round starts from: the facets and probes the hunt kept from its last run, when it has."""
    kept = project.hunts_dir / name
    if not (kept / "probes.yaml").is_file():
        return HuntState()
    facets = [Facet.model_validate(facet) for facet in read_yaml(kept / "facets.yaml")]
    probes = [ProbeState(name=entry["name"], facet=entry["facet"], kind=entry["kind"], why=entry["why"], since=0,
                         text=(kept / entry["file"]).read_text(encoding="utf-8"))
              for entry in read_yaml(kept / "probes.yaml")]
    return HuntState(facets=facets, probes=probes)


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
    proposal = calls.alone(function_name(PROPOSE, hunt.name), propose_instruction(hunt.spec, number), request,
                           leftovers)
    failed, retired, draws = begin_round(hunt, state, proposal, number)
    cases = write_cases(hunt, draws, f"rounds/{number}")
    readings, unanswered = ask_verdicts(hunt, calls, state, draws, cases, number)
    return end_round(hunt, state, number, readings, unanswered, failed, sorted(retired))


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


def write_cases(hunt: Hunt, draws: list[Draw], directory: str) -> list[Case]:
    """Write every drawn trajectory whole as a case file under `directory`."""
    return [Case(file=write_case(hunt.project, hunt.workspace, f"{directory}/cases", drawn.key), hits=drawn.hits)
            for drawn in draws]


def groups(hunt: Hunt, cases: list[Case]) -> list[list[Case]]:
    """The cases in the groups one reader reads, `cases_per_call` each."""
    size = hunt.spec.cases_per_call
    return [cases[i:i + size] for i in range(0, len(cases), size)]


def ask_verdicts(
    hunt: Hunt, calls: WorkspaceCalls, state: HuntState, draws: list[Draw], cases: list[Case], number: int
) -> tuple[list[Reading], list[str]]:
    """Read the cases with the aifn function of the hunt, one call per group, all calls side by side."""
    files = digests(hunt.workspace, [NEED, FACETS, SCENARIO])
    chunks = groups(hunt, cases)
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
    return record(state, draws, cases, answers, number)


def record(
    state: HuntState, draws: list[Draw], cases: list[Case], answers: Answers, number: int
) -> tuple[list[Reading], list[str]]:
    """What every case showed, with the trajectories that got no verdict. A facet a verdict reports as new joins the
    facets of `state`."""
    drawn_by_file = {case.file: drawn for case, drawn in zip(cases, draws, strict=True)}
    readings: list[Reading] = []
    unanswered: list[str] = []
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
                if facet not in facets:
                    state.facets.append(verdict.new_facet)
                    facets.add(facet)
            elif verdict.found and facet not in facets:
                unanswered.append(f"{drawn.key}: the verdict names the facet {facet}, which {FACETS} lacks")
                continue
            readings.append(Reading(key=drawn.key, round=number, source=drawn.source, hits=drawn.hits,
                                    found=verdict.found, facet=facet, steps=verdict.steps, why=verdict.why))
    return readings, unanswered


def end_round(
    hunt: Hunt, state: HuntState, number: int, readings: list[Reading], unanswered: list[str],
    failed: dict[str, str], retired: list[str],
) -> HuntState:
    """Add the readings of a round to the state, with its record and whether it brought a new finding."""
    new = new_findings(state, readings)
    state.quiet = 0 if new else state.quiet + 1
    state.round = number
    state.readings.update({reading.key: reading for reading in readings})
    state.rounds.append(RoundRecord(
        round=number, reads=dict(Counter(reading.source for reading in readings)), unanswered=unanswered,
        failed=sorted(failed), retired=retired, found=sum(reading.found for reading in readings), new=new,
        quiet=state.quiet))
    write_file(hunt.workspace, f"rounds/{number}/verdicts.json",
               _json([reading.model_dump(mode="json") for reading in readings]))
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
    probes first; None when the hunt reads none at its close."""
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
    chosen = sorted(best, key=lambda key: (-best[key], -count[key], key))[:hunt.spec.final_reads]
    hits = pq.read_table(hunt.workspace / "rounds" / str(state.round) / "hits.parquet").to_pandas()
    return [Draw(key=key, source="final", hits=hits_of(hits, key)) for key in chosen]


def finish(hunt: Hunt, state: HuntState, final: list[Reading], unanswered: list[str]) -> dict[str, Any]:
    """Write the findings table and the report, keep the active probes for the next run, and return the report."""
    project = hunt.project
    state = state.model_copy(update={"readings": {**state.readings, **{r.key: r for r in final}}})
    write_state(hunt, state)
    write_findings(project, hunt.name, list(state.readings.values()))
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
        "unverified": {probe: [key for key in keys if key not in state.readings]
                       for probe, keys in state.hits.items()},
        "final": {"read": len(final), "found": sum(reading.found for reading in final), "unanswered": unanswered},
        "per_round": [record.model_dump(mode="json") for record in state.rounds],
    }
    (hunt.directory / REPORT).write_text(_json(report), encoding="utf-8")
    return report


def write_findings(project: Project, name: str, readings: list[Reading]) -> None:
    """Replace the hunt's rows of the findings table with the trajectories it confirmed."""
    rows = [{"key": reading.key, "hunt": name, "facet": reading.facet, "step": min(reading.steps),
             "steps": reading.steps, "why": reading.why, "round": reading.round, "source": reading.source,
             "probes": sorted({hit.probe for hit in reading.hits})}
            for reading in sorted(readings, key=lambda r: r.key) if reading.found]
    schema = with_key(FINDINGS_SCHEMA)
    table = pa.Table.from_pylist(rows, schema=schema)
    path = table_path(project, FINDINGS_TABLE)
    if path.is_file():
        kept = pq.read_table(path, schema=schema)
        table = pa.concat_tables([kept.filter(pc.not_equal(kept["hunt"], name)), table])
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def keep_probes(project: Project, name: str, state: HuntState, probes: list[dict[str, Any]]) -> None:
    """Write the facets and the active probes to `hunts/<name>/`, where the next run of the hunt starts from them."""
    directory = project.hunts_dir / name
    directory.mkdir(parents=True, exist_ok=True)
    active = {probe_path(directory, probe).name: probe for probe in state.active()}
    for path in [*directory.glob("*.sql"), *directory.glob("*.py")]:
        if path.name not in active:
            path.unlink()
    counts = {row["name"]: row for row in probes}
    entries = []
    for file, probe in active.items():
        (directory / file).write_text(probe.text, encoding="utf-8")
        row = counts[probe.name]
        entries.append({"name": probe.name, "facet": probe.facet, "kind": probe.kind, "file": file, "why": probe.why,
                        "hits": row["hits"], "read": row["read"], "found": row["found"],
                        "precision": row["precision"]})
    yaml = YAML()
    yaml.width = 4096
    yaml.dump([facet.model_dump(mode="json") for facet in state.facets], directory / "facets.yaml")
    yaml.dump(entries, directory / "probes.yaml")


def digests(workspace: Path, files: list[str]) -> dict[str, str]:
    return {file: hashlib.sha256((workspace / file).read_bytes()).hexdigest()
            for file in files if (workspace / file).is_file()}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=1)
