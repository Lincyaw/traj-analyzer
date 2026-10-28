from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from aifn import Handle, Mailbox, Refused, Returned, Workspace

from traj_analyzer.batches import assign
from traj_analyzer.features.spec import ExtractRequest, value_validator
from traj_analyzer.features.table import KEY, FeatureRow, read_group, status_column, write_group
from traj_analyzer.harness import discover_harnesses, harness_for
from traj_analyzer.ingest import (
    IndexRow,
    load_index,
    load_trajectory,
    render_turns,
    render_window,
    turn_units,
    window_units,
)
from traj_analyzer.layers import (
    EVENTS,
    EVIDENCE,
    JUDGMENTS,
    STEPS,
    TRAJECTORIES,
    TableWriter,
    compute_tables,
    enabled_layers,
    schemas_of,
)
from traj_analyzer.operators.base import Aggregate, Evidenced, FeatureSpec
from traj_analyzer.operators.catalog import Group, Instance, load_groups, runtime_groups
from traj_analyzer.preflight import CHECKED, preflight
from traj_analyzer.project import ConfigError, Project
from traj_analyzer.runtime import engine_env, issue_call, known_calls, mailbox, model_name, submitting_function
from traj_analyzer.schema import Trajectory
from traj_analyzer.vectorize import build_frame


@dataclass
class GroupReport:
    group: str
    total: int = 0
    cached: int = 0
    ok: int = 0
    not_applicable: int = 0
    refused: int = 0
    failed: int = 0
    pending: int = 0
    invalid: int = 0
    calls: int = 0
    """Model calls the group takes for these trajectories: one per trajectory, or one per window."""
    errors: list[str] = field(default_factory=list)


def select_rows(
    project: Project, datasets: list[str] | None, keys: list[str] | None, limit: int | None
) -> list[IndexRow]:
    rows = load_index(project, datasets)
    if keys:
        wanted = set(keys)
        unknown = sorted(wanted - {row.key for row in rows})
        if unknown:
            raise ConfigError(f"Unknown trajectory keys: {unknown}")
        rows = [row for row in rows if row.key in wanted]
    return rows[:limit] if limit else rows


FUNCTIONS_VARIABLE = "TRAJ_FUNCTIONS"
"""Environment variable naming the aifn functions a worker run processes, comma-separated; every function
of the project when unset."""
MAILBOX_VARIABLE = "TRAJ_MAILBOX"
"""Environment variable naming the mailbox a worker run works from; the project's mailbox when unset."""


class WorkerRunner(Protocol):
    def check_credentials(self, project: Project) -> None: ...

    def run(self, project: Project, count: int, functions: frozenset[str] = frozenset()) -> None: ...


class SubprocessWorkers:
    """Run aifn worker processes that exit once the queue is empty.

    With `functions`, the workers take only the calls of those functions, so a run processes its own calls and
    leaves the calls other runs left pending alone. Before the first workers of a project start in this process,
    one call checks that a worker can run a call at all.
    """

    def check_credentials(self, project: Project) -> None:
        engine_env(project)

    def run(self, project: Project, count: int, functions: frozenset[str] = frozenset()) -> None:
        if project.root not in CHECKED:
            preflight(project, self.spawn)
        self.spawn(project, count, functions)

    def spawn(self, project: Project, count: int, functions: frozenset[str] = frozenset(),
              mailbox: Path | None = None) -> None:
        env = {**os.environ, "TRAJ_PROJECT": str(project.root)}
        if functions:
            env[FUNCTIONS_VARIABLE] = ",".join(sorted(functions))
        if mailbox is not None:
            env[MAILBOX_VARIABLE] = str(mailbox)
        command = [sys.executable, "-m", "aifn", "worker", "traj_analyzer.worker:make_worker",
                   "--once", "--id"]
        processes = [subprocess.Popen([*command, f"traj-{n}"], env=env, stdout=sys.stderr)
                     for n in range(count)]
        codes = [process.wait() for process in processes]
        if any(codes):
            raise RuntimeError(f"aifn workers exited with codes {codes}")


def extract(
    project: Project,
    *,
    groups: list[str] | None = None,
    datasets: list[str] | None = None,
    keys: list[str] | None = None,
    limit: int | None = None,
    workers: int | None = None,
    dry_run: bool = False,
    runner: WorkerRunner | None = None,
) -> list[GroupReport]:
    """Run the parsing layers, compute code groups and collect judgment groups for the selected trajectories.

    Code features and tables are recomputed on every run. Judgment calls whose request and instruction are unchanged
    are answered from the aifn mailbox, so a rerun only calls the model for new or changed trajectories and
    operators.
    """
    rows = select_rows(project, datasets, keys, limit)
    return run_groups(project, load_groups(project, groups), rows, workers=workers, dry_run=dry_run, runner=runner)


Units = dict[str, dict[str, list[tuple[int, int]]]]
"""Per judgment group over turns or windows, per trajectory key, the units: first and last step of each."""


@dataclass
class Scan:
    reports: list[GroupReport]
    applies: dict[str, dict[str, dict[str, bool]]]
    """Per judgment group and trajectory, whether each of its instances applies."""
    units: Units


def run_groups(
    project: Project,
    selected: list[Group],
    rows: list[IndexRow],
    *,
    workers: int | None = None,
    dry_run: bool = False,
    runner: WorkerRunner | None = None,
    second: bool = False,
    directory: Path | None = None,
) -> list[GroupReport]:
    """Run these groups on these trajectories and write their tables to `directory`, the feature table by default.

    With `second`, judgment groups are measured by `evaluation.model` through functions of their own, so the
    answers of the two models never share a cache entry. The tables of the project are written only on a run into
    the feature table by the first model.
    """
    runner = runner or SubprocessWorkers()
    code = [g for g in selected if g.is_code]
    judged = [g for g in selected if not g.is_code]
    if judged and not dry_run:
        runner.check_credentials(project)
    writes_tables = not dry_run and directory is None and not second
    scan = _scan(project, code, judged, rows, dry_run, directory, writes_tables)
    if not judged:
        return scan.reports
    box = mailbox(project)
    known = known_calls(box)
    issued: dict[str, dict[str, list[Handle[Any]]]] = {}
    reports: dict[str, GroupReport] = {}
    inside = in_scope(project, [f for g in judged for f in g.features], rows)
    sampled = _sampled(project, judged, rows)
    for group in judged:
        # A trajectory outside the scope of every feature of the group, or outside the sample, is not sent.
        asked = {row.key for row in rows
                 if any(f.name not in inside or row.key in inside[f.name] for f in group.features)}
        if group.name in sampled:
            asked &= sampled[group.name]
        units = scan.units.get(group.name, {})
        callable_rows = [row for row in rows if row.key in asked and any(scan.applies[group.name][row.key].values())
                         and (group.reads_file or units.get(row.key))]
        issued[group.name] = _issue(project, box, known, group, callable_rows, units, dry_run, second)
        cached = sum(all(h.poll() is not None for h in handles) for handles in issued[group.name].values())
        reports[group.name] = GroupReport(group.name, total=len(rows), cached=cached,
                                          not_applicable=len(rows) - len(callable_rows),
                                          calls=sum(len(h) for h in issued[group.name].values()))
    if not dry_run and any(len(issued[g.name]) > reports[g.name].cached for g in judged):
        functions = frozenset(f"eval-{g.name}" if second else g.function_name for g in judged)
        runner.run(project, workers or project.config.extract.workers, functions)
    writer = TableWriter(project, {"judgments": JUDGMENTS, "events": EVENTS, "evidence": EVIDENCE}) \
        if writes_tables else None
    for group in judged:
        out = _collect(group, rows, issued[group.name], scan.applies[group.name], scan.units.get(group.name, {}),
                       reports[group.name], writer)
        out = [FeatureRow(key=row.key, feature=row.feature, detail="outside the scope of the feature")
               if row.feature in inside and row.key not in inside[row.feature] else row for row in out]
        if not dry_run:
            write_group(project, group, out, directory)
    if writer is not None:
        writer.finish({
            "judgments": ("group", {g.name for g in judged}),
            "events": ("rule", {f.name for g in judged for f in g.unit_outputs if f.event}),
            "evidence": ("feature", {f.name for g in judged for f in g.features}),
        })
    return [*scan.reports, *reports.values()]


def in_scope(project: Project, features: list[FeatureSpec], rows: list[IndexRow]) -> dict[str, set[str]]:
    """Per feature with a `scope`, the trajectories among `rows` that meet it.

    The condition is evaluated over the columns the enabled code groups have in the feature table, so those groups
    are extracted before a group with scoped features.
    """
    scoped = [feature for feature in features if feature.scope]
    if not scoped:
        return {}
    query = _code_frame(project, rows).query
    return {feature.name: set(query.query(feature.scope, engine="python").index) for feature in scoped}


def _code_frame(project: Project, rows: list[IndexRow]) -> Any:
    code = [group for group in load_groups(project) if group.is_code]
    return build_frame(project, code, rows, project.config.sampling.vector_length)


def _sampled(project: Project, groups: list[Group], rows: list[IndexRow]) -> dict[str, set[str]]:
    """Per group with a `sample`, the trajectories drawn: `per_batch` at random from every batch."""
    sampling = [group for group in groups if group.sample is not None]
    if not sampling:
        return {}
    batches = assign(project, _code_frame(project, rows), rows, project.config.batches)
    chosen: dict[str, set[str]] = {}
    for group in sampling:
        assert group.sample is not None
        rng = np.random.default_rng([group.sample.seed, int(hashlib.sha256(group.name.encode()).hexdigest()[:8], 16)])
        keys: set[str] = set()
        for _, members in sorted(batches.groupby(batches).groups.items()):
            ordered = sorted(members)
            keys.update(rng.choice(ordered, size=min(group.sample.per_batch, len(ordered)), replace=False).tolist())
        chosen[group.name] = keys
    return chosen


def coverage(project: Project) -> dict[str, dict[str, int]]:
    """Per enabled group, over every ingested trajectory: how many have rows in the feature table, counted by status,
    and how many have none. A trajectory whose features differ in status counts under its first status other than ok."""
    keys = {row.key for row in load_index(project)}
    report = {}
    for group in load_groups(project):
        table = read_group(project, group)
        columns = [table[status_column(f.name)].to_pylist() for f in group.features]
        statuses: dict[str, str] = {}
        for key, row in zip(table[KEY].to_pylist(), zip(*columns, strict=True), strict=True):
            present = [status for status in row if status is not None]
            if key in keys and present:
                statuses[key] = next((status for status in present if status != "ok"), "ok")
        report[group.name] = {"extracted": len(statuses), "missing": len(keys) - len(statuses),
                              **Counter(statuses.values())}
    return report


def _scan(
    project: Project, code: list[Group], judged: list[Group], rows: list[IndexRow], dry_run: bool,
    directory: Path | None, writes_tables: bool,
) -> Scan:
    """Load each trajectory once: run the layers, compute every code group, decide which judgment operators apply,
    and cut the units of the groups over turns and windows."""
    reports = {g.name: GroupReport(g.name, total=len(rows)) for g in code}
    applies: dict[str, dict[str, dict[str, bool]]] = {g.name: {} for g in judged}
    units: Units = {g.name: {} for g in judged if not g.reads_file}
    out: dict[str, list[FeatureRow]] = {g.name: [] for g in code}
    writer = None
    if writes_tables and code:
        layers = enabled_layers(project, {table for group in code for table in group.tables})
        schemas = {"steps": STEPS, "trajectories": TRAJECTORIES, **schemas_of(layers), "evidence": EVIDENCE,
                   "events": EVENTS}
        writer = TableWriter(project, schemas)
    for key, applied, computed, tables, cut in _scan_results(project, code, judged, rows, dry_run):
        for name, flags in applied.items():
            applies[name][key] = flags
        for name, ranges in cut.items():
            units[name][key] = ranges
        for name, (applied_here, feature_rows) in computed.items():
            reports[name].ok += applied_here
            reports[name].not_applicable += not applied_here
            out[name] += feature_rows
        if writer is not None:
            writer.add(key, tables)
    for group in code:
        if dry_run:
            reports[group.name].pending = len(rows)
        else:
            write_group(project, group, out[group.name], directory)
    if writer is not None:
        writer.finish({
            "events": ("rule", {f.name for g in code for f in g.features if f.type == "event"}),
            "evidence": ("feature", {f.name for g in code for f in g.features}),
        })
    return Scan(reports=list(reports.values()), applies=applies, units=units)


ScanResult = tuple[str, dict[str, dict[str, bool]], dict[str, tuple[bool, list[FeatureRow]]],
                   dict[str, list[dict[str, Any]]], dict[str, list[tuple[int, int]]]]


def _scan_results(
    project: Project, code: list[Group], judged: list[Group], rows: list[IndexRow], dry_run: bool
) -> Iterator[ScanResult]:
    """Per trajectory, in order: which judgment operators apply, every code group's rows, the rows of the tables,
    and the units of the groups over turns and windows.

    Trajectories are loaded in `extract.code_workers` processes, each of which loads the project from disk.
    """
    needs_trajectory = (code and not dry_run) or any(
        i.ref.operator.requires is not None or not g.reads_file for g in judged for i in g.instances)
    if not needs_trajectory:
        for row in rows:
            yield row.key, {g.name: {i.id: True for i in g.instances} for g in judged}, {}, {}, {}
        return
    state = (str(project.root), [g.name for g in code], [g.name for g in judged], dry_run)
    keys = [row.key for row in rows]
    processes = project.config.extract.code_workers
    if processes == 1:
        _start_scan(*state)
        yield from map(_scan_one, keys)
        return
    # forkserver: forking this process, which may run threads, can deadlock the children.
    context = multiprocessing.get_context("forkserver")
    with ProcessPoolExecutor(max_workers=processes, mp_context=context, initializer=_start_scan,
                             initargs=state) as pool:
        yield from pool.map(_scan_one, keys, chunksize=64)


_SCAN: dict[str, Any] = {}
"""The groups one scanning process works with, set by `_start_scan`."""


def _start_scan(root: str, code: list[str], judged: list[str], dry_run: bool) -> None:
    project = Project.load(Path(root))
    groups = {g.name: g for g in runtime_groups(project)}
    code_groups = [groups[n] for n in code]
    layers = enabled_layers(project, {table for group in code_groups for table in group.tables}) \
        if code and not dry_run else []
    _SCAN.update(project=project, code=code_groups, judged=[groups[n] for n in judged], dry_run=dry_run,
                 layers=layers, harnesses=discover_harnesses(project),
                 validators={f.name: value_validator(f) for n in code for f in groups[n].features})


def _scan_one(key: str) -> ScanResult:
    project: Project = _SCAN["project"]
    trajectory = load_trajectory(project, key)
    applied = {g.name: {i.id: i.applies(trajectory) for i in g.instances} for g in _SCAN["judged"]}
    cut: dict[str, list[tuple[int, int]]] = {}
    for group in _SCAN["judged"]:
        # The files a call reads are derived data of the project, written on a dry run too, so the calls can be
        # looked up.
        if group.unit == "turn":
            cut[group.name] = _write_turns(project, group, trajectory)
        elif group.unit == "window":
            cut[group.name] = _write_windows(project, group, trajectory)
        elif group.tail is not None:
            cut[group.name] = _write_tail(project, group, trajectory)
    computed: dict[str, tuple[bool, list[FeatureRow]]] = {}
    if _SCAN["dry_run"]:
        return key, applied, computed, {}, cut
    rows, tables = compute_tables(trajectory, _SCAN["layers"], project.config.render,
                                  harness_for(trajectory, _SCAN["harnesses"]))
    rows["evidence"], rows["events"] = [], []
    validators = _SCAN["validators"]
    for group in _SCAN["code"]:
        (instance,) = group.instances
        if not instance.applies(trajectory):
            computed[group.name] = (False, _not_applicable(instance, key))
            continue
        values = instance.compute(trajectory, tables)
        feature_rows = []
        for feature in instance.features:
            value = values[feature.name]
            steps: tuple[int, ...] = ()
            note = ""
            if isinstance(value, Evidenced):
                value, steps, note = value.value, value.steps, value.note
            value = validators[feature.name](value)
            feature_rows.append(FeatureRow(key=key, feature=feature.name, value=value, steps=steps, note=note))
            rows["evidence"] += [{"feature": feature.name, "step": step, "note": note} for step in steps]
            if feature.type == "event" and value is not None:
                assert feature.event is not None
                rows["events"] += [{"rule": feature.name, "edge": feature.event.edge, **event} for event in value]
        computed[group.name] = (True, feature_rows)
    return key, applied, computed, rows, cut


def _call_directory(project: Project, group: Group, trajectory: Trajectory) -> tuple[Path, str]:
    directory = project.windows_dir / group.name / trajectory.dataset
    directory.mkdir(parents=True, exist_ok=True)
    return directory, project.config.datasets[trajectory.dataset].description


def _write_windows(project: Project, group: Group, trajectory: Trajectory) -> list[tuple[int, int]]:
    """Render every window of the trajectory for the group into its windows directory."""
    units = window_units(trajectory, group.window.steps, group.window.stride)
    directory, description = _call_directory(project, group, trajectory)
    for index, (start, end) in enumerate(units):
        text = render_window(trajectory, start, end, project.config.render, description)
        (directory / f"{trajectory.id}.{index}.md").write_text(text, encoding="utf-8")
    return units


def _write_turns(project: Project, group: Group, trajectory: Trajectory) -> list[tuple[int, int]]:
    """Render the turns of the trajectory, each with the steps before its message, into the group's directory,
    one file per `turns_per_call` turns."""
    units = turn_units(trajectory)
    if not units:
        return units
    directory, description = _call_directory(project, group, trajectory)
    first = 0
    for index, part in enumerate(group.parts(units)):
        text = render_turns(trajectory, part, group.context, project.config.render, description, first)
        (directory / f"{trajectory.id}.{index}.md").write_text(text, encoding="utf-8")
        first += len(part)
    return units


def _write_tail(project: Project, group: Group, trajectory: Trajectory) -> list[tuple[int, int]]:
    """Render the last `tail` steps of the trajectory as one window into the group's directory."""
    assert group.tail is not None
    if not trajectory.steps:
        return []
    start = trajectory.steps[max(0, len(trajectory.steps) - group.tail)].index
    end = trajectory.steps[-1].index
    directory, description = _call_directory(project, group, trajectory)
    text = render_window(trajectory, start, end, project.config.render, description)
    (directory / f"{trajectory.id}.0.md").write_text(text, encoding="utf-8")
    return [(start, end)]


def _not_applicable(instance: Instance, key: str) -> list[FeatureRow]:
    return [FeatureRow(key=key, feature=f.name, detail="not applicable") for f in instance.features]


def _issue(
    project: Project, box: Mailbox, known: dict[str, str], group: Group, rows: list[IndexRow],
    units: dict[str, list[tuple[int, int]]], dry_run: bool, second: bool,
) -> dict[str, list[Handle[Any]]]:
    """The calls of the group per trajectory: one, or one per window."""
    function = submitting_function(project, group, second)
    instruction = function.instructions()
    model = model_name(project, group, second)
    handles: dict[str, list[Handle[Any]]] = {}
    for row in rows:
        if group.reads_file:
            request = ExtractRequest(trajectory_file=f"{row.id}.md", n_chunks=row.n_chunks, sha256=row.sha256,
                                     model=model)
            workspace = Workspace(directory=project.dataset_dir(row.dataset))
            handles[row.key] = [issue_call(box, known, function, instruction, request, workspace, dry_run)]
            continue
        directory = project.windows_dir / group.name / row.dataset
        handles[row.key] = []
        for index, part in enumerate(group.parts(units[row.key])):
            file = f"{row.id}.{index}.md"
            digest = hashlib.sha256((directory / file).read_bytes()).hexdigest()
            request = ExtractRequest(trajectory_file=file, n_chunks=1, sha256=digest, model=model, unit=group.unit,
                                     units=part if group.unit == "turn" else [],
                                     index=index if group.unit == "window" else None)
            handles[row.key].append(issue_call(box, known, function, instruction, request,
                                               Workspace(directory=directory), dry_run))
    return handles


def _collect(
    group: Group,
    rows: list[IndexRow],
    handles: dict[str, list[Handle[Any]]],
    applies: dict[str, dict[str, bool]],
    units: dict[str, list[tuple[int, int]]],
    report: GroupReport,
    writer: TableWriter | None,
) -> list[FeatureRow]:
    out: list[FeatureRow] = []
    for row in rows:
        own = handles.get(row.key)
        if own is None:
            # The run covers the trajectory with no rows, so the rows an earlier run wrote for it go.
            if writer is not None:
                writer.add(row.key, {})
            out += [r for instance in group.instances for r in _not_applicable(instance, row.key)]
            continue
        results = [handle.poll() for handle in own]
        if any(result is None for result in results):
            report.pending += 1
            continue
        failed = [r for r in results if r is not None and not isinstance(r.outcome, Returned)]
        if failed:
            outcome = failed[0].outcome
            refused = isinstance(outcome, Refused)
            detail = outcome.reason if isinstance(outcome, Refused) else f"{outcome.fault}: {outcome.detail}"
            report.refused += refused
            report.failed += not refused
            report.errors.append(f"{row.key}: {detail}")
            if writer is not None:
                writer.add(row.key, {})
            out += [FeatureRow(key=row.key, feature=f.name,
                               status="refused" if refused else "failed", detail=detail) for f in group.features]
            continue
        values = [r.outcome.value.model_dump(mode="json") for r in results if r is not None]  # type: ignore[union-attr]
        report.ok += 1
        if group.unit == "trajectory":
            out += _trajectory_rows(group, row, values[0], applies[row.key], report)
        else:
            out += _unit_rows(group, row, values, units[row.key], applies[row.key], report, writer)
    return out


def _trajectory_rows(
    group: Group, row: IndexRow, data: dict[str, Any], applies: dict[str, bool], report: GroupReport
) -> list[FeatureRow]:
    out = []
    for instance in group.instances:
        if not applies[instance.id]:
            out += _not_applicable(instance, row.key)
            continue
        for feature in instance.features:
            answer, evidence = data[feature.name], None
            if group.evidence:
                answer, evidence = answer["value"], answer["evidence"]
            status = "ok"
            if feature.per == "chunk" and len(answer) != row.n_chunks:
                status = "invalid_length"
                report.invalid += 1
            out.append(FeatureRow(key=row.key, feature=feature.name, value=answer, evidence=evidence, status=status))
    return out


def _unit_rows(
    group: Group, row: IndexRow, values: list[dict[str, Any]], units: list[tuple[int, int]],
    applies: dict[str, bool], report: GroupReport, writer: TableWriter | None,
) -> list[FeatureRow]:
    """The judgments of every unit into the tables, and the aggregates of every instance as feature rows.

    `values` holds one answer per call: the units of each part for a group over turns, one window each otherwise.
    """
    answers = [unit for value in values for unit in value["units"]] if group.unit == "turn" else values
    if len(answers) != len(units):
        report.invalid += 1
        if writer is not None:
            writer.add(row.key, {})
        return [FeatureRow(key=row.key, feature=f.name, status="invalid_length",
                           detail=f"{len(answers)} units answered for {len(units)}") for f in group.features]
    judgments: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    per_unit: list[dict[str, Any]] = []
    for index, ((start, end), answer) in enumerate(zip(units, answers, strict=True)):
        plain: dict[str, Any] = {}
        for feature in group.unit_outputs:
            value, cited = answer[feature.name], None
            if group.evidence:
                value, cited = value["value"], value["evidence"]
            plain[feature.name] = value
            judgments.append({"group": group.name, "unit": group.unit, "index": index, "start": start, "end": end,
                              "feature": feature.name, "value": json.dumps(value, ensure_ascii=False),
                              "evidence": cited})
            if feature.event is not None and _matches(value, feature.event.when):
                events.append({"rule": feature.name, "step": start, "end": end, "edge": feature.event.edge,
                               "objects": [], "recovered": False, "recovered_at": None, "note": cited or ""})
        per_unit.append(plain)
    out = []
    for instance in group.instances:
        if not applies[instance.id]:
            out += _not_applicable(instance, row.key)
            continue
        for feature in instance.features:
            aggregate = next(a for a in instance.aggregates if a.name == instance.original[feature.name])
            output = instance.output_names[aggregate.of]
            value, steps = _aggregate(aggregate, [(unit[output], start) for unit, (start, _) in
                                                   zip(per_unit, units, strict=True)])
            out.append(FeatureRow(key=row.key, feature=feature.name, value=value, steps=steps))
            evidence += [{"feature": feature.name, "step": step, "note": ""} for step in steps]
    if writer is not None:
        writer.add(row.key, {"judgments": judgments, "events": events, "evidence": evidence})
    return out


def _matches(value: Any, when: list[str]) -> bool:
    return str(value).lower() in when if isinstance(value, bool) else value in when


def _aggregate(aggregate: Aggregate, values: list[tuple[Any, int]]) -> tuple[Any, tuple[int, ...]]:
    """The aggregate over (value, unit start) pairs, with the starts of the units it rests on."""
    if not values:
        return None, ()
    hits = [start for value, start in values if _matches(value, aggregate.values)]
    match aggregate.kind:
        case "share":
            return len(hits) / len(values), tuple(hits)
        case "any":
            return bool(hits), tuple(hits)
        case "count":
            return float(len(hits)), tuple(hits)
        case "mean":
            return float(np.mean([float(value) for value, _ in values])), ()
        case "last":
            return values[-1][0], (values[-1][1],)
        case "first":
            return values[0][0], (values[0][1],)
        case "set":
            first_seen: dict[str, int] = {}
            for value, start in values:
                first_seen.setdefault(str(value), start)
            return sorted(first_seen), tuple(first_seen[value] for value in sorted(first_seen))
    raise AssertionError(aggregate.kind)
