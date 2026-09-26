from __future__ import annotations

import json
import multiprocessing
import os
import subprocess
import sys
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from aifn import Call, Handle, Mailbox, Refused, Returned, Workspace

from traj_analyzer.features.spec import ExtractRequest, value_validator
from traj_analyzer.features.table import FeatureRow, write_group
from traj_analyzer.ingest import IndexRow, load_index, load_trajectory
from traj_analyzer.operators.catalog import Group, Instance, load_groups
from traj_analyzer.project import ConfigError, Project
from traj_analyzer.runtime import engine_env, mailbox, model_name, submitting_function


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


class WorkerRunner(Protocol):
    def check(self, project: Project) -> None: ...

    def run(self, project: Project, count: int) -> None: ...


class SubprocessWorkers:
    """Run aifn worker processes that exit once the queue is empty."""

    def check(self, project: Project) -> None:
        engine_env(project)

    def run(self, project: Project, count: int) -> None:
        env = {**os.environ, "TRAJ_PROJECT": str(project.root)}
        command = [sys.executable, "-m", "aifn", "worker", "traj_analyzer.runtime:make_worker",
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
    """Compute code groups and collect LLM groups for the selected trajectories, overwriting their rows.

    Code features are recomputed on every run. LLM calls whose request and instruction are unchanged are answered
    from the aifn mailbox, so a rerun only calls the model for new or changed trajectories and operators.
    """
    runner = runner or SubprocessWorkers()
    rows = select_rows(project, datasets, keys, limit)
    selected = load_groups(project, groups)
    code = [g for g in selected if g.is_code]
    llm = [g for g in selected if not g.is_code]
    if llm and not dry_run:
        runner.check(project)
    code_reports, applies = _scan(project, code, llm, rows, dry_run)
    if not llm:
        return code_reports
    box = mailbox(project)
    known = _known_calls(box)
    issued: dict[str, list[tuple[Handle[Any], IndexRow]]] = {}
    reports: dict[str, GroupReport] = {}
    for group in llm:
        callable_rows = [row for row in rows if any(applies[group.name][row.key].values())]
        issued[group.name] = _issue(project, box, known, group, callable_rows, dry_run)
        cached = sum(handle.poll() is not None for handle, _ in issued[group.name])
        reports[group.name] = GroupReport(group.name, total=len(rows), cached=cached,
                                          not_applicable=len(rows) - len(callable_rows))
    if not dry_run and any(len(issued[g.name]) > reports[g.name].cached for g in llm):
        runner.run(project, workers or project.config.extract.workers)
    for group in llm:
        handles = {row.key: handle for handle, row in issued[group.name]}
        out = _collect(group, rows, handles, applies[group.name], reports[group.name])
        if not dry_run:
            write_group(project, group.name, out)
    return [*code_reports, *reports.values()]


def _scan(
    project: Project, code: list[Group], llm: list[Group], rows: list[IndexRow], dry_run: bool
) -> tuple[list[GroupReport], dict[str, dict[str, dict[str, bool]]]]:
    """Load each trajectory once to compute every code group and decide which LLM operators apply to it."""
    reports = {g.name: GroupReport(g.name, total=len(rows)) for g in code}
    applies: dict[str, dict[str, dict[str, bool]]] = {g.name: {} for g in llm}
    out: dict[str, list[FeatureRow]] = {g.name: [] for g in code}
    for key, applied, computed in _scan_results(project, code, llm, rows, dry_run):
        for name, flags in applied.items():
            applies[name][key] = flags
        for name, feature_rows in computed.items():
            reports[name].ok += feature_rows[0].detail != NOT_APPLICABLE
            reports[name].not_applicable += feature_rows[0].detail == NOT_APPLICABLE
            out[name] += feature_rows
    for group in code:
        if dry_run:
            reports[group.name].pending = len(rows)
        else:
            write_group(project, group.name, out[group.name])
    return list(reports.values()), applies


def _scan_results(
    project: Project, code: list[Group], llm: list[Group], rows: list[IndexRow], dry_run: bool
) -> Iterator[tuple[str, dict[str, dict[str, bool]], dict[str, list[FeatureRow]]]]:
    """Per trajectory, in order: which LLM operators apply, and the rows of every code group.

    Trajectories are loaded in `extract.code_workers` processes, each of which loads the project from disk.
    """
    needs_trajectory = (code and not dry_run) or any(
        i.ref.operator.requires is not None for g in llm for i in g.instances)
    if not needs_trajectory:
        for row in rows:
            yield row.key, {g.name: {i.id: True for i in g.instances} for g in llm}, {}
        return
    state = (str(project.root), [g.name for g in code], [g.name for g in llm], dry_run)
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


def _start_scan(root: str, code: list[str], llm: list[str], dry_run: bool) -> None:
    project = Project.load(Path(root))
    groups = {g.name: g for g in load_groups(project)}
    _SCAN.update(project=project, code=[groups[n] for n in code], llm=[groups[n] for n in llm], dry_run=dry_run,
                 validators={f.name: value_validator(f) for n in code for f in groups[n].features})


def _scan_one(key: str) -> tuple[str, dict[str, dict[str, bool]], dict[str, list[FeatureRow]]]:
    trajectory = load_trajectory(_SCAN["project"], key)
    applied = {g.name: {i.id: i.applies(trajectory) for i in g.instances} for g in _SCAN["llm"]}
    computed: dict[str, list[FeatureRow]] = {}
    if _SCAN["dry_run"]:
        return key, applied, computed
    validators = _SCAN["validators"]
    for group in _SCAN["code"]:
        (instance,) = group.instances
        if not instance.applies(trajectory):
            computed[group.name] = _not_applicable(group, instance, key)
            continue
        values = instance.compute(trajectory)
        computed[group.name] = [FeatureRow(key=key, group=group.name, feature=f.name,
                                           value=validators[f.name](values[f.name])) for f in instance.features]
    return key, applied, computed


NOT_APPLICABLE = "not applicable"


def _not_applicable(group: Group, instance: Instance, key: str) -> list[FeatureRow]:
    return [FeatureRow(key=key, group=group.name, feature=f.name, value=None, detail=NOT_APPLICABLE)
            for f in instance.features]


def _identity(call: Call) -> str:
    """What aifn compares to reuse a call: everything except the id and the time it was issued."""
    return json.dumps(call.model_dump(mode="json", exclude={"id", "issued_at"}), sort_keys=True)


def _known_calls(box: Mailbox) -> dict[str, str]:
    """Identity of every call in the mailbox, read once; aifn's own lookup rescans the mailbox per submission."""
    requests = box.root / "requests"
    if not requests.is_dir():
        return {}
    known = {}
    for path in requests.glob("*.json"):
        call = Call.model_validate_json(path.read_bytes())
        known[_identity(call)] = call.id
    return known


def _issue(
    project: Project, box: Mailbox, known: dict[str, str], group: Group, rows: list[IndexRow], dry_run: bool
) -> list[tuple[Handle[Any], IndexRow]]:
    """Look up or enqueue one call per row; a dry run only looks calls up and writes nothing."""
    function = submitting_function(project, group)
    instruction = function.instructions()
    model = model_name(project, group)
    handles = []
    for row in rows:
        request = ExtractRequest(trajectory_file=f"{row.id}.md", n_chunks=row.n_chunks, sha256=row.sha256,
                                 model=model)
        call = Call(id=uuid4().hex, function=function.name, request=request.model_dump(mode="json", by_alias=True),
                    issued_at=datetime.now(UTC), workspace=Workspace(directory=project.dataset_dir(row.dataset)),
                    instruction=instruction)
        identity = _identity(call)
        if identity in known:
            call = call.model_copy(update={"id": known[identity]})
        elif not dry_run:
            box.issue(call)
            known[identity] = call.id
        handles.append((Handle(id=call.id, returns=function.returns, mailbox=box), row))
    return handles


def _collect(
    group: Group,
    rows: list[IndexRow],
    handles: dict[str, Handle[Any]],
    applies: dict[str, dict[str, bool]],
    report: GroupReport,
) -> list[FeatureRow]:
    out: list[FeatureRow] = []
    for row in rows:
        handle = handles.get(row.key)
        result = handle.poll() if handle is not None else None
        if handle is not None and result is None:
            report.pending += 1
            continue
        outcome = result.outcome if result is not None else None
        if outcome is not None and not isinstance(outcome, Returned):
            refused = isinstance(outcome, Refused)
            detail = outcome.reason if isinstance(outcome, Refused) else f"{outcome.fault}: {outcome.detail}"
            report.refused += refused
            report.failed += not refused
            report.errors.append(f"{row.key}: {detail}")
            out += [FeatureRow(key=row.key, group=group.name, feature=f.name,
                               status="refused" if refused else "failed", detail=detail) for f in group.features]
            continue
        data = outcome.value.model_dump(mode="json") if outcome is not None else {}
        report.ok += outcome is not None
        for instance in group.instances:
            if outcome is None or not applies[row.key][instance.id]:
                out += _not_applicable(group, instance, row.key)
                continue
            for feature in instance.features:
                answer, evidence = data[feature.name], None
                if group.evidence:
                    answer, evidence = answer["value"], answer["evidence"]
                status = "ok"
                if feature.per == "chunk" and len(answer) != row.n_chunks:
                    status = "invalid_length"
                    report.invalid += 1
                out.append(FeatureRow(key=row.key, group=group.name, feature=feature.name, value=answer,
                                      evidence=evidence, status=status))
    return out
