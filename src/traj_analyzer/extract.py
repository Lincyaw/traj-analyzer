from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Protocol

from aifn import Handle, Refused, Returned, Stage, Workspace
from pydantic import BaseModel

from traj_analyzer.features.spec import ExtractRequest, validate_value
from traj_analyzer.features.table import FeatureRow, write_group
from traj_analyzer.ingest import IndexRow, load_index, load_trajectory
from traj_analyzer.operators.catalog import Group, Instance, load_groups
from traj_analyzer.project import ConfigError, Project
from traj_analyzer.runtime import build_function, engine_env, mailbox


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
        known = {row.key for row in rows}
        unknown = sorted(set(keys) - known)
        if unknown:
            raise ConfigError(f"Unknown trajectory keys: {unknown}")
        rows = [row for row in rows if row.key in set(keys)]
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
    runner = runner or SubprocessWorkers()
    rows = select_rows(project, datasets, keys, limit)
    selected = load_groups(project, groups)
    llm = [group for group in selected if not group.is_code]
    if llm and not dry_run:
        runner.check(project)
    reports = [_code(project, group, rows, dry_run) for group in selected if group.is_code]
    if not llm:
        return reports
    applies = {group.name: _applicability(project, group, rows) for group in llm}
    issued = {}
    llm_reports = {}
    for group in llm:
        callable_rows = [row for row in rows if any(applies[group.name][row.key].values())]
        issued[group.name] = _issue(project, group, callable_rows, dry_run)
        cached = sum(handle.poll() is not None for handle, _ in issued[group.name])
        llm_reports[group.name] = GroupReport(group.name, total=len(rows), cached=cached,
                                              not_applicable=len(rows) - len(callable_rows))
    outstanding = sum(len(issued[g.name]) - llm_reports[g.name].cached for g in llm)
    if not dry_run and outstanding:
        runner.run(project, workers or project.config.extract.workers)
    for group in llm:
        _collect(project, group, rows, issued[group.name], applies[group.name], llm_reports[group.name], dry_run)
    return [*reports, *llm_reports.values()]


def _not_applicable(group: Group, instance: Instance, row: IndexRow) -> list[FeatureRow]:
    return [FeatureRow(key=row.key, group=group.name, feature=f.name, value=None, detail="not applicable",
                       spec_hash=group.spec_hash(), sha256=row.sha256) for f in instance.features]


def _code(project: Project, group: Group, rows: list[IndexRow], dry_run: bool) -> GroupReport:
    report = GroupReport(group.name, total=len(rows))
    if dry_run:
        report.pending = len(rows)
        return report
    (instance,) = group.instances
    digest = group.spec_hash()
    out: list[FeatureRow] = []
    for row in rows:
        trajectory = load_trajectory(project, row)
        if not instance.applies(trajectory):
            report.not_applicable += 1
            out += _not_applicable(group, instance, row)
            continue
        values = instance.compute(trajectory)
        out += [FeatureRow(key=row.key, group=group.name, feature=f.name, value=validate_value(f, values[f.name]),
                           spec_hash=digest, sha256=row.sha256) for f in instance.features]
        report.ok += 1
    write_group(project, group.name, out)
    return report


def _applicability(project: Project, group: Group, rows: list[IndexRow]) -> dict[str, dict[str, bool]]:
    if all(instance.ref.operator.requires is None for instance in group.instances):
        return {row.key: {instance.id: True for instance in group.instances} for row in rows}
    result = {}
    for row in rows:
        trajectory = load_trajectory(project, row)
        result[row.key] = {instance.id: instance.applies(trajectory) for instance in group.instances}
    return result


def _issue(
    project: Project, group: Group, rows: list[IndexRow], dry_run: bool
) -> list[tuple[Handle[Any], IndexRow]]:
    """Enqueue one call per row; a dry run only looks the calls up and writes nothing."""
    stage = Stage(function=build_function(project, group), mailbox=mailbox(project))
    handles = []
    for row in rows:
        request = ExtractRequest(trajectory_file=row.markdown, n_chunks=row.n_chunks,
                                 sha256=row.sha256)
        workspace = Workspace(directory=project.dataset_dir(row.dataset))
        if dry_run:
            call = stage.call(request, workspace=workspace)
            handle = Handle(id=call.id, returns=stage.function.returns, mailbox=stage.mailbox)
        else:
            handle = stage.issue(request, workspace=workspace)
        handles.append((handle, row))
    return handles


def _collect(
    project: Project,
    group: Group,
    rows: list[IndexRow],
    issued: list[tuple[Handle[Any], IndexRow]],
    applies: dict[str, dict[str, bool]],
    report: GroupReport,
    dry_run: bool,
) -> None:
    digest = group.spec_hash()
    called = {row.key for _, row in issued}
    out: list[FeatureRow] = []
    for row in rows:
        if row.key not in called:
            out += [r for i in group.instances for r in _not_applicable(group, i, row)]
    for handle, row in issued:
        result = handle.poll()
        if result is None:
            report.pending += 1
            continue
        outcome = result.outcome
        if isinstance(outcome, Returned):
            report.ok += 1
            out += _rows_from_value(group, row, outcome.value, applies[row.key], report)
            continue
        refused = isinstance(outcome, Refused)
        detail = outcome.reason if isinstance(outcome, Refused) else f"{outcome.fault}: {outcome.detail}"
        report.refused += refused
        report.failed += not refused
        report.errors.append(f"{row.key}: {detail}")
        out += [FeatureRow(key=row.key, group=group.name, feature=f.name,
                           status="refused" if refused else "failed", detail=detail,
                           spec_hash=digest, sha256=row.sha256)
                for f in group.features]
    if not dry_run:
        write_group(project, group.name, out)


def _rows_from_value(
    group: Group, row: IndexRow, value: BaseModel, applies: dict[str, bool], report: GroupReport
) -> list[FeatureRow]:
    data = value.model_dump(mode="json")
    digest = group.spec_hash()
    rows = []
    for instance in group.instances:
        if not applies[instance.id]:
            rows += _not_applicable(group, instance, row)
            continue
        for feature in instance.features:
            answer = data[feature.name]
            evidence = None
            if group.evidence:
                answer, evidence = answer["value"], answer["evidence"]
            status = "ok"
            if feature.per == "chunk" and len(answer) != row.n_chunks:
                status = "invalid_length"
                report.invalid += 1
            rows.append(FeatureRow(key=row.key, group=group.name, feature=feature.name, value=answer,
                                   evidence=evidence, status=status, spec_hash=digest, sha256=row.sha256))
    return rows
