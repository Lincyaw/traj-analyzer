from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from aifn import Handle, Refused, Returned, Stage, Workspace
from pydantic import BaseModel

from traj_analyzer.features.builtin import REGISTRY
from traj_analyzer.features.spec import ExtractRequest, GroupSpec, load_groups
from traj_analyzer.features.table import FeatureRow, write_group
from traj_analyzer.ingest import IndexRow, load_index, load_trajectory
from traj_analyzer.project import ConfigError, Project
from traj_analyzer.runtime import build_function, engine_env, mailbox

WorkerRunner = Callable[[Project, int], None]


@dataclass
class GroupReport:
    group: str
    total: int = 0
    cached: int = 0
    ok: int = 0
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


def spawn_workers(project: Project, count: int) -> None:
    """Run `count` aifn worker processes until the queue is empty."""
    engine_env(project)
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
    run_workers: WorkerRunner = spawn_workers,
) -> list[GroupReport]:
    rows = select_rows(project, datasets, keys, limit)
    specs = load_groups(project, groups)
    reports = [_builtin(project, spec, rows, dry_run) for spec in specs if spec.is_builtin]
    llm = [spec for spec in specs if not spec.is_builtin]
    if not llm:
        return reports
    issued = {spec.group: _issue(project, spec, rows, dry_run) for spec in llm}
    llm_reports = {}
    for spec in llm:
        cached = sum(handle.poll() is not None for handle, _ in issued[spec.group])
        llm_reports[spec.group] = GroupReport(spec.group, total=len(rows), cached=cached)
    outstanding = sum(r.total - r.cached for r in llm_reports.values())
    if not dry_run and outstanding:
        run_workers(project, workers or project.config.extract.workers)
    for spec in llm:
        _collect(project, spec, issued[spec.group], llm_reports[spec.group], dry_run)
    return [*reports, *llm_reports.values()]


def _builtin(project: Project, spec: GroupSpec, rows: list[IndexRow], dry_run: bool) -> GroupReport:
    report = GroupReport(spec.group, total=len(rows))
    if dry_run:
        report.pending = len(rows)
        return report
    digest = spec.spec_hash()
    out: list[FeatureRow] = []
    for row in rows:
        trajectory = load_trajectory(project, row)
        out += [FeatureRow(key=row.key, group=spec.group, feature=feature.name,
                           value=REGISTRY[feature.fn](trajectory), spec_hash=digest)
                for feature in spec.features if feature.fn]
    write_group(project, spec.group, out)
    report.ok = len(rows)
    return report


def _issue(
    project: Project, spec: GroupSpec, rows: list[IndexRow], dry_run: bool
) -> list[tuple[Handle[Any], IndexRow]]:
    """Enqueue one call per row; a dry run only looks the calls up and writes nothing."""
    stage = Stage(function=build_function(project, spec), mailbox=mailbox(project))
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
    spec: GroupSpec,
    issued: list[tuple[Handle[Any], IndexRow]],
    report: GroupReport,
    dry_run: bool,
) -> None:
    digest = spec.spec_hash()
    out: list[FeatureRow] = []
    for handle, row in issued:
        result = handle.poll()
        if result is None:
            report.pending += 1
            continue
        outcome = result.outcome
        if isinstance(outcome, Returned):
            report.ok += 1
            out += _rows_from_value(spec, row, outcome.value, digest, report)
            continue
        refused = isinstance(outcome, Refused)
        detail = outcome.reason if isinstance(outcome, Refused) else (
            f"{outcome.fault}: {outcome.detail}")
        report.refused += refused
        report.failed += not refused
        report.errors.append(f"{row.key}: {detail}")
        out += [FeatureRow(key=row.key, group=spec.group, feature=f.name,
                           status="refused" if refused else "failed", detail=detail,
                           spec_hash=digest)
                for f in spec.features]
    if not dry_run:
        write_group(project, spec.group, out)


def _rows_from_value(
    spec: GroupSpec, row: IndexRow, value: BaseModel, digest: str, report: GroupReport
) -> list[FeatureRow]:
    data = value.model_dump(mode="json")
    rows = []
    for feature in spec.features:
        answer = data[feature.name]
        evidence = None
        if spec.evidence:
            answer, evidence = answer["value"], answer["evidence"]
        status = "ok"
        if feature.per == "chunk" and len(answer) != row.n_chunks:
            status = "invalid_length"
            report.invalid += 1
        rows.append(FeatureRow(key=row.key, group=spec.group, feature=feature.name,
                               value=answer, evidence=evidence, status=status,
                               spec_hash=digest))
    return rows
