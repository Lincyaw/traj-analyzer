from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from traj_analyzer.anomalies import context_frame
from traj_analyzer.evaluation import evaluate, reference_keys
from traj_analyzer.extract import SubprocessWorkers, WorkerRunner, known_calls
from traj_analyzer.ingest import load_index
from traj_analyzer.mining.checks import apply, candidate_items, flagged_items, revise, tail_check, trial
from traj_analyzer.mining.functions import (
    CANDIDATES,
    CHECKS,
    DEFINE,
    EXPLORE,
    FINDINGS,
    HYPOTHESES,
    HYPOTHESIZE,
    REVIEW,
    REVIEWS,
    SCENARIO,
    SURVEY,
    define_instruction,
    explore_instruction,
    functions,
    hypothesize_instruction,
    operator_guide,
    review_instruction,
    survey_instruction,
)
from traj_analyzer.mining.report import RoundReport
from traj_analyzer.mining.session import WORKSPACE, Session
from traj_analyzer.mining.workspace import prepare_workspace
from traj_analyzer.operators.catalog import load_groups
from traj_analyzer.project import ConfigError, Project
from traj_analyzer.runtime import mailbox

REPORT = "report.json"


def mine(
    project: Project, *, rounds: int = 1, workers: int | None = None, runner: WorkerRunner | None = None
) -> list[RoundReport]:
    """Run mining rounds until one adds no feature or `rounds` is reached, then evaluate the resulting features.

    A round that leaves the extraction of its judgment features to `traj extract` is the last one, and no
    evaluation follows it.
    """
    if project.config.evaluation.model is None:
        raise ConfigError("evaluation.model is not set; mining measures every candidate with two models")
    runner = runner or SubprocessWorkers()
    reports: list[RoundReport] = []
    for _ in range(rounds):
        report = run_round(project, workers=workers, runner=runner)
        reports.append(report)
        project = Project.load(project.root)
        if not report.added or not report.extracted:
            break
    if (reports[-1].added or reports[-1].removed) and reports[-1].extracted:
        evaluate(project, workers=workers, runner=runner)
    return reports


def run_round(project: Project, *, workers: int | None, runner: WorkerRunner) -> RoundReport:
    """One round: describe the scenario, propose and test hypotheses, define features, check them and apply."""
    evaluation = evaluate(project, workers=workers, runner=runner)
    number, directory = _round_directory(project)
    groups = load_groups(project)
    rows = load_index(project)
    reference = reference_keys(project, rows)
    held_out = evaluation.held_out()
    session = Session(
        project=project, runner=runner, workers=workers, box=mailbox(project), known=known_calls(mailbox(project)),
        functions=functions(project), workspace=directory / WORKSPACE,
        number=number, directory=directory, rows={row.key: row for row in rows}, reference=reference,
        chosen=evaluation.evaluation_set, readable=[key for key in reference if key not in held_out],
        context=context_frame(project, groups, rows, evaluation.context()))
    report = RoundReport(round=number)
    report.starting, tables = prepare_workspace(session, groups, evaluation)
    config = project.config.mining

    # Each function builds on the files named here; the driver writes what it submitted before the next one runs.
    scenario = session.step(SURVEY, survey_instruction(), ["features.csv", "features.md", "start.txt", SCENARIO])
    session.hand_off(SCENARIO, scenario.notes)
    hypotheses = session.step(HYPOTHESIZE, hypothesize_instruction(config.hypotheses),
                              [SCENARIO, "features.csv", "features.md", f"previous/{CHECKS}"]).hypotheses
    session.hand_off(HYPOTHESES, _json([h.model_dump(mode="json") for h in hypotheses]))
    report.hypotheses = [h.name for h in hypotheses]
    findings = session.step(EXPLORE, explore_instruction(), [SCENARIO, HYPOTHESES, "features.csv"]).findings
    session.hand_off(FINDINGS, _json([f.model_dump(mode="json") for f in findings]))
    for finding in findings:
        report.findings[finding.status] = report.findings.get(finding.status, 0) + 1
    candidates = session.step(DEFINE, define_instruction(config.candidates, operator_guide(tables)),
                              [FINDINGS, SCENARIO, "features.csv", "features.md"]).candidates
    session.hand_off(CANDIDATES, _json([c.model_dump(mode="json") for c in candidates]))
    report.candidates = len(candidates)
    for candidate in candidates:
        if candidate.source is not None:
            session.hand_off(f"operators/{candidate.spec.name}.py", candidate.source)
    reviews = session.step(REVIEW, review_instruction(), [CANDIDATES, FINDINGS, SCENARIO, "features.csv"]).reviews
    session.hand_off(REVIEWS, _json([r.model_dump(mode="json") for r in reviews]))
    for review in reviews:
        report.reviews[review.decision] = report.reviews.get(review.decision, 0) + 1

    items = candidate_items(session, groups, candidates, findings, reviews)
    items += flagged_items(session, groups, evaluation)
    trial(session, [item for item in items if item.decision is None], "trial-1")
    taken = {f.name for g in groups for f in g.features} | {item.name for item in items}
    revised = revise(session, [item for item in items if item.decision == "revise"], taken)
    trial(session, [item for item in revised if item.decision is None], "trial-2")
    for item in items:
        if item.decision == "revise":
            # Revised definitions carry the verdicts; the definition they replace is rejected with their reason.
            item.settle("reject", item.reason)
    tail_check(session, [*items, *revised])
    report.verdicts = [item.verdict() for item in [*items, *revised] if item.replaces is None or item.revised]
    apply(session, [*items, *revised], evaluation, report)
    session.hand_off(CHECKS, _json([verdict.model_dump(mode="json") for verdict in report.verdicts]))
    shutil.copyfile(session.workspace / SCENARIO, project.scenario_path)
    (directory / REPORT).write_text(report.model_dump_json(indent=1), encoding="utf-8")
    return report


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=1)


def _round_directory(project: Project) -> tuple[int, Path]:
    """The directory of the round to run: the last one when it has no report yet, a new one otherwise."""
    numbers = sorted(int(p.name) for p in project.mining_dir.glob("[0-9]*") if p.is_dir() and p.name.isdigit())
    number = numbers[-1] if numbers else 1
    if (project.mining_dir / str(number) / REPORT).is_file():
        number += 1
    directory = project.mining_dir / str(number)
    directory.mkdir(parents=True, exist_ok=True)
    return number, directory
