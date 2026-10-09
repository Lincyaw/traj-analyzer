from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest
from test_pipeline import edit_config
from test_tables import project  # noqa: F401, F811

from traj_analyzer.extract import extract
from traj_analyzer.ingest import load_index, load_trajectory
from traj_analyzer.labels import labels_report, map_reaction, stratified_rho
from traj_analyzer.layers import JUDGMENTS, TableWriter
from traj_analyzer.operators.catalog import load_groups
from traj_analyzer.project import ConfigError, Project
from traj_analyzer.vectorize import build_frame


def _judge(writer: TableWriter, key: str, reactions: list[tuple[int, int, str]]) -> None:
    writer.add(key, {"judgments": [
        {"group": "reactions", "unit": "turn", "index": i, "start": start, "end": end, "feature": "reaction",
         "value": json.dumps(reaction), "evidence": None}
        for i, (start, end, reaction) in enumerate(reactions)]})


def _label(loaded: Project, key: str, items: list[dict[str, Any]]) -> None:
    trajectory = load_trajectory(loaded, key)
    labelled = trajectory.model_copy(update={"metadata": {**trajectory.metadata, "prompt_labels": items}})
    loaded.trajectory_path(key, ".json").write_text(labelled.model_dump_json(), encoding="utf-8")


def test_map_reaction_collapses_what_the_turn_labels_do_not_tell_apart() -> None:
    assert [map_reaction(r) for r in ("correction", "rejection", "failure_report")] == [
        "correction", "rejection", "failure_report"]
    assert {map_reaction(r) for r in ("new_task", "continue", "clarify", "repeat", "other")} == {"non_pushback"}


def test_judged_reactions_are_checked_against_the_turn_labels(
    project: Project,  # noqa: F811
) -> None:
    extract(project, groups=["stats-basic"])
    keys = [row.key for row in load_index(project, ["ultrachat"])][:2]
    writer = TableWriter(project, {"judgments": JUDGMENTS})
    _judge(writer, keys[0], [(0, 1, "new_task"), (2, 3, "correction"), (4, 5, "repeat"), (6, 7, "failure_report")])
    _judge(writer, keys[1], [(0, 1, "new_task"), (2, 3, "rejection")])
    writer.finish()
    # The second trajectory's label for step 2 disagrees, and step 9 has no judgment.
    _label(project, keys[0], [{"step": 0, "pushback": "non_pushback"}, {"step": 2, "pushback": "correction"},
                              {"step": 4, "pushback": "non_pushback"}, {"step": 6, "pushback": "failure_report"}])
    _label(project, keys[1], [{"step": 0, "pushback": None}, {"step": 2, "pushback": "correction"},
                              {"step": 9, "pushback": "rejection"}])
    edit_config(project.root, lambda config: config.update(labels={"turn_correction": "prompt_labels"}))
    labelled = Project.load(project.root)
    groups = load_groups(labelled)
    rows = load_index(labelled)
    report = labels_report(labelled, build_frame(labelled, groups, rows, 4), groups, rows)
    assert report.turn_correction == "prompt_labels" and report.turns_compared == 5
    assert report.turn_agreement == pytest.approx(4 / 5) and report.turn_agreement_binary == pytest.approx(1.0)
    assert report.turn_kappa is not None and 0 < report.turn_kappa < 1 and report.turn_kappa_binary == 1.0
    assert report.turn_confusion == {"correction": {"correction": 1}, "failure_report": {"failure_report": 1},
                                     "non_pushback": {"non_pushback": 2}, "rejection": {"correction": 1}}

    _label(project, keys[1], [{"step": 2, "pushback": "nope"}])
    with pytest.raises(ConfigError, match="is not one of"):
        labels_report(labelled, build_frame(labelled, groups, rows, 4), groups, rows)


def test_stratified_rho_ignores_differences_between_strata() -> None:
    # Two strata of 40: in each, the feature and the failure are unrelated, while the stratum with more events also
    # fails more often, which a correlation over the pooled trajectories would report.
    index = pd.Index([f"t{i}" for i in range(80)])
    stratum = pd.Series(["a"] * 40 + ["b"] * 40, index=index)
    feature = pd.Series([0.0] * 30 + [1.0] * 10 + [0.0] * 10 + [1.0] * 30, index=index)
    # Stratum a fails 3 of 30 without the event and 1 of 10 with it; stratum b 5 of 10 and 15 of 30.
    failed = pd.Series(([1.0] * 3 + [0.0] * 27 + [1.0] + [0.0] * 9) + ([1.0] * 5 + [0.0] * 5 + [1.0] * 15 + [0.0] * 15),
                       index=index)
    assert stratified_rho(feature, failed, stratum) == pytest.approx(0.0, abs=1e-9)
    assert stratified_rho(feature, feature, stratum) == pytest.approx(1.0)
    assert stratified_rho(feature, pd.Series(0.0, index=index), stratum) is None
