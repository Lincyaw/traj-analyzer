from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_pipeline import edit_config, frame_of
from test_tables import project  # noqa: F401, F811

from traj_analyzer.anomalies import (
    OVERSIGHT,
    Lift,
    Population,
    anomalies,
    consequence_lifts,
    event_weight,
    oversight,
    run_anomalies,
    shifts,
    spread,
    turn_consequences,
)
from traj_analyzer.batches import assign
from traj_analyzer.cli import main
from traj_analyzer.events import RULES
from traj_analyzer.extract import extract
from traj_analyzer.ingest import load_index
from traj_analyzer.labels import labels_report
from traj_analyzer.layers import read_table, table_path
from traj_analyzer.operators.base import FeatureSpec
from traj_analyzer.operators.catalog import load_groups
from traj_analyzer.project import Config, MiningConfig, Project
from traj_analyzer.reviews import Verdict, append_verdict, precision_by_feature, read_verdicts
from traj_analyzer.vectorize import build_frame


def ensure_turns(target: Project) -> None:
    """Write a minimal turns table from the steps table when no layer has written one."""
    if table_path(target, "turns").is_file():
        return
    steps = read_table(target, "steps").to_pandas().sort_values(["key", "step"])
    rows = []
    for key, group in steps.groupby("key", sort=False):
        starts = group[(group["role"] == "user") & (group["kind"] == "message")]["step"].tolist()
        last = int(group["step"].max())
        for index, start in enumerate(starts):
            end = starts[index + 1] - 1 if index + 1 < len(starts) else last
            inside = group[(group["step"] >= start) & (group["step"] <= end)]
            rows.append({"key": key, "index": index, "start": int(start), "end": int(end),
                         "calls": int((inside["kind"] == "tool_call").sum()), "ended_by": "response",
                         "duration_s": None, "asks": 0})
    pq.write_table(pa.Table.from_pylist(rows), table_path(target, "turns"))


def test_batches_join_columns_and_cut_numbers_into_quartiles(project: Project) -> None:  # noqa: F811
    extract(project, groups=["stats-basic"])
    rows = load_index(project)
    frame = build_frame(project, load_groups(project, ["stats-basic"]), rows, vector_length=4)
    batch = assign(project, frame, rows, ["meta.subset_name"])
    assert batch.index.equals(frame.query.index)
    assert set(batch[[k for k in batch.index if k.startswith("ultrachat/")]]) == {"none"}
    assert all(not value.startswith("none") for value in batch[[k for k in batch.index if k.startswith("toucan/")]])
    stratum = assign(project, frame, rows, ["n_steps"])
    assert set(stratum) <= {f"n_steps:q{i}" for i in range(1, 5)}
    joined = assign(project, frame, rows, ["dataset", "n_steps"]) if "dataset" in frame.query else None
    assert joined is None
    assert set(assign(project, frame, rows, [])) == {"all"}


def population_of(seed: int = 0) -> Population:
    rng = np.random.default_rng(seed)
    n = 200
    batch = pd.Series(["a"] * 100 + ["b"] * 100)
    stratum = pd.Series(["s1", "s2"] * 100)
    shifted = list(rng.normal(0, 1, 100)) + list(rng.normal(1.5, 1, 100))
    same = list(rng.normal(0, 1, n))
    flag = [float(v) for v in rng.integers(0, 2, n)]
    frame = frame_of({"shifted": shifted, "same": same, "flag": flag},
                     {"shifted": "scalar", "same": "scalar", "flag": "boolean"},
                     {"shifted": ["shifted"], "same": ["same"], "flag": ["flag"]})
    batch.index = frame.query.index
    stratum.index = frame.query.index
    return Population(frame=frame, batch=batch, stratum=stratum)


def test_shifts_rank_the_columns_that_moved_between_batches() -> None:
    found = shifts(population_of(), Config())
    assert list(found.columns)[-1] == "rank" and found["rank"].tolist() == list(range(1, len(found) + 1))
    top = found.iloc[0]
    assert top["feature"] == "shifted" and top["n"] == 50 and top["n_baseline"] == 50
    assert abs(top["effect"]) > 1 and top["ks"] > 0.4
    assert set(found["batch"]) == {"a", "b"} and set(found["stratum"]) == {"s1", "s2"}
    assert found[found["feature"] == "same"]["effect"].abs().max() < 0.6
    assert found[found["feature"] == "flag"]["ks"].isna().all()
    few = shifts(population_of(), Config.model_validate({"alerts": {"min_batch": 60}}))
    assert few.empty
    assert set(shifts(population_of(), Config(), frozenset({"shifted"}))["feature"]) == {"same", "flag"}


def turn(key: str, index: int, start: int, end: int, calls: int = 2, ended_by: str = "response",
         duration_s: float | None = 10.0, asks: int = 0) -> dict[str, Any]:
    return {"key": key, "index": index, "start": start, "end": end, "calls": calls, "ended_by": ended_by,
            "duration_s": duration_s, "asks": asks}


def turns_of() -> pd.DataFrame:
    """Three trajectories: k1 has a turn the next message pushes back on, k2 an interrupted turn, k3 a quiet
    one."""
    return pd.DataFrame([
        turn("k1", 0, 0, 9), turn("k1", 1, 10, 19), turn("k1", 2, 20, 29, calls=0),
        turn("k2", 0, 0, 9, ended_by="interrupt"), turn("k2", 1, 10, 39), turn("k2", 2, 40, 49),
        turn("k3", 0, 0, 9), turn("k3", 1, 10, 19),
    ])


def test_consequence_lifts_compare_events_with_every_turn() -> None:
    config = Config.model_validate({"alerts": {"consequences": ["reaction", "interrupt"]}})
    events = pd.DataFrame([
        {"key": "k1", "rule": "late", "step": 3}, {"key": "k1", "rule": "reaction", "step": 10},
        {"key": "k2", "rule": "late", "step": 2}, {"key": "k2", "rule": "late", "step": 30},
        {"key": "k3", "rule": "quiet", "step": 2}, {"key": "k3", "rule": "late", "step": 12},
    ])
    turns = turns_of()
    stratum = pd.Series("all", index=["k1", "k2", "k3"])
    marked = turn_consequences(turns, events, config)
    followed = dict(zip(zip(marked["key"], marked["index"], strict=True), marked["followed"], strict=True))
    # k1 turn 0 is followed by the reaction at the start of turn 1; k2 turn 0 was interrupted; the rest are not.
    assert followed == {("k1", 0): True, ("k1", 1): False, ("k1", 2): False, ("k2", 0): True, ("k2", 1): False,
                        ("k2", 2): False, ("k3", 0): False, ("k3", 1): False}
    lifts = consequence_lifts(events, turns, stratum, config)
    late = lifts["late"]
    # The events at k1 step 3 and k2 step 2 fall in followed turns; the ones at k2 step 30 and k3 step 12 do not.
    assert (late.hits, late.followed) == (4, pytest.approx(2 / 4))
    # Of the 7 turns with a tool call, k1 turn 0 and k2 turn 0 are followed.
    assert late.baseline == pytest.approx(2 / 7) and late.lift == pytest.approx((2 / 4) / (2 / 7))
    assert lifts["quiet"].followed == 0.0 and lifts["quiet"].lift == 0.0
    assert "reaction" not in lifts
    # With the judged consequence extracted from k1 alone, the interruption of k2 still counts.
    covered = consequence_lifts(events, turns, stratum, config, covered={"k1"})
    assert "quiet" not in covered and covered["late"].hits == 1 and covered["late"].baseline == pytest.approx(1 / 2)
    only_interrupt = Config.model_validate({"alerts": {"consequences": ["interrupt"]}})
    assert consequence_lifts(events, turns, stratum, only_interrupt, covered={"k1"})["late"].hits == 4


def test_oversight_summarizes_turns_per_batch() -> None:
    config = Config.model_validate({"alerts": {"consequences": ["reaction", "interrupt"]}})
    events = pd.DataFrame([{"key": "k1", "rule": "reaction", "step": 10}])
    batch = pd.Series({"k1": "a", "k2": "b", "k3": "b"})
    table = oversight(turns_of(), events, batch, config, covered={"k1"})
    assert list(table.columns) == [f.name for f in OVERSIGHT]
    a, b = table.set_index("batch").loc["a"], table.set_index("batch").loc["b"]
    assert (a["turns"], a["interrupt_share"], a["ask_share"]) == (3, 0.0, 0.0)
    # Of k1's judged turns, the first is followed by a reaction and the second is not; the last has no next message.
    assert a["pushback_share"] == pytest.approx(1 / 2) and a["calls_per_turn"] == pytest.approx(4 / 3)
    assert (b["turns"], b["interrupt_share"]) == (5, pytest.approx(1 / 5)) and pd.isna(b["pushback_share"])
    assert a["duration_p50"] == 10.0 and b["duration_p99"] == 10.0


def test_reviews_keep_the_latest_verdict_per_trajectory(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    assert main(["init", str(root)]) == 0
    loaded = Project.load(root)
    append_verdict(loaded, Verdict(key="d/1", feature="f", step=3, worth=True, note="a real failure"))
    append_verdict(loaded, Verdict(key="d/2", feature="f", worth=False))
    append_verdict(loaded, Verdict(key="d/1", feature="f", worth=False))
    append_verdict(loaded, Verdict(key="d/1", feature="g", worth=True))
    verdicts = read_verdicts(loaded)
    assert len(verdicts) == 4 and verdicts[0].at and verdicts[0].step == 3
    assert precision_by_feature(verdicts) == {"f": (0.0, 2), "g": (1.0, 1)}
    assert loaded.reviews_path == root / "reviews" / "verdicts.jsonl"


def test_anomalies_score_outliers_and_unrecovered_events() -> None:
    population = population_of()
    key = population.frame.query.index[7]
    population.frame.distance.loc[key, "same"] = 12.0
    events = pd.DataFrame([{"key": key, "rule": "late", "step": 4, "recovered": False},
                           {"key": population.frame.query.index[8], "rule": "late", "step": 4, "recovered": True}])
    scored = anomalies(population, events, {"late": Lift(rule="late", hits=2, followed=0.5, baseline=0.25)})
    assert scored.loc[scored["key"] == key, "anomaly_rank"].item() == 1
    row = scored[scored["key"] == key].iloc[0]
    # A lift of 2 adds 1 before the logarithm; a rule followed no more often than any call would add nothing.
    assert row["events_score"] == pytest.approx(math.log1p(1.0), abs=1e-4) and row["outlier"] > 1
    assert event_weight(Lift(rule="flat", hits=1, followed=0.1, baseline=0.2)) == 0.0 and event_weight(None) == 1.0
    reasons = json.loads(row["reasons"])
    assert reasons["columns"][0]["column"] == "same" and reasons["events"] == [{"rule": "late", "step": 4, "lift": 2.0}]
    assert scored.loc[scored["key"] == population.frame.query.index[8], "events_score"].item() == 0.0


def test_spread_explains_variance_by_batch_inside_strata() -> None:
    config = MiningConfig()
    feature = FeatureSpec(name="n", type="scalar", description="A count.")
    keys = [f"d/{n}" for n in range(100)]
    context = pd.DataFrame({"dataset": ["d"] * 100, "batch": ["a"] * 50 + ["b"] * 50,
                            "stratum": ["s1", "s2"] * 50}, index=keys)
    by_batch = spread(feature, {key: float(i >= 50) + (i % 7) / 10 for i, key in enumerate(keys)}, context, config)
    assert by_batch.batch_explained is not None and by_batch.batch_explained > 0.5
    assert by_batch.objection(config) is None
    by_stratum = spread(feature, {key: float(i % 2) + (i % 7) / 100 for i, key in enumerate(keys)}, context, config)
    assert by_stratum.batch_explained is not None and by_stratum.batch_explained < 0.01
    assert "batch_explained_min" in (by_stratum.objection(config) or "")


def test_cli_anomalies_writes_tables_and_evaluate_reports_labels(
    project: Project, capsys: pytest.CaptureFixture[str]  # noqa: F811
) -> None:
    extract(project)
    ensure_turns(project)
    capsys.readouterr()
    assert main(["review", "toucan/x", "--feature", "blind_retry", "--worth", "--step", "4", "--note", "seen"]) == 0
    assert json.loads(capsys.readouterr().out) == {"feature": "blind_retry", "precision": 1.0, "verdicts": 1,
                                                   "file": "reviews/verdicts.jsonl"}
    assert main(["review", "toucan/y", "--feature", "blind_retry", "--no-worth"]) == 0
    assert json.loads(capsys.readouterr().out)["precision"] == 0.5
    assert main(["anomalies"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["trajectories"] == 70 and len(result["batches"]) > 1
    assert result["reviews"] == {"blind_retry": {"precision": 0.5, "verdicts": 2}}
    scored = read_table(project, "anomalies").to_pandas()
    assert len(scored) == 70 and scored["anomaly_rank"].min() == 1
    assert set(read_table(project, "lifts")["rule"].to_pylist()) <= {rule.name for rule in RULES}
    watched = read_table(project, "oversight").to_pandas()
    assert set(watched["batch"]) == set(result["batches"]) and (watched["turns"] > 0).all()
    assert watched["pushback_share"].isna().all() and (watched["interrupt_share"] == 0).all()
    frame = build_frame(project, load_groups(project), load_index(project), vector_length=4)
    assert {"anomaly_rank", "anomaly_score", "batch", "stratum"} <= set(frame.query.columns)

    def labelled(config: dict[str, Any]) -> None:
        config["labels"] = {"outcome": "n_steps", "correction": "n_user_turns"}

    edit_config(project.root, labelled)
    labelled_project = Project.load(project.root)
    groups = load_groups(labelled_project)
    rows = load_index(labelled_project)
    report = labels_report(labelled_project, build_frame(labelled_project, groups, rows, 4), groups, rows)
    assert report.outcome == "n_steps" and report.failed > 0 and report.anomaly_top == 7
    assert report.anomaly_lift is not None and report.corrected == 70
    assert report.event_coverage == pytest.approx(len(set(read_table(project, "events")["key"].to_pylist())) / 70)
    assert all(a.n >= 20 for a in report.agreement)
    assert run_anomalies(labelled_project)["trajectories"] == 70
    assert Path(labelled_project.tables_dir / "shifts.parquet").is_file()
