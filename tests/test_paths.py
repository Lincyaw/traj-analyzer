from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from test_pipeline import edit_config, project  # noqa: F401

from traj_analyzer.cli import main
from traj_analyzer.evaluation import evaluate, surprisal
from traj_analyzer.extract import extract
from traj_analyzer.features.spec import value_validator
from traj_analyzer.ingest import load_index
from traj_analyzer.operators.base import FeatureSpec
from traj_analyzer.operators.catalog import load_groups
from traj_analyzer.paths import PathTree, describe, explained, level_surprisal
from traj_analyzer.project import Project
from traj_analyzer.vectorize import Frame, build_frame
from traj_analyzer.viewer.server import create_app

LEVELS = ["kind", "scope"]


def population() -> pd.Series:
    """Ten trajectories: each reads logs by service, k0 to k4 also read traces by span, and k9 alone reads
    traces by trace id and metrics."""
    values: list[Any] = [[["logs", "service"]] for _ in range(10)]
    for i in range(5):
        values[i] = [["logs", "service"], ["traces", "span"]]
    values[9] = [["logs", "service"], ["traces", "trace_id"], ["metrics", "name"], ["metrics", "name"]]
    return pd.Series(values, index=[f"k{i}" for i in range(10)], dtype=object)


def test_paths_spec_needs_levels() -> None:
    spec = FeatureSpec(name="calls", type="paths", levels=LEVELS, description="Calls.")
    check = value_validator(spec)
    assert check([["logs", "service"]]) == [["logs", "service"]]
    with pytest.raises(ValidationError):
        check([["logs"]])
    with pytest.raises(ValidationError, match="levels are required"):
        FeatureSpec(name="calls", type="paths", description="No levels.")
    with pytest.raises(ValidationError, match="levels are required"):
        FeatureSpec(name="calls", type="set", levels=LEVELS, description="Levels on a set.")


def test_tree_cuts_each_path_at_its_deepest_frequent_node() -> None:
    values = population()
    tree = PathTree.build(LEVELS, values, share=0.3)
    assert (tree.trajectories, tree.minimum, tree.events[()]) == (10, 3, 18)
    assert tree.support["logs",] == 10 and tree.support["traces",] == 6 and tree.support["traces", "span"] == 5
    assert tree.cut(["traces", "span"]) == ("traces", "span")
    assert tree.cut(["traces", "trace_id"]) == ("traces",)
    assert tree.cut(["metrics", "name"]) == ()
    assert [tree.label(node) for node in [(), ("traces",), ("traces", "span")]] == ["*", "traces | *",
                                                                                 "traces | span"]
    shares = tree.share_frame(values)
    assert list(shares.columns) == ["logs | service", "traces | span", "*", "traces | *"]
    assert shares.loc["k9"].to_dict() == {"logs | service": 0.25, "traces | span": 0.0, "*": 0.5, "traces | *": 0.25}
    assert tree.rare(values["k9"]) == 0.75 and tree.rare(values["k0"]) == 0.0
    summary = tree.level_summary()
    assert [(level["nodes"], level["frequent"]) for level in summary] == [(3, 2), (4, 2)]
    assert tree.cut_summary(values) == {"nodes": 4, "events_by_depth": {"(none)": 0.1111, "kind": 0.0556,
                                                                      "scope": 0.8333}}


def test_level_surprisal_charges_a_new_path_to_its_first_new_level() -> None:
    values = population()
    values["new"] = [["traces", "status"], ["events", "name"]]
    values["none"] = None
    inside = np.array([True] * 10 + [False, False])
    by_level = level_surprisal(LEVELS, values, inside)
    # Reference: 18 events under 3 kinds; traces has 6 events under 2 scopes.
    kind = (-math.log(7 / 22) - math.log(1 / 22)) / 2
    scope = -math.log(1 / 9) / 2
    assert by_level.loc["new"].to_dict() == pytest.approx({"kind": kind, "scope": scope})
    assert by_level.loc["none"].isna().all()
    assert by_level.loc["k1", "kind"] == pytest.approx((-math.log(11 / 22) - math.log(7 / 22)) / 2)


def test_explained_share_of_variance() -> None:
    values = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    model = np.array(["a", "a", "b", "b", "a", "a", "b", "b"])
    case = np.array(["x", "x", "x", "x", "y", "y", "y", "y"])
    assert explained(values, case) == pytest.approx(0.8)
    assert explained(values, model) == pytest.approx(0.2)
    assert explained(values, model, case) == pytest.approx(0.2)
    assert explained(np.ones(8), model) is None


def test_paths_feature_in_the_wide_table_and_surprisal() -> None:
    values = population()
    values["new"] = [["traces", "status"]]
    spec = FeatureSpec(name="calls", type="paths", levels=LEVELS, description="Calls.")
    tree = PathTree.build(LEVELS, values, share=0.3)
    shares = tree.share_frame(values)
    frame = Frame(query=shares, distance=shares, columns={"calls": list(shares.columns)},
                  types={"calls": spec.type}, extracted={"calls": set(values.index)}, paths={"calls": values},
                  levels={"calls": LEVELS})
    scores = surprisal(frame, [f"k{i}" for i in range(10)], {})
    assert list(scores.terms.columns) == ["calls@kind", "calls@scope"]
    assert scores.owner == {"calls@kind": "calls", "calls@scope": "calls"}
    # No reference trajectory has a scope level as unusual as the new one; one has a kind level as unusual.
    assert scores.terms.loc["new", "calls@scope"] == pytest.approx(math.log(11))
    assert scores.terms.loc["new", "calls@kind"] == pytest.approx(math.log(11 / 2))
    assert scores.shared.loc["new"].to_dict() == {"calls@kind": 1.0, "calls@scope": 0.0}


def with_tool_paths(root: Any, **sections: Any) -> Project:
    def change(config: dict[str, Any]) -> None:
        config["operators"] = [{"use": "stats.basic"}, {"use": "stats.tool_paths"},
                               {"use": "meta.fields", "as": "meta",
                                "params": {"fields": {"subset": {"key": "subset_name", "type": "category",
                                                                 "required": False}}}}]
        config["calls"] = {}
        config.update(sections)

    edit_config(root, change)
    return Project.load(root)


def test_tool_paths_tree_over_real_conversations(project: Project, capsys: pytest.CaptureFixture[str]) -> None:  # noqa: F811
    loaded = with_tool_paths(project.root, sampling={"path_support": 0.1})
    extract(loaded)
    frame = build_frame(loaded, load_groups(loaded), load_index(loaded), vector_length=4)
    toucan = [key for key in frame.query.index if key.startswith("toucan/")]
    # Every Toucan conversation calls tools and every result arrives, while UltraChat has no tool call.
    assert frame.paths["tool_paths"][toucan].map(lambda paths: all(path[1] == "ok" for path in paths)).all()
    assert frame.query.loc[toucan, "tool_paths__count"].equals(frame.query.loc[toucan, "n_steps"] * 0 +
                                                               frame.paths["tool_paths"][toucan].map(len))
    assert frame.query["tool_paths__count"].notna().sum() == 30
    node_columns = frame.columns["tool_paths"]
    assert "tool_paths__*" in node_columns
    assert frame.query.loc[toucan, node_columns].sum(axis=1).round(9).eq(1).all()

    capsys.readouterr()
    assert main(["tree", "tool_paths", "--by", "subset", "--top", "3"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert (shown["trajectories"], shown["minimum_support"]) == (30, 3)
    assert [level["level"] for level in shown["levels"]] == ["tool", "outcome"]
    assert shown["levels"][1]["bits"] == 0.0 and "explained_median" in shown["levels"][0]
    assert len(shown["nodes"]) == 3 and shown["nodes"][0]["events"] >= shown["nodes"][1]["events"]
    assert sum(shown["cut"]["events_by_depth"].values()) == pytest.approx(1.0, abs=1e-3)
    assert main(["tree", "n_steps"]) == 2
    assert main(["tree", "tool_paths", "--within", "subset"]) == 2

    evaluation = evaluate(loaded)
    roles = {card.feature: card.role for card in evaluation.features}
    assert roles["subset"] == "context" and roles["tool_paths"] == "distinguishing"
    assert evaluation.context() == frozenset({"subset"})

    client = TestClient(create_app(loaded))
    response = client.post("/api/analysis", json={"table": "features", "feature": "tool_paths",
                                                  "columns": ["key"]})
    assert response.status_code == 200, response.text
    assert int(response.headers["X-Analysis-Rows"]) == int(frame.query["tool_paths__count"].sum())


def test_describe_compares_groups_at_every_node() -> None:
    values = population()
    tree = PathTree.build(LEVELS, values, share=0.3)
    group = pd.Series(["a"] * 5 + ["b"] * 5, index=values.index, name="model")
    shown = describe(tree, values, by=group, top=10)
    nodes = {node["node"]: node for node in shown["nodes"]}
    # Only group a reads traces by span, so the group explains all of that node's variation.
    assert nodes["traces | span"]["explained"] == 1.0
    assert nodes["traces | *"]["depth"] == 1 and nodes["traces | *"]["trajectories"] == 6
    assert shown["levels"][1]["explained_median"] is not None
