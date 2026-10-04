from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_pipeline import ReplayWorkers, edit_config, frame_of, project  # noqa: F401

from traj_analyzer.cli import main
from traj_analyzer.evaluation import (
    Card,
    Evaluation,
    Relation,
    assign_roles,
    dependence,
    disagreement,
    drift,
    evaluate,
    scaled,
    surprisal,
)
from traj_analyzer.extract import extract
from traj_analyzer.features.spec import render_instruction
from traj_analyzer.ingest import load_index
from traj_analyzer.operators.base import FeatureSpec
from traj_analyzer.operators.catalog import load_groups
from traj_analyzer.project import ConfigError, Project
from traj_analyzer.sampling import Reference, SamplerSpec, StrategySpec, sample

CODE_GROUPS = ["stats-basic", "stats-tool_usage", "fixture-tool_share"]


def spec(type: str, **fields: Any) -> FeatureSpec:
    return FeatureSpec(name="f", type=type, description="A feature.", **fields)


def test_disagreement_follows_the_feature_type() -> None:
    assert disagreement(spec("boolean"), True, False) == 1.0
    assert disagreement(spec("category"), "a", "a") == 0.0
    assert disagreement(spec("set"), ["a", "b"], ["b", "c"]) == pytest.approx(2 / 3)
    assert disagreement(spec("set"), [], []) == 0.0
    assert disagreement(spec("scalar", range=(0, 10)), 2.0, 4.0) == pytest.approx(0.2)
    assert disagreement(spec("scalar"), 2.0, 4.0) == pytest.approx(0.5)
    assert disagreement(spec("vector", per="chunk"), [1.0, 2.0], [1.0]) == 1.0
    assert disagreement(spec("map", range=(0, 1)), {"a": 1.0}, {"a": 0.5, "b": 0.5}) == pytest.approx(0.5)


def relation_frame() -> tuple[Any, list[str]]:
    """Reference rows k0 to k59 where y is 2x and c is constant; three monitored rows, each breaking one thing."""
    rng = np.random.default_rng(0)
    x = [float(v) for v in rng.integers(0, 6, size=60)]
    y = [2 * v for v in x]
    noise = list(rng.normal(size=60))
    # k60 keeps the relation, k61 breaks it with values that are each common, k62 changes the constant.
    x += [3.0, 3.0, 2.0]
    y += [6.0, 10.0, 4.0]
    noise += [0.0, 0.0, 0.0]
    constant = [1.0] * 62 + [5.0]
    frame = frame_of({"x": x, "y": y, "noise": noise, "c": constant},
                     {"x": "scalar", "y": "scalar", "noise": "scalar", "c": "scalar"},
                     {"x": ["x"], "y": ["y"], "noise": ["noise"], "c": ["c"]})
    return frame, [f"k{i}" for i in range(60)]


def test_roles_follow_variation_and_dependence() -> None:
    frame, reference = relation_frame()
    inside = frame.distance.index.isin(reference)
    blocks = {name: scaled(frame, name, frame.distance.index[inside]) for name in frame.columns}
    assert dependence(blocks["x"], blocks["y"], inside) == pytest.approx(1.0)
    assert dependence(blocks["x"], blocks["noise"], inside) < 0.5
    roles = assign_roles(["c", "x", "noise", "y"], {"c": 1, "x": 6, "noise": 60, "y": 6},
                         lambda a, b: dependence(blocks[a], blocks[b], inside), 0.9)
    assert roles["c"] == ("invariant", None)
    assert roles["x"] == ("distinguishing", None) and roles["noise"] == ("distinguishing", None)
    assert roles["y"] == ("dependent", Relation(source="x", r=1.0))


def test_surprisal_names_the_violated_invariant_and_relation() -> None:
    frame, reference = relation_frame()
    scores = surprisal(frame, reference, {"y": "x"})
    assert set(scores.terms.columns) == {"x", "x->y", "noise", "c"}
    assert scores.owner["x->y"] == "y"
    most = math.log(61)
    assert scores.terms.loc["k62", "c"] == pytest.approx(most)
    assert scores.terms.loc["k60", "c"] == pytest.approx(0.0)
    assert scores.terms.loc["k61", "x->y"] == pytest.approx(most)
    assert scores.terms.loc["k60", "x->y"] < 0.1
    assert scores.shared.loc["k62", "c"] == 0 and scores.shared.loc["k60", "c"] == 60
    assert scores.total().loc[["k60", "k61", "k62"]].idxmin() == "k60"


def test_surprisal_of_labels_and_missing_values() -> None:
    answers = [1.0] * 20 + [0.0]
    labels = ["a"] * 15 + ["b"] * 5 + ["c"]
    onehot = {f"kind__{label}": [float(v == label) for v in labels] for label in "abc"}
    tags = {"tags__x": [1.0] * 10 + [0.0] * 11, "tags__new": [0.0] * 20 + [1.0]}
    partial = [0.5] * 19 + [np.nan, np.nan]
    frame = frame_of({"flag": answers, **onehot, **tags, "partial": partial},
                     {"flag": "boolean", "kind": "category", "tags": "set", "partial": "scalar"},
                     {"flag": ["flag"], "kind": list(onehot), "tags": list(tags), "partial": ["partial"]},
                     query={"kind": labels})
    scores = surprisal(frame, [f"k{i}" for i in range(20)], {})
    row = scores.terms.loc["k20"]
    assert row["flag"] == pytest.approx(math.log(22))
    assert row["kind"] == pytest.approx(math.log(23))
    assert row["tags__new"] == pytest.approx(math.log(22))
    assert row["partial__missing"] == pytest.approx(-math.log(2 / 22))
    assert row["partial"] == 0.0
    assert drift(frame, "kind", frame.distance.index.isin([f"k{i}" for i in range(20)])) == pytest.approx(1.0)


def card(feature: str, role: str, source: str | None = None) -> Card:
    relation = Relation(source=source, r=1.0) if source else None
    return Card(feature=feature, group="g", kind="code", type="scalar", noise=0.0, distinct=2, role=role,
                relation=relation, decision="keep")


def test_surprise_strategy_ranks_monitored_trajectories() -> None:
    frame, keys = relation_frame()
    evaluation = Evaluation(reference=60, monitored=3, evaluation_set=[], features=[
        card("x", "distinguishing"), card("noise", "distinguishing"), card("c", "invariant"),
        card("y", "dependent", "x")])
    reference = Reference(keys=keys, evaluation=evaluation)
    sampler = SamplerSpec(name="s", budget=2, features="distinguishing",
                          strategies=[StrategySpec(kind="surprise", quota="rest")])
    result = sample(sampler, frame, [], reference)
    assert {pick.key for pick in result.picks} == {"k61", "k62"}
    reasons = {pick.key: pick.reason for pick in result.picks}
    assert reasons["k61"]["columns"][0]["column"] == "x->y" and reasons["k61"]["of"] == 3
    assert reasons["k62"]["columns"][0] == {"column": "c", "surprisal": round(math.log(61), 3), "value": 5.0,
                                            "shared": 0.0}
    with pytest.raises(ConfigError, match="traj evaluate"):
        sample(sampler, frame, [])
    everything = Reference(keys=list(frame.distance.index), evaluation=evaluation)
    with pytest.raises(ConfigError, match="reference set"):
        sample(sampler, frame, [], everything)


def without_llm(root: Any, **sections: Any) -> Project:
    def change(config: dict[str, Any]) -> None:
        config["operators"] = [op for op in config["operators"] if "call" not in op]
        config["calls"] = {}
        config.update(sections)

    edit_config(root, change)
    return Project.load(root)


def test_evaluate_requires_features_for_the_reference_set(project: Project) -> None:  # noqa: F811
    extract(project, groups=CODE_GROUPS)
    with pytest.raises(ConfigError, match="Group outcome has no features for any reference trajectory"):
        evaluate(project)
    with pytest.raises(ConfigError, match="unknown trajectory keys"):
        evaluate(without_llm(project.root, reference={"exclude": ["toucan/none"]}))


def test_evaluate_assigns_roles_and_monitoring_finds_the_new_value(
    project: Project, capsys: pytest.CaptureFixture[str]  # noqa: F811
) -> None:
    extract(project, groups=CODE_GROUPS)
    reloaded = without_llm(project.root, reference={"datasets": ["toucan"]})
    capsys.readouterr()
    assert main(["evaluate"]) == 0
    result = Evaluation.model_validate(json.loads(capsys.readouterr().out))
    assert (result.reference, result.monitored, len(result.evaluation_set)) == (30, 40, 30)
    cards = {c.feature: c for c in result.features}
    # Every Toucan conversation has one user turn, and none of the samples records a duration or a tool error.
    assert cards["n_user_turns"].role == "invariant" and cards["n_user_turns"].distinct == 1
    assert cards["duration_minutes"].distinct == 0 and cards["duration_minutes"].drift is None
    assert cards["n_steps"].role == "distinguishing" and cards["n_user_turns"].drift == 1.0
    assert all(c.noise == 0.0 and c.decision == "keep" for c in result.features)

    (reloaded.root / "samplers" / "watch.yaml").write_text(
        "name: watch\nbudget: 3\npopulation: all\nfeatures: distinguishing\n"
        "strategies:\n  - {kind: surprise, quota: rest}\n", encoding="utf-8")
    assert main(["sample", "--sampler", "watch"]) == 0
    picks = json.loads(capsys.readouterr().out)["picks"]
    assert len(picks) == 3 and all(p["key"].startswith("ultrachat/") for p in picks)
    named = {c["column"] for p in picks for c in p["reason"]["columns"]}
    assert "n_user_turns" in named
    turns = next(c for c in picks[0]["reason"]["columns"] if c["column"] == "n_user_turns")
    assert turns["shared"] == 0.0 and turns["surprisal"] == round(math.log(31), 3)

    reference_rows = load_index(reloaded, ["toucan"])
    assert len(reference_rows) == result.reference


def test_evaluate_needs_a_second_model_for_llm_groups(project: Project, tmp_path: Path) -> None:  # noqa: F811
    edit_config(project.root, lambda config: config.update(reference={"datasets": ["ultrachat"]}))
    reloaded = Project.load(project.root)
    # The outcome operator applies to no UltraChat conversation, so this extraction makes no model call.
    extract(reloaded, datasets=["ultrachat"], runner=ReplayWorkers(tmp_path / "replay"))
    with pytest.raises(ConfigError, match="evaluation.model is not set"):
        evaluate(reloaded)
    edit_config(project.root, lambda config: config.update(evaluation={"model": config["engine"]["model"]}))
    with pytest.raises(ConfigError, match="equals the model of group outcome"):
        evaluate(Project.load(project.root))


def test_llm_features_are_defined_in_configuration(project: Project, capsys: pytest.CaptureFixture[str]) -> None:  # noqa: F811
    features = json.dumps({"agent_apologizes": {
        "type": "boolean", "description": "The assistant writes an apology.",
        "guidance": "A refusal without an apology does not count."}})
    assert main(["operators", "enable", "llm.features", "--as", "mined", "--call", "mined",
                 "--param", f"features={features}"]) == 0
    capsys.readouterr()
    (group,) = load_groups(Project.load(project.root), ["mined"])
    assert [(f.name, f.type) for f in group.features] == [("agent_apologizes", "boolean")]
    instruction = render_instruction(group)
    assert "- `agent_apologizes`: A refusal without an apology does not count." in instruction
    assert main(["operators", "enable", "llm.features", "--as", "bad", "--call", "mined",
                 "--param", "features={n: {type: scalar, description: A number.}}"]) == 2
