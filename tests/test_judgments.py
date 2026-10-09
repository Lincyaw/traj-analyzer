from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from test_pipeline import TOUCAN, ULTRACHAT, edit_config

from traj_analyzer.cli import main
from traj_analyzer.extract import GroupReport, _aggregate, _collect, _unit_rows, extract
from traj_analyzer.features.spec import ExtractRequest, output_model, render_instruction
from traj_analyzer.ingest import ingest, load_index
from traj_analyzer.layers import EVENTS, EVIDENCE, JUDGMENTS, TableWriter, read_table
from traj_analyzer.operators.base import Aggregate, FeatureSpec
from traj_analyzer.operators.catalog import load_groups
from traj_analyzer.project import ConfigError, Project


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Project:
    root = tmp_path / "proj"
    assert main(["init", str(root)]) == 0

    def configure(config: dict[str, Any]) -> None:
        config["datasets"] = {"toucan": TOUCAN, "ultrachat": ULTRACHAT}
        config["operators"] = [
            {"use": "stats.basic"},
            {"use": "signals.user_reaction", "call": "reactions"},
            {"use": "signals.request_fit", "call": "fit"},
            {"use": "signals.end_state", "call": "ending"},
            {"use": "task.prompt_intent", "call": "intents"},
        ]
        config["calls"] = {"reactions": {"evidence": True, "context": 3, "turns_per_call": 2},
                           "fit": {"evidence": False, "sample": {"per_batch": 3}, "window": {"steps": 6, "stride": 3}},
                           "ending": {"evidence": True, "tail": 4},
                           "intents": {"evidence": False, "context": 0}}
        config["batches"] = ["meta.subset_name"]

    edit_config(root, configure)
    monkeypatch.chdir(root)
    loaded = Project.load(root)
    assert ingest(loaded) == {"toucan": 30, "ultrachat": 40}
    return loaded


def test_judgment_groups_carry_their_unit_outputs_and_aggregates(project: Project) -> None:
    groups = {g.name: g for g in load_groups(project)}
    reactions = groups["reactions"]
    assert reactions.unit == "turn" and [f.name for f in reactions.unit_outputs] == ["reaction"]
    assert [f.name for f in reactions.features] == ["pushback_share", "pushbacks", "failure_reports", "repeats",
                                                     "ended_with"]
    assert reactions.features[0].range == (0, 1) and reactions.features[4].labels
    assert reactions.unit_outputs[0].event is not None
    assert reactions.unit_outputs[0].event.when == ["correction", "rejection", "failure_report", "repeat"]
    intents = groups["intents"]
    assert intents.unit == "turn" and intents.context == 0
    assert [f.name for f in intents.unit_outputs] == ["intent"] and "none" in (intents.unit_outputs[0].labels or {})
    assert [(f.name, f.type) for f in intents.features] == [("intents", "set"), ("first_intent", "category")]
    assert intents.features[0].labels == intents.unit_outputs[0].labels
    assert "from the message alone" in render_instruction(intents)
    fit = groups["fit"]
    assert fit.unit == "window" and fit.sample is not None and fit.window.steps == 6
    ending = groups["ending"]
    assert ending.unit == "trajectory" and [f.name for f in ending.features] == ["end_state", "agent_claims_done"]
    assert (reactions.context, reactions.reads_file, ending.tail, ending.reads_file) == (3, False, 4, False)
    units = output_model(reactions)
    units.model_validate({"units": [{"reaction": {"value": "continue", "evidence": "#3"}}]})
    with pytest.raises(ValueError):
        units.model_validate({"units": [{"reaction": {"value": "nope", "evidence": "#3"}}]})
    window = output_model(fit)
    window.model_validate({"request_fit": "no", "unfit_edge": "owner"})
    instruction = render_instruction(reactions)
    assert "one object per turn" in instruction and "`reaction` (category)" in instruction
    assert "### Before the message" in instruction
    assert "## Request" in render_instruction(fit) and "the end of an LLM trajectory" in render_instruction(ending)

    def mixed(config: dict[str, Any]) -> None:
        config["operators"].append({"use": "user.corrects", "call": "reactions"})

    edit_config(project.root, mixed)
    with pytest.raises(ConfigError, match="one call judges one kind of unit"):
        load_groups(Project.load(project.root))


def test_extract_issues_one_call_per_trajectory_or_window(project: Project) -> None:
    extract(project, groups=["stats-basic"])
    reports = {r.group: r for r in extract(project, groups=["reactions", "fit", "ending"], dry_run=True)}
    # The chats of UltraChat have several turns and take two or more calls of two turns each.
    assert reports["reactions"].calls > 70 and reports["ending"].calls == 70
    # Three trajectories per batch of toucan subsets and three of the ultrachat chats without a subset.
    fit = reports["fit"]
    assert 0 < fit.total - fit.not_applicable <= 70 and fit.calls >= fit.total - fit.not_applicable
    request = ExtractRequest(trajectory_file="x.md", n_chunks=1, sha256="a", model="m")
    assert request.model_dump(mode="json", exclude_defaults=True) == {"trajectory_file": "x.md", "n_chunks": 1,
                                                                       "sha256": "a", "model": "m"}
    # A call over turns reads the turns file, and a call over the end reads the tail file, both rendered per group.
    row = load_index(project, ["ultrachat"])[0]
    parts = sorted((project.windows_dir / "reactions" / "ultrachat").glob(f"{row.id}.*.md"))
    assert len(parts) >= 2
    turns = parts[0].read_text(encoding="utf-8")
    assert turns.count("## Turn ") == turns.count("### The message") == 2
    assert "## Turn 0:" in turns and "## Turn 2:" in parts[1].read_text(encoding="utf-8")
    assert "### Before the message" in turns and "### #0 user" in turns
    tail = (project.windows_dir / "ending" / "ultrachat" / f"{row.id}.0.md").read_text(encoding="utf-8")
    assert "## Request" in tail and tail.count("\n### #") <= 4 + 1


def test_unit_rows_aggregate_judgments_and_derive_events(project: Project) -> None:
    group = next(g for g in load_groups(project) if g.name == "reactions")
    row = load_index(project, ["ultrachat"])[0]
    units = [(0, 3), (4, 7), (8, 9)]
    # Two calls of two turns each, as `turns_per_call: 2` cuts them.
    values = [{"units": [
        {"reaction": {"value": "new_task", "evidence": "#0"}},
        {"reaction": {"value": "correction", "evidence": "#4"}},
    ]}, {"units": [
        {"reaction": {"value": "failure_report", "evidence": "#8"}},
    ]}]
    writer = TableWriter(project, {"judgments": JUDGMENTS, "events": EVENTS, "evidence": EVIDENCE})
    report = GroupReport(group.name)
    out = {r.feature: r for r in _unit_rows(group, row, values, units, {"signals.user_reaction": True}, report, writer)}
    assert out["pushback_share"].value == pytest.approx(2 / 3) and out["pushback_share"].steps == (4, 8)
    assert out["pushbacks"].value == 2.0 and out["failure_reports"].value == 1.0 and out["repeats"].value == 0.0
    assert out["ended_with"].value == "failure_report"
    writer.finish()
    judgments = read_table(project, "judgments").to_pandas()
    assert len(judgments) == 3 and json.loads(judgments.iloc[1]["value"]) == "correction"
    events = read_table(project, "events").to_pandas()
    assert events[["rule", "step", "end", "edge"]].to_dict("records") == [
        {"rule": "reaction", "step": 4, "end": 7, "edge": "owner"},
        {"rule": "reaction", "step": 8, "end": 9, "edge": "owner"}]
    assert set(read_table(project, "evidence").to_pandas()["step"]) == {4, 8}
    # A later run whose answer is invalid, or that has no call for the trajectory, removes the rows written before.
    scopes = {"judgments": ("group", {group.name}), "events": ("rule", {"reaction"}),
              "evidence": ("feature", {f.name for f in group.features})}
    writer = TableWriter(project, {"judgments": JUDGMENTS, "events": EVENTS, "evidence": EVIDENCE})
    short = _unit_rows(group, row, values[:1], units, {"signals.user_reaction": True}, report, writer)
    assert all(r.status == "invalid_length" for r in short) and report.invalid == 1
    writer.finish(scopes)
    assert len(read_table(project, "judgments")) == 0 and len(read_table(project, "events")) == 0
    writer = TableWriter(project, {"judgments": JUDGMENTS, "events": EVENTS, "evidence": EVIDENCE})
    _unit_rows(group, row, values, units, {"signals.user_reaction": True}, report, writer)
    writer.finish(scopes)
    assert len(read_table(project, "judgments")) == 3
    writer = TableWriter(project, {"judgments": JUDGMENTS, "events": EVENTS, "evidence": EVIDENCE})
    absent = _collect(group, [row], {}, {}, {}, report, writer)
    assert all(r.detail == "not applicable" for r in absent)
    writer.finish(scopes)
    assert len(read_table(project, "judgments")) == 0


def test_aggregate_kinds() -> None:
    values = [(True, 1), (False, 5), (True, 9)]
    assert _aggregate(Aggregate(name="s", of="x", kind="share", values=["true"], description="d"), values) == (
        pytest.approx(2 / 3), (1, 9))
    assert _aggregate(Aggregate(name="a", of="x", kind="any", values=["true"], description="d"), values)[0] is True
    assert _aggregate(Aggregate(name="c", of="x", kind="count", values=["true"], description="d"), values)[0] == 2.0
    assert _aggregate(Aggregate(name="l", of="x", kind="last", description="d"), values) == (True, (9,))
    assert _aggregate(Aggregate(name="f", of="x", kind="first", description="d"), values) == (True, (1,))
    assert _aggregate(Aggregate(name="m", of="x", kind="mean", description="d"), [(1.0, 1), (3.0, 2)])[0] == 2.0
    assert _aggregate(Aggregate(name="m", of="x", kind="mean", description="d"), []) == (None, ())
    # A set holds the distinct values in sorted order, each with the first unit it was seen in.
    kinds = [("fix", 0), ("build", 3), ("fix", 6)]
    assert _aggregate(Aggregate(name="k", of="x", kind="set", description="d"), kinds) == (["build", "fix"], (3, 0))
    with pytest.raises(ValueError, match="set takes a category output"):
        Aggregate(name="k", of="x", kind="set", description="d").spec(
            {"x": FeatureSpec(name="x", type="boolean", description="d")})
