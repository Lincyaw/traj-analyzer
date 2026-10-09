from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
from test_pipeline import TOUCAN, ULTRACHAT, edit_config

from traj_analyzer.cli import main
from traj_analyzer.events import RULES, Context, events_of
from traj_analyzer.extract import extract
from traj_analyzer.harness import EMPTY, discover_harnesses
from traj_analyzer.ingest import ingest, load_index, render_turns, render_window, turn_units, window_units
from traj_analyzer.layers import Layer, LayerRef, compute_tables, discover_layers, enabled_layers, read_table
from traj_analyzer.layers.base import ordered
from traj_analyzer.layers.tables import STEPS, Tables, column, steps_rows
from traj_analyzer.operators.base import Event, EventMark, FeatureSpec
from traj_analyzer.operators.catalog import load_groups
from traj_analyzer.project import ConfigError, Project, RenderConfig
from traj_analyzer.schema import Step, Trajectory
from traj_analyzer.vectorize import build_frame

OPERATORS = [
    {"use": "stats.basic"},
    {"use": "stats.actions"},
    {"use": "stats.dialogue"},
    {"use": "events.generic"},
]


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Project:
    root = tmp_path / "proj"
    assert main(["init", str(root)]) == 0

    def configure(config: dict[str, Any]) -> None:
        config["datasets"] = {"toucan": TOUCAN, "ultrachat": ULTRACHAT}
        config["operators"] = OPERATORS
        config["calls"] = {}
        config["batches"] = ["meta.subset_name"]
        config["strata"] = ["n_user_turns"]

    edit_config(root, configure)
    monkeypatch.chdir(root)
    loaded = Project.load(root)
    assert ingest(loaded) == {"toucan": 30, "ultrachat": 40}
    return loaded


def test_layers_are_enabled_by_the_tables_operators_read(project: Project) -> None:
    assert [ref.name for ref in enabled_layers(project, set())] == []
    assert [ref.name for ref in enabled_layers(project, {"actions"})] == ["actions"]
    with pytest.raises(ConfigError, match="no layer writes"):
        enabled_layers(project, {"nothing"})
    # The turns layer requires the actions table, so the actions layer runs with it, and before it.
    assert [ref.name for ref in enabled_layers(project, {"turns"})] == ["actions", "turns"]
    edit_config(project.root, lambda config: config.update(layers=["sql"]))
    names = [ref.name for ref in enabled_layers(Project.load(project.root), {"relations"})]
    assert names == ["actions", "sql"]
    assert set(discover_layers(project)) == {"actions", "sql", "turns"}


def test_layers_run_after_the_layers_they_require() -> None:
    refs = discover_layers(None)
    assert [ref.name for ref in ordered([refs["turns"], refs["actions"]])] == ["actions", "turns"]
    with pytest.raises(ConfigError, match="no enabled layer writes"):
        ordered([refs["turns"]])
    with pytest.raises(ValueError, match="requires the tables it writes"):
        Layer(name="loop", description="x", schemas={"loops": pa.schema([])}, compute=lambda t, x, h: {},
              requires=("loops",))
    chained = Layer(name="chained", description="x", schemas={"chains": pa.schema([column("n", pa.int64(), "n")])},
                    compute=lambda t, tables, h: {"chains": [{"n": len(tables.records("turns"))}]},
                    requires=("turns",))
    ref = LayerRef(name="chained", origin="project", path=Path("chained.py"), layer=chained)
    rows, tables = compute_tables(chat(), [ref, refs["turns"], refs["actions"]], RenderConfig(), EMPTY)
    assert set(rows) == {"steps", "trajectories", "actions", "relations", "turns", "chains"}
    assert tables.records("chains") == [{"n": 2}]


def test_extract_writes_the_tables_of_the_layers_and_the_features(project: Project) -> None:
    reports = {r.group: r for r in extract(project)}
    assert reports["stats-actions"].ok == 30 and reports["events-generic"].ok == 30
    steps = read_table(project, "steps")
    assert steps.num_rows == sum(row.n_steps for row in load_index(project))
    trajectories = read_table(project, "trajectories")
    assert trajectories.num_rows == 70 and set(trajectories["dataset"].to_pylist()) == {"toucan", "ultrachat"}
    actions = read_table(project, "actions").to_pandas()
    assert set(actions["key"].str.split("/").str[0]) == {"toucan"}
    assert {"act", "outcome", "objects", "turn"} <= set(actions.columns)
    assert read_table(project, "relations").num_rows >= 0
    turns = read_table(project, "turns").to_pandas()
    index = {row.key: row for row in load_index(project)}
    user_messages = sum(row.metadata.get("turns", 0) for row in index.values())
    assert len(turns) == sum(trajectories["turns"].to_pylist()) and user_messages >= 0
    assert set(turns["ended_by"]) <= {"interrupt", "ask", "response", "tool"}
    assert (turns["notices"] == 0).all()
    assert turns["calls"].sum() == steps.to_pandas()["kind"].eq("tool_call").sum()
    toucan_turns = turns[turns["key"].str.startswith("toucan/")]
    assert (toucan_turns["actions"] >= toucan_turns["calls"]).all() and toucan_turns["first_act"].notna().any()
    evidence = read_table(project, "evidence").to_pandas()
    assert {"interjections", "repeated_prompts", "side_effects"} >= set(evidence["feature"]) or evidence.empty
    events = read_table(project, "events").to_pandas()
    assert set(events["rule"]) <= {rule.name for rule in RULES}
    assert set(events["edge"]) <= {"tool", "owner"}

    # Running again on a part of the trajectories keeps the rows of the others.
    keys = [row.key for row in load_index(project, ["toucan"])][:3]
    extract(project, groups=["stats-basic"], keys=keys)
    assert read_table(project, "steps").num_rows == steps.num_rows
    assert read_table(project, "actions").num_rows == len(actions)

    frame = build_frame(project, load_groups(project), load_index(project), vector_length=4)
    assert {"result_unanswered", "result_unanswered__unrecovered", "first_unrecovered_step"} <= set(frame.query.columns)
    assert frame.types["blind_retry"] == "event"
    assert (frame.query["first_unrecovered_rule"].dropna().isin({rule.name for rule in RULES})).all()


def test_query_runs_sql_over_the_tables(project: Project, capsys: pytest.CaptureFixture[str]) -> None:
    extract(project, groups=["stats-basic"])
    capsys.readouterr()
    assert main(["query", "SELECT role, count(*) AS n FROM 'steps.parquet' GROUP BY role ORDER BY role"]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert out[0] == "role,n" and len(out) >= 3


ROOT = "/home/u/proj"


def action(position: int, **fields: Any) -> dict[str, Any]:
    base = {"position": position, "call": position * 2 + 1, "ordinal": 0, "turn": 0, "tool": "Bash", "act": "run",
            "program": "x", "subcommand": None, "effect": "none", "forced": False, "outcome": "ok", "error": None,
            "shape": [], "lines": None, "objects": [], "text": "x"}
    return {**base, **fields}


def test_generic_rules_find_events_and_recovery() -> None:
    rules = {rule.name: rule for rule in RULES}
    file = [{"kind": "file", "name": f"{ROOT}/a.py", "role": "source"}]
    actions = [
        action(0, outcome="unanswered", tool="Read", act="read", objects=file, text=None),
        action(1, outcome="ok", tool="Read", act="read", objects=file, text=None),
        action(2, outcome="error", text="pytest -q", act="test"),
        action(3, outcome="error", text="pytest -q", act="test"),
        action(4, outcome="error", text="pytest -q -x", act="test"),
        action(5, effect="remote", text="git push", act="vcs", program="git", subcommand="push"),
        action(6, effect="destructive", text="git push --force", act="vcs", program="git", subcommand="push"),
        action(7, turn=1, effect="remote", text="git push", act="vcs", program="git", subcommand="push"),
    ]
    relations = [{"kind": "retry", "later": 3, "earlier": 2}, {"kind": "retry", "later": 4, "earlier": 3}]
    context = Context(actions=actions, relations=relations)
    unanswered = rules["result_unanswered"].events(context)
    assert [(e.step, e.recovered, e.recovered_at) for e in unanswered] == [(1, True, 3)]
    retries = rules["blind_retry"].events(context)
    assert [(e.step, e.recovered) for e in retries] == [(7, False)]
    unasked = rules["side_effect_unasked"].events(context)
    assert [e.step for e in unasked] == [13]
    runs = rules["failure_run"].events(context)
    assert [(e.step, e.recovered, e.recovered_at) for e in runs] == [(9, True, 11)]


def turn(index: int, **fields: Any) -> dict[str, Any]:
    base = {"index": index, "start": index * 10, "end": index * 10 + 9, "prompt_chars": 5, "notices": 0,
            "steps": 10, "messages": 1, "calls": 3, "actions": 3, "edits": 1, "verifications": 0, "asks": 0,
            "plans": 0, "side_effects": 0, "failed_calls": 0, "delegations": 0, "duration_s": None,
            "ended_by": "response", "first_act": "read", "last_act": "edit", "last_program": "edit"}
    return {**base, **fields}


def test_turn_rules_find_interrupted_turns() -> None:
    rules = {rule.name: rule for rule in RULES}
    turns = [
        turn(0, ended_by="interrupt", last_act="plan", last_program="exitplanmode", messages=0, edits=0),
        turn(1, ended_by="interrupt", last_act="read", last_program="read", messages=0, edits=0),
        turn(2, ended_by="interrupt", last_act="edit", edits=2, messages=1),
        turn(3, ended_by="response"),
    ]
    context = Context(actions=[], relations=[], turns=turns)
    assert context.turn_of(15) == turns[1] and context.turn_of(99) is None
    at_plan = rules["interrupted_at_plan"].events(context)
    assert [(e.step, e.recovered, e.objects) for e in at_plan] == [(0, False, [])]
    assert at_plan[0].note == "interrupted after exitplanmode"
    without_output = rules["interrupted_without_output"].events(context)
    assert [e.step for e in without_output] == [0, 10]


def interrupted_chat() -> Trajectory:
    steps = [
        Step(index=0, role="user", content="update the readme"),
        Step(index=1, role="assistant", kind="tool_call", name="Bash",
             content=json.dumps({"command": "brew install x"})),
        Step(index=2, role="tool", kind="tool_result", name="Bash", content="installed"),
        Step(index=3, role="assistant", kind="tool_call", name="Bash", content=json.dumps({"command": "x --version"})),
        Step(index=4, role="user", kind="notice", name="interruption", content="[Request interrupted by user]"),
        Step(index=5, role="user", content="update the readme, nothing else"),
        Step(index=6, role="assistant", kind="tool_call", name="Edit",
             content=json.dumps({"file_path": "/home/u/proj/README.md", "old_string": "a", "new_string": "b"})),
        Step(index=7, role="tool", kind="tool_result", name="Edit", content="The file has been updated"),
        Step(index=8, role="assistant", content="Done.", timestamp="2026-10-01T10:00:30+00:00"),
    ]
    return Trajectory(id="i", dataset="d", steps=steps, harness="claude_code")


def test_turns_layer_reads_interruptions_and_what_each_turn_did() -> None:
    refs = discover_layers(None)
    harness = discover_harnesses(None)["claude_code"]
    _, tables = compute_tables(interrupted_chat(), [refs["turns"], refs["actions"]], RenderConfig(), harness)
    turns = tables.records("turns")
    # The notice opens no turn: it ends the turn it interrupts.
    assert [(t["index"], t["start"], t["end"]) for t in turns] == [(0, 0, 4), (1, 5, 8)]
    first, last = turns
    # `x` is no program the shell component knows, so the last act is `other`.
    assert (first["ended_by"], first["calls"], first["edits"], first["last_act"]) == ("interrupt", 2, 0, "other")
    assert first["last_program"] == "x" and first["first_act"] == "install" and first["notices"] == 1
    assert (last["ended_by"], last["edits"], last["messages"], last["last_program"]) == ("response", 1, 1, "edit")
    assert last["notices"] == 0 and first["duration_s"] is None
    assert tables.records("steps")[4]["kind"] == "notice"
    events = events_of(RULES, tables)
    assert [e["step"] for e in events["interrupted_without_output"]] == [0]
    assert events["interrupted_at_plan"] == []


def test_event_feature_spec_needs_an_edge() -> None:
    FeatureSpec(name="e", type="event", description="x", event=EventMark(edge="tool"))
    with pytest.raises(ValueError, match="names its edge"):
        FeatureSpec(name="e", type="event", description="x")
    with pytest.raises(ValueError, match="lists the values"):
        FeatureSpec(name="b", type="boolean", description="x", event=EventMark(edge="tool"))
    with pytest.raises(ValueError, match="labels the feature lacks"):
        FeatureSpec(name="c", type="category", description="x", labels={"a": "A"},
                    event=EventMark(edge="tool", when=["b"]))
    Event(step=3).model_dump()


def chat() -> Trajectory:
    steps = [
        Step(index=0, role="user", content="first"),
        Step(index=1, role="assistant", content="ok"),
        Step(index=2, role="assistant", kind="tool_call", name="Read", content=json.dumps({"file_path": "a"})),
        Step(index=3, role="tool", kind="tool_result", name="Read", content="x"),
        Step(index=4, role="user", content="second"),
        Step(index=5, role="assistant", content="done"),
    ]
    return Trajectory(id="t", dataset="d", steps=steps, metadata={"m": 1})


def test_units_cut_turns_and_windows_and_render_windows() -> None:
    trajectory = chat()
    assert turn_units(trajectory) == [(0, 3), (4, 5)]
    assert window_units(trajectory, steps=4, stride=2) == [(0, 3), (2, 5)]
    assert window_units(trajectory, steps=10, stride=2) == [(0, 5)]
    assert window_units(trajectory, steps=4, stride=1) == [(0, 3), (1, 4), (2, 5)]
    text = render_window(trajectory, 2, 5, RenderConfig(hide_metadata=["m"]), "A chat.")
    assert "## Request" in text and "### #4 user" in text and '"m"' not in text
    assert text.index("### #4 user") < text.index("## Steps") and "### #0 user" not in text
    turns = render_turns(trajectory, turn_units(trajectory), 2, RenderConfig(), "A chat.")
    assert turns.count("## Turn ") == 2 and "## Turn 1: steps #4 to #5" in turns
    # The second turn shows the two steps before its message and the message; the first has nothing before it.
    assert turns.index("### #2 assistant") < turns.index("### #3 tool") < turns.index("### #4 user")
    assert "### #1 assistant" not in turns and "### #5 assistant" not in turns
    assert turns.count("### Before the message") == 1
    rows = steps_rows(trajectory, RenderConfig())
    assert [row["turn"] for row in rows] == [1, 1, 1, 1, 2, 2]
    tables = Tables.of({"steps": rows}, {"steps": STEPS})
    assert tables.records("steps")[2]["name"] == "Read" and tables["steps"].shape == (6, len(STEPS))
    with pytest.raises(KeyError, match="declares the tables"):
        tables["actions"]
