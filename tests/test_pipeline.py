from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest
import yaml
from aifn import Worker
from aifn.engines import ReplayEngine
from pydantic import ValidationError

from traj_analyzer.adapters.claude_code import ClaudeCodeAdapter
from traj_analyzer.adapters.messages import MessagesAdapter
from traj_analyzer.cli import main
from traj_analyzer.extract import extract
from traj_analyzer.features.spec import GroupSpec, load_groups, output_model
from traj_analyzer.ingest import ingest, load_index, render_markdown
from traj_analyzer.project import Project, RenderConfig
from traj_analyzer.runtime import mailbox, policy, registry
from traj_analyzer.sampling import expand_where, load_sampler, sample
from traj_analyzer.vectorize import build_frame

DATA = Path(__file__).parent / "data"
REPLAY = DATA / "replay"
SESSION = next((DATA / "claude_code").glob("*.jsonl"))
TOUCAN = {
    "adapter": "messages",
    "input": [str(DATA / "toucan.jsonl")],
    "options": {
        "messages_encoding": "json_string",
        "id_key": "uuid",
        "metadata_keys": ["subset_name", "question"],
        "role_map": {"tool_call": {"role": "assistant", "kind": "tool_call"},
                     "tool_response": {"role": "tool", "kind": "tool_result"}},
        "tool_call_format": "python_literal",
    },
}
ULTRACHAT = {
    "adapter": "messages",
    "input": [str(DATA / "ultrachat.jsonl")],
    "options": {"id_key": "prompt_id", "metadata_keys": ["prompt"]},
}
OUTCOME = {
    "group": "outcome",
    "features": [
        {"name": "task_type", "type": "category", "description": "Kind of request.",
         "labels": {"lookup": "Answer from tools", "creative": "Write something", "other": "Else"}},
        {"name": "frustration", "type": "scalar", "range": [0, 1],
         "thresholds": {"high": 0.7}, "description": "User dissatisfaction."},
        {"name": "curve", "type": "vector", "per": "chunk", "range": [0, 1],
         "description": "Dissatisfaction per chunk."},
        {"name": "time_split", "type": "distribution", "description": "Where effort went.",
         "labels": {"tools": "Calling tools", "answer": "Writing the answer"}},
    ],
}


def test_claude_code_adapter_reads_real_session() -> None:
    (trajectory,) = ClaudeCodeAdapter().read(SESSION, "cc")
    kinds = {s.kind for s in trajectory.steps}
    assert {"message", "tool_call", "tool_result"} <= kinds
    assert [s.index for s in trajectory.steps] == list(range(len(trajectory.steps)))
    assert trajectory.metadata["title"]
    calls = [s.name for s in trajectory.steps if s.kind == "tool_call"]
    results = [s.name for s in trajectory.steps if s.kind == "tool_result"]
    assert results == calls[: len(results)]
    notifications = [s for s in trajectory.steps if s.content.lstrip().startswith("<task-notification>")]
    assert notifications and all(s.role == "system" for s in notifications)


def test_claude_code_adapter_fails_on_a_corrupt_line(tmp_path: Path) -> None:
    lines = SESSION.read_text(encoding="utf-8").splitlines()
    broken = tmp_path / SESSION.name
    broken.write_text("\n".join([*lines[:5], lines[5][:40], *lines[6:]]), encoding="utf-8")
    with pytest.raises(ValueError, match=f"{broken.name}:6"):
        list(ClaudeCodeAdapter().read(broken, "cc"))


def test_messages_adapter_reads_toucan_tool_calls() -> None:
    trajectories = list(MessagesAdapter(**TOUCAN["options"]).read(DATA / "toucan.jsonl", "t"))
    assert len(trajectories) == 30
    calls = [s for t in trajectories for s in t.steps if s.kind == "tool_call"]
    assert calls and all(s.name for s in calls)
    assert {s.role for t in trajectories for s in t.steps} == {"user", "assistant", "tool"}


def test_messages_adapter_rejects_unmapped_roles() -> None:
    adapter = MessagesAdapter(messages_encoding="json_string", id_key="uuid")
    with pytest.raises(ValueError, match="tool_call"):
        list(adapter.read(DATA / "toucan.jsonl", "t"))


def test_chunks_break_between_steps() -> None:
    (trajectory,) = ClaudeCodeAdapter().read(SESSION, "cc")
    text, n = render_markdown(trajectory, RenderConfig(chunk_chars=5000, max_step_chars=2000))
    assert n > 1 and text.count("<!-- chunk ") == n
    assert all(f"### #{s.index} " in text for s in trajectory.steps)


def test_output_model_enforces_constraints() -> None:
    model = output_model(GroupSpec.model_validate(OUTCOME))
    answer = {
        "task_type": {"value": "lookup", "evidence": "#1"},
        "frustration": {"value": 0.2, "evidence": "#2"},
        "curve": {"value": [0.1, 0.9], "evidence": "#3"},
        "time_split": {"value": {"tools": 0.5, "answer": 0.5}, "evidence": "#4"},
    }
    model.model_validate(answer)
    for name, bad in [("task_type", "nope"), ("frustration", 1.5), ("curve", []),
                      ("time_split", {"tools": 0.9, "answer": 0.5})]:
        with pytest.raises(ValidationError):
            model.model_validate({**answer, name: {"value": bad, "evidence": "#1"}})


def test_spec_rejects_bad_shapes() -> None:
    with pytest.raises(ValidationError):
        GroupSpec.model_validate({"group": "g", "features": [
            {"name": "c", "type": "category", "description": "no labels"}]})
    with pytest.raises(ValidationError):
        GroupSpec.model_validate({"group": "g", "features": [
            {"name": "v", "type": "vector", "description": "neither per nor length"}]})
    with pytest.raises(ValidationError):
        GroupSpec.model_validate({"group": "g", "engine": "builtin", "features": [
            {"name": "s", "type": "scalar", "fn": "missing", "description": "unknown fn"}]})


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Project:
    root = tmp_path / "proj"
    assert main(["init", str(root)]) == 0
    config = yaml.safe_load((root / "traj.yaml").read_text(encoding="utf-8"))
    config["datasets"] = {"toucan": TOUCAN, "ultrachat": ULTRACHAT}
    (root / "traj.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    stats = yaml.safe_load((root / "features" / "stats.yaml").read_text(encoding="utf-8"))
    for feature in stats["features"]:
        if feature["name"] == "n_tool_calls":
            feature["thresholds"] = {"many": 4}
    (root / "features" / "stats.yaml").write_text(yaml.safe_dump(stats), encoding="utf-8")
    (root / "features" / "outcome.yaml").write_text(yaml.safe_dump(OUTCOME), encoding="utf-8")
    monkeypatch.chdir(root)
    loaded = Project.load(root)
    assert ingest(loaded) == {"toucan": 30, "ultrachat": 40}
    return loaded


def test_sampling_over_builtin_features(project: Project) -> None:
    (report,) = extract(project, groups=["stats"])
    assert report.ok == 70
    groups = load_groups(project, ["stats"])
    frame = build_frame(project, groups, load_index(project), vector_length=5)
    assert {"n_steps", "n_tool_calls", "tools_used__count"} <= set(frame.query.columns)

    spec = load_sampler(project, "default").model_copy(update={"budget": 12})
    spec.strategies.insert(0, spec.strategies[0].model_copy(update={
        "kind": "target", "label": "busy", "quota": 4,
        "where": "n_tool_calls >= {n_tool_calls.many}", "within": "top:n_tool_calls"}))
    result = sample(spec, frame, groups)
    picks = result.picks
    assert result.population == 70
    assert len(picks) == 12 and len({p.key for p in picks}) == 12
    busy = [p for p in picks if p.strategy == "busy"]
    assert len(busy) == 4 and all(p.reason["n_tool_calls"] >= 4 for p in busy)
    assert {p.strategy for p in picks} == {"busy", "outlier", "diverse"}
    (busy_report, *_) = result.strategies
    assert busy_report.matched == int((frame.query["n_tool_calls"] >= 4).sum())


def test_complete_population_excludes_missing_features(project: Project, tmp_path: Path) -> None:
    extract(project, groups=["stats"])
    keys = [row.key for row in load_index(project) if row.sha256 in {p.name for p in REPLAY.iterdir()}]
    extract(project, groups=["outcome"], keys=keys, runner=ReplayWorkers(tmp_path / "replay"))
    groups = load_groups(project)
    frame = build_frame(project, groups, load_index(project), vector_length=4)
    spec = load_sampler(project, "default").model_copy(update={
        "budget": 3, "features": ["frustration", "n_steps"]})
    assert sample(spec, frame, groups).population == len(keys)
    assert sample(spec.model_copy(update={"population": "all"}), frame, groups).population == 70


def test_cli_discover_and_sample(project: Project, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["extract", "--group", "stats"]) == 0
    capsys.readouterr()
    assert main(["discover", "--n", "5"]) == 0
    discovered = json.loads(capsys.readouterr().out)
    assert len(discovered["trajectories"]) == 5
    assert all(Path(t["markdown"]).is_file() for t in discovered["trajectories"])
    assert main(["sample", "--budget", "6"]) == 0
    selection = json.loads(capsys.readouterr().out)
    assert len(selection["picks"]) == 6 and Path(selection["selection"]).is_file()


class ReplayWorkers:
    """Answer every pending call with a transcript recorded from a real model run on the same trajectory."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def check(self, project: Project) -> None:
        pass

    def run(self, project: Project, count: int) -> None:
        box = mailbox(project)
        self.directory.mkdir(parents=True, exist_ok=True)
        for call in box.pending():
            shutil.copytree(REPLAY / str(call.request["sha256"]), self.directory / call.id)
        Worker(mailbox=box, registry=registry(project), engine=ReplayEngine(self.directory),
               policy=policy(project), id="replay").work()


def test_llm_extraction_replays_real_runs(project: Project, tmp_path: Path) -> None:
    recorded = sorted(p.name for p in REPLAY.iterdir())
    rows = [row for row in load_index(project, ["toucan"]) if row.sha256 in recorded]
    assert len(rows) == len(recorded)
    keys = [row.key for row in rows]
    (report,) = extract(project, groups=["outcome"], keys=keys, runner=ReplayWorkers(tmp_path / "replay"))
    assert (report.ok, report.failed, report.refused, report.pending) == (len(keys), 0, 0, 0)

    (again,) = extract(project, groups=["outcome"], keys=keys, runner=ReplayWorkers(tmp_path / "unused"))
    assert again.cached == len(keys)

    groups = load_groups(project, ["outcome"])
    frame = build_frame(project, groups, rows, vector_length=4)
    assert frame.query["frustration"].between(0, 1).all()
    assert set(frame.query["task_type"]) <= set(OUTCOME["features"][0]["labels"])
    assert {"curve__t3", "time_split__tools"} <= set(frame.distance.columns)
    for key in keys:
        assert main(["show", key]) == 0


def test_changed_content_makes_features_stale(project: Project, capsys: pytest.CaptureFixture[str]) -> None:
    extract(project, groups=["stats"])
    config = yaml.safe_load((project.root / "traj.yaml").read_text(encoding="utf-8"))
    config["render"]["max_step_chars"] = 200
    (project.root / "traj.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    ingest(Project.load(project.root), ["toucan"])
    capsys.readouterr()
    assert main(["status"]) == 0
    stats = json.loads(capsys.readouterr().out)["groups"]["stats"]
    assert (stats["ok"], stats["stale"]) == (40, 30)
    frame = build_frame(project, load_groups(project, ["stats"]), load_index(project), vector_length=4)
    assert len(frame.complete(["n_steps"]).query) == 40


def test_threshold_expansion() -> None:
    assert expand_where("x >= {x.high}", {"x": {"high": 0.7}}) == "x >= 0.7"


def test_dry_run_writes_nothing(project: Project) -> None:
    (report,) = extract(project, groups=["outcome"], limit=3, dry_run=True)
    assert report.pending == 3
    requests = project.mailbox_dir / "requests"
    assert not requests.exists() or not os.listdir(requests)
