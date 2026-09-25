from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from aifn import Worker
from aifn.engines import ReplayEngine
from pydantic import ValidationError
from ruamel.yaml import YAML

from traj_analyzer.adapters.claude_code import ClaudeCodeAdapter
from traj_analyzer.adapters.messages import MessagesAdapter
from traj_analyzer.cli import main
from traj_analyzer.extract import extract
from traj_analyzer.features.spec import output_model, render_instruction
from traj_analyzer.ingest import ingest, load_index, render_markdown
from traj_analyzer.operators.base import FeatureSpec, Operator
from traj_analyzer.operators.catalog import discover, load_groups
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
ULTRACHAT_DESCRIPTION = "Multi-turn chats from the public UltraChat dataset."
ULTRACHAT = {
    "adapter": "messages",
    "input": [str(DATA / "ultrachat.jsonl")],
    "options": {"id_key": "prompt_id", "metadata_keys": ["prompt"]},
}
OPERATORS = [
    {"use": "stats.basic"},
    {"use": "stats.tool_usage"},
    {"use": "fixture.tool_share"},
    {"use": "fixture.outcome", "call": "outcome"},
]


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


def test_feature_spec_rejects_bad_shapes() -> None:
    with pytest.raises(ValidationError):
        FeatureSpec(name="b", type="boolean", labels={"x": "labels on a boolean"}, description="bad")
    with pytest.raises(ValidationError):
        FeatureSpec(name="s", type="set", range=(0, 1), description="range on a set")
    with pytest.raises(ValidationError):
        FeatureSpec(name="v", type="vector", description="neither per nor length")
    with pytest.raises(ValidationError):
        FeatureSpec(name="bad__name", type="scalar", description="double underscore")


def test_operator_kind_must_match_its_functions() -> None:
    with pytest.raises(ValueError, match="compute"):
        Operator(kind="code", description="no compute", outputs=lambda p: [])
    with pytest.raises(ValueError, match="compute"):
        Operator(kind="llm", description="compute", outputs=lambda p: [], compute=lambda t, p: {})


def test_library_operators_are_atomic() -> None:
    refs = discover(None)
    assert {"stats.basic", "stats.tool_usage", "meta.fields", "task.request_kinds", "user.dissatisfied",
            "user.corrects", "user.follow_up", "user.approves", "assistant.claims_done",
            "assistant.asks_user"} == set(refs)
    for name, ref in refs.items():
        outputs = ref.operator.outputs(ref.operator.params())
        assert all(isinstance(f, FeatureSpec) for f in outputs)
        assert outputs or name == "meta.fields"
        if ref.operator.kind == "llm":
            assert all(f.type in ("boolean", "category", "set") for f in outputs), name


def edit_config(root: Path, change: Callable[[dict[str, Any]], None]) -> None:
    editor = YAML(typ="safe")
    config = editor.load(root / "traj.yaml")
    change(config)
    editor.dump(config, root / "traj.yaml")


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Project:
    root = tmp_path / "proj"
    assert main(["init", str(root)]) == 0

    def configure(config: dict[str, Any]) -> None:
        config["datasets"] = {"toucan": TOUCAN, "ultrachat": {**ULTRACHAT, "description": ULTRACHAT_DESCRIPTION}}
        config["operators"] = OPERATORS
        config["calls"] = {"outcome": {"evidence": True}}

    edit_config(root, configure)
    shutil.copytree(DATA / "operators", root / "operators", dirs_exist_ok=True)
    monkeypatch.chdir(root)
    loaded = Project.load(root)
    assert ingest(loaded) == {"toucan": 30, "ultrachat": 40}
    return loaded


def test_groups_follow_enabled_operators(project: Project) -> None:
    groups = {g.name: g for g in load_groups(project)}
    assert set(groups) == {"stats-basic", "stats-tool_usage", "fixture-tool_share", "outcome"}
    outcome = groups["outcome"]
    assert [f.name for f in outcome.features] == ["task_type", "frustration", "curve", "time_split"]
    assert "### Operator `fixture.outcome`" in render_instruction(outcome)
    ultrachat = load_index(project, ["ultrachat"])[0].key
    assert ULTRACHAT_DESCRIPTION in project.trajectory_path(ultrachat, ".md").read_text(encoding="utf-8")
    model = output_model(outcome)
    answer = {
        "task_type": {"value": "lookup", "evidence": "#1"},
        "frustration": {"value": 0.2, "evidence": "#2"},
        "curve": {"value": [0.1, 0.9], "evidence": "#3"},
        "time_split": {"value": {"tools": 0.5, "answer": 0.5}, "evidence": "#4"},
    }
    model.model_validate(answer)
    for name, bad in [("task_type", "nope"), ("frustration", 1.5), ("curve", []),
                      ("time_split", {"tools": 1.5}), ("time_split", {"other": 0.5})]:
        with pytest.raises(ValidationError):
            model.model_validate({**answer, name: {"value": bad, "evidence": "#1"}})


def test_where_expands_thresholds_from_feature_definitions(project: Project) -> None:
    thresholds = {f.name: f.thresholds for g in load_groups(project) for f in g.features}
    assert expand_where("frustration >= {frustration.high}", thresholds) == "frustration >= 0.7"


def test_code_operators_and_requires(project: Project) -> None:
    reports = {r.group: r for r in extract(project, groups=["stats-basic", "stats-tool_usage", "fixture-tool_share"])}
    assert reports["stats-basic"].ok == 70
    share = reports["fixture-tool_share"]
    assert (share.ok, share.not_applicable) == (30, 40)
    groups = load_groups(project)
    frame = build_frame(project, groups, load_index(project), vector_length=4)
    assert frame.query["tool_share"].notna().sum() == 30
    assert len(frame.complete(["tool_share", "n_steps"]).query) == 70
    tool_columns = frame.columns["tool_calls"]
    assert tool_columns and all(c.startswith("tool_calls__") for c in tool_columns)
    per_tool_total = frame.query[tool_columns].sum(axis=1)
    toucan = [k for k in frame.query.index if k.startswith("toucan/")]
    assert (per_tool_total[toucan] == frame.query.loc[toucan, "n_tool_calls"]).all()


def test_sampling_over_code_features(project: Project) -> None:
    extract(project, groups=["stats-basic", "stats-tool_usage"])
    groups = load_groups(project, ["stats-basic", "stats-tool_usage"])
    frame = build_frame(project, groups, load_index(project), vector_length=5)
    assert {"n_steps", "n_tool_calls", "tool_error_rate"} <= set(frame.query.columns)

    spec = load_sampler(project, "default").model_copy(update={"budget": 12})
    spec.strategies.insert(0, spec.strategies[0].model_copy(update={
        "kind": "target", "label": "busy", "quota": 4, "where": "n_tool_calls >= 4", "within": "top:n_tool_calls"}))
    result = sample(spec, frame, groups)
    picks = result.picks
    assert result.population == 70
    assert len(picks) == 12 and len({p.key for p in picks}) == 12
    busy = [p for p in picks if p.strategy == "busy"]
    assert len(busy) == 4 and all(p.reason["n_tool_calls"] >= 4 for p in busy)
    assert {p.strategy for p in picks} == {"busy", "outlier", "diverse"}
    (busy_report, *_) = result.strategies
    assert busy_report.matched == int((frame.query["n_tool_calls"] >= 4).sum())


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


def _recorded_keys(project: Project) -> list[str]:
    recorded = {p.name for p in REPLAY.iterdir()}
    keys = [row.key for row in load_index(project, ["toucan"]) if row.sha256 in recorded]
    assert len(keys) == len(recorded)
    return keys


def test_llm_extraction_replays_real_runs(project: Project, tmp_path: Path) -> None:
    keys = _recorded_keys(project)
    (report,) = extract(project, groups=["outcome"], keys=keys, runner=ReplayWorkers(tmp_path / "replay"))
    assert (report.ok, report.failed, report.refused, report.pending) == (len(keys), 0, 0, 0)

    (again,) = extract(project, groups=["outcome"], keys=keys, runner=ReplayWorkers(tmp_path / "unused"))
    assert again.cached == len(keys)

    rows = [row for row in load_index(project) if row.key in keys]
    frame = build_frame(project, load_groups(project, ["outcome"]), rows, vector_length=4)
    assert frame.query["frustration"].between(0, 1).all()
    assert set(frame.query["task_type"]) <= {"lookup", "creative", "other"}
    assert {"curve__t3", "time_split__tools"} <= set(frame.distance.columns)
    for key in keys:
        assert main(["show", key]) == 0


def test_llm_operators_skip_trajectories_they_do_not_apply_to(project: Project, tmp_path: Path) -> None:
    keys = [*_recorded_keys(project), *(row.key for row in load_index(project, ["ultrachat"])[:2])]
    (report,) = extract(project, groups=["outcome"], keys=keys, runner=ReplayWorkers(tmp_path / "replay"))
    assert (report.ok, report.not_applicable) == (3, 2)
    requests = list((project.mailbox_dir / "requests").iterdir())
    assert len(requests) == 3


def test_complete_population_excludes_missing_features(project: Project, tmp_path: Path) -> None:
    extract(project, groups=["stats-basic"])
    keys = _recorded_keys(project)
    extract(project, groups=["outcome"], keys=keys, runner=ReplayWorkers(tmp_path / "replay"))
    groups = load_groups(project)
    frame = build_frame(project, groups, load_index(project), vector_length=4)
    spec = load_sampler(project, "default").model_copy(update={"budget": 3, "features": ["frustration", "n_steps"]})
    assert sample(spec, frame, groups).population == len(keys)
    assert sample(spec.model_copy(update={"population": "all"}), frame, groups).population == 70


def test_cli_discover_and_sample(project: Project, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["extract", "--group", "stats-basic", "--group", "stats-tool_usage",
                 "--group", "fixture-tool_share"]) == 0
    capsys.readouterr()
    assert main(["discover", "--n", "5"]) == 0
    discovered = json.loads(capsys.readouterr().out)
    assert len(discovered["trajectories"]) == 5
    assert all(Path(t["markdown"]).is_file() for t in discovered["trajectories"])
    assert main(["sample", "--budget", "6"]) == 0
    selection = json.loads(capsys.readouterr().out)
    assert len(selection["picks"]) == 6 and Path(selection["selection"]).is_file()


def test_cli_operators_enable_and_disable(project: Project, capsys: pytest.CaptureFixture[str]) -> None:
    capsys.readouterr()
    assert main(["operators", "list", "--kind", "llm"]) == 0
    listed = {op["name"]: op for op in json.loads(capsys.readouterr().out)["operators"]}
    assert listed["fixture.outcome"]["origin"] == "project"
    assert listed["fixture.outcome"]["enabled_as"] == ["fixture.outcome"]
    assert listed["user.corrects"]["enabled_as"] == []

    assert main(["operators", "show", "task.request_kinds"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert "fix" in shown["params_default"]["labels"]

    before = (project.root / "traj.yaml").read_text(encoding="utf-8")
    assert main(["operators", "enable", "fixture.tool_share"]) == 2
    assert (project.root / "traj.yaml").read_text(encoding="utf-8") == before

    assert main(["operators", "enable", "user.corrects", "--call", "dialogue"]) == 0
    assert main(["operators", "enable", "task.request_kinds", "--call", "dialogue",
                 "--param", "labels={fix: Fix a bug., other: Anything else.}"]) == 0
    assert main(["operators", "enable", "task.request_kinds", "--as", "coarse", "--call", "dialogue"]) == 2
    assert main(["operators", "enable", "task.request_kinds", "--as", "coarse", "--prefix", "coarse",
                 "--call", "dialogue"]) == 0
    capsys.readouterr()
    groups = {g.name: g for g in load_groups(Project.load(project.root))}
    features = {f.name: f for f in groups["dialogue"].features}
    assert list(features) == ["user_corrects", "request_kinds", "coarse_request_kinds"]
    assert set(features["request_kinds"].labels or {}) == {"fix", "other"}

    assert main(["operators", "disable", "coarse"]) == 0
    assert main(["operators", "disable", "task.request_kinds"]) == 0
    assert main(["operators", "disable", "user.corrects"]) == 0
    assert "dialogue" not in {g.name for g in load_groups(Project.load(project.root))}


def test_cli_adapters_list_finds_builtin_and_project_adapters(
    project: Project, capsys: pytest.CaptureFixture[str]
) -> None:
    shutil.copyfile(Path(__file__).parents[1] / "src/traj_analyzer/adapters/messages.py",
                    project.root / "adapters" / "messages.py")
    capsys.readouterr()
    assert main(["adapters", "list"]) == 0
    adapters = {a["name"]: a for a in json.loads(capsys.readouterr().out)["adapters"]}
    assert adapters["claude_code"]["origin"] == "builtin"
    assert adapters["messages"]["origin"] == "project"
    assert adapters["messages"]["used_by"] == ["toucan", "ultrachat"]
    assert ingest(Project.load(project.root), ["ultrachat"]) == {"ultrachat": 40}


def test_meta_fields_copy_metadata_into_features(project: Project, capsys: pytest.CaptureFixture[str]) -> None:
    subset = json.dumps({"key": "subset_name", "type": "category", "required": False})
    assert main(["operators", "enable", "meta.fields", "--as", "meta", "--param", f"fields={{subset: {subset}}}"]) == 0
    capsys.readouterr()
    reloaded = Project.load(project.root)
    (report,) = extract(reloaded, groups=["meta"])
    assert report.ok == 70
    frame = build_frame(reloaded, load_groups(reloaded, ["meta"]), load_index(reloaded), vector_length=4)
    records = [json.loads(line) for line in (DATA / "toucan.jsonl").open(encoding="utf-8")]
    expected = {f"toucan/{r['uuid']}": r["subset_name"] for r in records}
    column = frame.query["subset"]
    assert {k: column[k] for k in expected} == expected
    assert column[[k for k in column.index if k.startswith("ultrachat/")]].isna().all()
    assert {f"subset__{name.replace('-', '_')}" for name in expected.values()} <= set(frame.distance.columns)

    strict = json.dumps({"key": "subset_name", "type": "category"})
    assert main(["operators", "disable", "meta"]) == 0
    assert main(["operators", "enable", "meta.fields", "--as", "meta", "--param", f"fields={{subset: {strict}}}"]) == 0
    with pytest.raises(KeyError, match="subset_name"):
        extract(Project.load(project.root), groups=["meta"])


def test_hidden_metadata_stays_out_of_the_markdown() -> None:
    (trajectory,) = ClaudeCodeAdapter().read(SESSION, "cc")
    shown, _ = render_markdown(trajectory, RenderConfig())
    hidden, _ = render_markdown(trajectory, RenderConfig(hide_metadata=["title", "cwd"]))
    assert f'"title": "{trajectory.metadata["title"]}"' in shown
    assert '"title"' not in hidden and '"cwd"' not in hidden and '"gitBranch"' in hidden
    patterned, _ = render_markdown(trajectory, RenderConfig(hide_metadata=["*ed_at"]))
    assert '"started_at"' not in patterned and '"ended_at"' not in patterned and '"title"' in patterned


def test_llm_answers_are_reused_only_for_the_same_content_and_model(project: Project, tmp_path: Path) -> None:
    keys = _recorded_keys(project)
    extract(project, groups=["outcome"], keys=keys, runner=ReplayWorkers(tmp_path / "replay"))
    (same,) = extract(project, groups=["outcome"], keys=keys, dry_run=True)
    assert (same.cached, same.pending) == (3, 0)

    edit_config(project.root, lambda config: config["calls"]["outcome"].update(model="another-model"))
    (other_model,) = extract(Project.load(project.root), groups=["outcome"], keys=keys, dry_run=True)
    assert (other_model.cached, other_model.pending) == (0, 3)

    edit_config(project.root, lambda config: config["calls"]["outcome"].pop("model"))
    edit_config(project.root, lambda config: config["render"].update(max_step_chars=200))
    reingested = Project.load(project.root)
    ingest(reingested, ["toucan"])
    (changed,) = extract(reingested, groups=["outcome"], keys=keys, dry_run=True)
    assert (changed.cached, changed.pending) == (0, 3)


def test_status_counts_rows_by_status(project: Project, capsys: pytest.CaptureFixture[str]) -> None:
    extract(project, groups=["fixture-tool_share"], datasets=["toucan"])
    capsys.readouterr()
    assert main(["status"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["datasets"] == {"toucan": 30, "ultrachat": 40}
    assert status["groups"]["fixture-tool_share"] == {"operators": ["fixture.tool_share"], "extracted": 30,
                                                      "missing": 40, "ok": 30}


def test_dry_run_writes_nothing(project: Project) -> None:
    (report,) = extract(project, groups=["outcome"], limit=3, dry_run=True)
    assert report.pending == 3
    requests = project.mailbox_dir / "requests"
    assert not requests.exists() or not os.listdir(requests)
