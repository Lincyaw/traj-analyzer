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
        FeatureSpec(name="d", type="distribution", description="no labels")
    with pytest.raises(ValidationError):
        FeatureSpec(name="v", type="vector", description="neither per nor length")
    with pytest.raises(ValidationError):
        FeatureSpec(name="bad__name", type="scalar", description="double underscore")


def test_operator_kind_must_match_its_functions() -> None:
    with pytest.raises(ValueError, match="compute"):
        Operator(kind="code", description="no compute", outputs=lambda p: [])
    with pytest.raises(ValueError, match="compute"):
        Operator(kind="llm", description="compute", outputs=lambda p: [], compute=lambda t, p: {})


def test_library_operators_load_and_render() -> None:
    refs = discover(None)
    assert {"stats.basic", "stats.tool_usage", "outcome.task_type", "outcome.task_completed",
            "collab.user_frustration", "collab.failure_modes"} <= set(refs)
    for name, ref in refs.items():
        outputs = ref.operator.outputs(ref.operator.params())
        assert all(isinstance(f, FeatureSpec) for f in outputs)
        assert outputs or name == "meta.fields"


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Project:
    root = tmp_path / "proj"
    assert main(["init", str(root)]) == 0
    config = yaml.safe_load((root / "traj.yaml").read_text(encoding="utf-8"))
    config["datasets"] = {"toucan": TOUCAN, "ultrachat": ULTRACHAT}
    config["operators"] = OPERATORS
    config["calls"] = {"outcome": {"evidence": True}}
    (root / "traj.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
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
    model = output_model(outcome)
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


def test_code_operators_and_requires(project: Project) -> None:
    reports = {r.group: r for r in extract(project, groups=["stats-basic", "stats-tool_usage", "fixture-tool_share"])}
    assert reports["stats-basic"].ok == 70
    share = reports["fixture-tool_share"]
    assert (share.ok, share.not_applicable) == (30, 40)
    groups = load_groups(project)
    frame = build_frame(project, groups, load_index(project), vector_length=4)
    assert frame.query["tool_share"].notna().sum() == 30
    assert len(frame.complete(["tool_share", "n_steps"]).query) == 70


def test_sampling_over_code_features(project: Project) -> None:
    extract(project, groups=["stats-basic", "stats-tool_usage"])
    groups = load_groups(project, ["stats-basic", "stats-tool_usage"])
    frame = build_frame(project, groups, load_index(project), vector_length=5)
    assert {"n_steps", "n_tool_calls", "tools_used__count"} <= set(frame.query.columns)

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
    assert listed["collab.user_frustration"]["enabled_as"] == []

    assert main(["operators", "show", "collab.user_frustration"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["params_default"] == {"high": 0.5}

    before = (project.root / "traj.yaml").read_text(encoding="utf-8")
    assert main(["operators", "enable", "fixture.tool_share"]) == 2
    assert (project.root / "traj.yaml").read_text(encoding="utf-8") == before

    assert main(["operators", "enable", "collab.user_frustration", "--as", "pushback", "--call", "collab",
                 "--param", "high=0.8"]) == 0
    capsys.readouterr()
    reloaded = Project.load(project.root)
    groups = {g.name: g for g in load_groups(reloaded)}
    assert [f.name for f in groups["collab"].features] == ["pushback_user_frustration", "pushback_frustration_curve"]
    assert groups["collab"].features[0].thresholds == {"high": 0.8}
    assert expand_where("pushback_user_frustration >= {pushback_user_frustration.high}",
                        {f.name: f.thresholds for f in groups["collab"].features}) == \
        "pushback_user_frustration >= 0.8"

    assert main(["operators", "disable", "pushback"]) == 0
    assert "collab" not in {g.name for g in load_groups(Project.load(project.root))}


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


def test_changed_content_makes_features_stale(project: Project, capsys: pytest.CaptureFixture[str]) -> None:
    extract(project, groups=["stats-basic"])
    config = yaml.safe_load((project.root / "traj.yaml").read_text(encoding="utf-8"))
    config["render"]["max_step_chars"] = 200
    (project.root / "traj.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    ingest(Project.load(project.root), ["toucan"])
    capsys.readouterr()
    assert main(["status"]) == 0
    stats = json.loads(capsys.readouterr().out)["groups"]["stats-basic"]
    assert (stats["ok"], stats["stale"]) == (40, 30)
    frame = build_frame(project, load_groups(project, ["stats-basic"]), load_index(project), vector_length=4)
    assert len(frame.complete(["n_steps"]).query) == 40


def test_dry_run_writes_nothing(project: Project) -> None:
    (report,) = extract(project, groups=["outcome"], limit=3, dry_run=True)
    assert report.pending == 3
    requests = project.mailbox_dir / "requests"
    assert not requests.exists() or not os.listdir(requests)
