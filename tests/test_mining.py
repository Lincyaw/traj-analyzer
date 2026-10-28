from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import pytest
from aifn import Call, Worker
from aifn.engines import ReplayEngine
from pydantic import ValidationError
from test_pipeline import DATA, TOUCAN, ULTRACHAT, edit_config

from traj_analyzer.anomalies import Spread, spread
from traj_analyzer.cli import main
from traj_analyzer.evaluation import load_evaluation
from traj_analyzer.extract import coverage, extract, in_scope
from traj_analyzer.ingest import ingest, load_index
from traj_analyzer.mining import MINED, load_code_operator, mine
from traj_analyzer.mining.functions import Candidate, Definition, Finding, Hypothesis, Revision
from traj_analyzer.operators.base import FeatureSpec
from traj_analyzer.operators.catalog import load_groups
from traj_analyzer.project import ConfigError, MiningConfig, Project
from traj_analyzer.runtime import mailbox, policy
from traj_analyzer.worker import registry

MINING = DATA / "mining"
REPLAY = MINING / "replay"


def mining_project(root: Path) -> Project:
    """A project over 6 Toucan and 8 UltraChat conversations with code features, configured for one small round."""
    assert main(["init", str(root)]) == 0

    def configure(config: dict[str, Any]) -> None:
        config["datasets"] = {
            "toucan": {**TOUCAN, "input": [str(MINING / "toucan.jsonl")],
                       "description": "Tool-use conversations from the public Toucan dataset."},
            "ultrachat": {**ULTRACHAT, "input": [str(MINING / "ultrachat.jsonl")],
                          "description": "Multi-turn chats from the public UltraChat dataset."},
        }
        config["operators"] = [{"use": "stats.basic"}, {"use": "stats.tool_usage"}]
        config["calls"] = {}
        config["engine"]["model"] = "DeepSeek-V4-flash"
        config["evaluation"] = {"size": 4, "model": "DeepSeek-V4-pro"}
        config["mining"] = {"starting": 4, "hypotheses": 4, "candidates": 4}

    edit_config(root, configure)
    project = Project.load(root)
    assert ingest(project) == {"toucan": 6, "ultrachat": 8}
    extract(project)
    return project


def call_key(call: Call) -> str:
    """Names the recording of a call by its function, request and instruction."""
    identity = json.dumps([call.function, call.request, call.instruction], sort_keys=True)
    return hashlib.sha256(identity.encode()).hexdigest()[:32]


class MiningReplay:
    """Answer every pending call with the submission a real model run gave to the same call."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def check_credentials(self, project: Project) -> None:
        pass

    def run(self, project: Project, count: int, functions: frozenset[str] = frozenset()) -> None:
        box = mailbox(project)
        self.directory.mkdir(parents=True, exist_ok=True)
        for call in box.pending():
            shutil.copytree(REPLAY / call_key(call), self.directory / call.id)
        Worker(mailbox=box, registry=registry(project), engine=ReplayEngine(self.directory),
               policy=policy(project), id="replay").work()


def test_mining_round_replays_real_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = mining_project(tmp_path / "proj")
    monkeypatch.chdir(project.root)
    # What a function left when it was stopped before it submitted goes before the function runs again.
    workspace = project.mining_dir / "1" / "workspace"
    for name in ("operators/leftover.py", "notes/mine-survey.md"):
        (workspace / name).parent.mkdir(parents=True, exist_ok=True)
        (workspace / name).write_text("planted\n", encoding="utf-8")
    (report,) = mine(project, runner=MiningReplay(tmp_path / "replay"))
    assert not (workspace / "operators" / "leftover.py").exists()
    assert not (workspace / "notes" / "mine-survey.md").exists()
    assert (workspace / "operators" / "batched_tool_calls.py").is_file()
    assert (report.round, len(report.starting), report.candidates) == (1, 4, 2)
    assert report.hypotheses == ["parallel_tool_call_batching", "tool_error_immediate_retry",
                                 "assistant_message_interleave_density"]
    # The expectation was seen broken once, so it was abandoned; the two measures became features.
    assert report.findings == {"supported": 2, "abandoned": 1}
    assert report.reviews == {"keep": 2}
    assert report.added == ["batched_tool_calls", "intermediate_assistant_messages"] and not report.removed
    batched, intermediate = report.verdicts
    assert (batched.reads, batched.hypothesis, batched.decision) == ("fields", "parallel_tool_call_batching", "add")
    # The operator has a value only for the six conversations with tool calls, and two of them differ.
    assert (batched.noise, batched.distinct, batched.varying_share) == (0.0, 2, 0.3333)
    # The tail check found the tails of both measures not worth a look, so neither raises alerts.
    assert (batched.tail_precision, batched.alert) == (0.25, False)
    assert (intermediate.tail_precision, intermediate.alert, intermediate.distinct) == (0.0, False, 3)

    # The workspace holds what the driver prepared and the accepted result of every function.
    evaluation = load_evaluation(Project.load(project.root))
    held_out = set(evaluation.evaluation_set)
    readable = {f"{path.parent.name}/{path.stem}" for path in (workspace / "trajectories").glob("*/*.md")}
    assert len(readable) == 10 and not readable & held_out
    steps = pq.read_table(workspace / "tables" / "steps.parquet")
    assert set(steps["key"].to_pylist()) == readable and "tool_call" in steps["kind"].to_pylist()
    assert set(pq.read_table(workspace / "tables" / "trajectories.parquet")["key"].to_pylist()) == readable
    assert {p.stem for p in (workspace / "tables").glob("*.parquet")} >= {"steps", "trajectories", "events", "evidence"}
    for name in ("README.md", "query.py", "features.csv", "features.md", "start.txt", "operators/rules.py",
                 "operators/example_rules.py", "operators/tables.py"):
        assert (workspace / name).is_file()
    assert "tables/steps.parquet" in (workspace / "README.md").read_text(encoding="utf-8")
    assert {"batch", "stratum"} <= set(pd.read_csv(workspace / "features.csv").columns)
    hypotheses = json.loads((workspace / "hypotheses.json").read_text(encoding="utf-8"))
    assert all({o["value"] for o in h["observed"]} == {True, False} and h["scope"] for h in hypotheses)
    assert [h["kind"] for h in hypotheses] == ["measure", "expectation", "measure"]
    findings = json.loads((workspace / "findings.json").read_text(encoding="utf-8"))
    assert [(f["shows"], f["lacks"]) for f in findings if f["status"] == "supported"] == [(3, 2), (2, 2)]
    assert [r["decision"] for r in json.loads((workspace / "reviews.json").read_text(encoding="utf-8"))] == [
        "keep", "keep"]
    checks = json.loads((workspace / "checks.json").read_text(encoding="utf-8"))
    assert [check["decision"] for check in checks] == ["add", "add"]
    assert (workspace / "cases" / "toucan").is_dir()
    assert project.scenario_path.read_text(encoding="utf-8") == (workspace / "scenario.md").read_text(encoding="utf-8")

    mined = Project.load(project.root)
    groups = [f"{MINED}-batched_tool_calls", f"{MINED}-intermediate_assistant_messages"]
    assert [g.name for g in load_groups(mined) if g.name.startswith(MINED)] == groups
    assert (mined.operators_dir / MINED / "batched_tool_calls.py").is_file()
    assert coverage(mined)[groups[0]] == {"extracted": 14, "missing": 0, "ok": 14}
    uses = {use.use: use for use in mined.config.operators}
    assert uses[f"{MINED}.batched_tool_calls"].tail_precision == 0.25 and not uses[f"{MINED}.batched_tool_calls"].alert
    assert not mined.trial_path.exists()
    assert json.loads((mined.mining_dir / "1" / "report.json").read_text(encoding="utf-8"))["added"] == report.added

    card = next(card for card in evaluation.features if card.feature == "batched_tool_calls")
    assert (card.kind, card.noise, card.coverage, card.distinct) == ("code", 0.0, 6, 2)


def test_spread_finds_too_little_variation_and_confinement() -> None:
    config = MiningConfig()
    feature = FeatureSpec(name="user_thanks", type="boolean", description="The user writes thanks.")
    keys = [f"d/{n}" for n in range(100)]
    context = pd.DataFrame({"repo": ["a"] * 10 + ["b"] * 90, "dataset": ["d"] * 100}, index=keys)

    def judged(true_keys: list[str]) -> Spread:
        return spread(feature, {key: key in true_keys for key in keys}, context, config)

    assert judged([]).objection(config) == "it takes one value on every trajectory it was extracted from"
    rare = judged(keys[50:51])
    assert (rare.varying, rare.varying_share) == (1, 0.01) and "below mining.varying_min" in rare.objection(config)
    # Every trajectory that differs is of repository a, which a tenth of the trajectories are.
    confined = judged(keys[:8])
    assert confined.confined is not None and (confined.confined.context, confined.confined.value) == ("repo", "a")
    assert (confined.confined.share, confined.confined.base) == (1.0, 0.1) and "repo = a" in confined.objection(config)
    # The same number of differing trajectories across both repositories is no objection, and neither is a single
    # dataset, which every trajectory shares.
    mixed = judged([*keys[:4], *keys[50:54]])
    assert mixed.confined is None and mixed.objection(config) is None


def test_scope_selects_the_trajectories_a_feature_applies_to(tmp_path: Path) -> None:
    project = mining_project(tmp_path / "proj")
    rows = load_index(project)
    scoped = FeatureSpec(name="tool_reply_used", type="boolean", description="The reply uses a tool result.",
                         scope="n_tool_calls > 0")
    everywhere = FeatureSpec(name="user_thanks", type="boolean", description="The user writes thanks.")
    inside = in_scope(project, [scoped, everywhere], rows)
    assert set(inside) == {"tool_reply_used"}
    matching = inside["tool_reply_used"]
    assert {key.split("/")[0] for key in matching} == {"toucan"} and len(matching) == 6
    with pytest.raises(ValidationError, match="stays empty for reads=fields"):
        Candidate(spec=scoped.model_copy(update={"type": "scalar"}), reads="fields", hypothesis="h", source="x = 1",
                  examples=[{"file": "d/1.md", "steps": "#2", "value": 1.0}])


def test_proposals_check_example_values_and_revisions() -> None:
    boolean = FeatureSpec(name="user_thanks", type="boolean", description="The user writes thanks.")
    example = {"file": "d/1.md", "steps": "#2", "value": True}
    Candidate(spec=boolean, reads="text", hypothesis="thanks", examples=[example])
    with pytest.raises(ValidationError, match="does not fit the feature type"):
        Candidate(spec=boolean, reads="text", hypothesis="thanks", examples=[{**example, "value": ["a"]}])
    count = FeatureSpec(name="turns", type="scalar", description="Number of turns.")
    with pytest.raises(ValidationError, match="boolean, category or set"):
        Definition(spec=count, reads="text")
    # A code candidate carries its operator file, and a text candidate carries none.
    Candidate(spec=count, reads="fields", hypothesis="length", examples=[{**example, "value": 3.0}], source="x = 1")
    with pytest.raises(ValidationError, match="exactly when reads=fields"):
        Candidate(spec=count, reads="fields", hypothesis="length", examples=[{**example, "value": 3.0}])
    with pytest.raises(ValidationError, match="exactly when reads=fields"):
        Candidate(spec=boolean, reads="text", hypothesis="thanks", examples=[example], source="x = 1")
    # A hypothesis rests on a trajectory that shows the behaviour and one that does not.
    stated = {"name": "user_thanks", "kind": "measure", "behaviour": "The user thanks.", "reason": "It ends a task.",
              "signal": "Thanks.", "scope": "Every trajectory."}
    Hypothesis(**stated, observed=[example, {**example, "file": "d/2.md", "value": False}])
    with pytest.raises(ValidationError, match="one that does not"):
        Hypothesis(**stated, observed=[example, {**example, "file": "d/2.md"}])
    # A finding seen once is abandoned; a supported one states counts, spread and consequence.
    Finding(hypothesis="thanks", status="abandoned", statement="One user writes thanks.", shows=1, lacks=9)
    found = {"hypothesis": "thanks", "status": "supported", "statement": "Some users write thanks."}
    with pytest.raises(ValidationError, match="has examples and states its spread"):
        Finding(**found, shows=3, lacks=5)
    stated_finding = {**found, "spread": "Both datasets.", "consequence": "Nothing else differs.",
                      "examples": [example]}
    Finding(**stated_finding, shows=3, lacks=5)
    with pytest.raises(ValidationError, match="return the finding as abandoned"):
        Finding(**stated_finding, shows=1, lacks=9)
    definition = Definition(spec=boolean, reads="text")
    Revision(action="rewrite", definitions=[definition])
    with pytest.raises(ValidationError, match="at least two"):
        Revision(action="split", definitions=[definition])
    with pytest.raises(ValidationError, match="spec has labels"):
        Revision(action="labels", definitions=[definition])
    with pytest.raises(ValidationError, match="a reason"):
        Revision(action="drop")


def test_mine_needs_a_second_model(tmp_path: Path) -> None:
    project = mining_project(tmp_path / "proj")
    edit_config(project.root, lambda config: config.pop("evaluation"))
    with pytest.raises(ConfigError, match="evaluation.model is not set"):
        mine(Project.load(project.root))


def test_code_operator_files_must_match_the_candidate(tmp_path: Path) -> None:
    spec = FeatureSpec(name="tool_share", type="scalar", range=(0, 1), description="Share of steps that call tools.")
    path = DATA / "operators" / "fixture" / "tool_share.py"
    assert load_code_operator(path, spec).kind == "code"
    with pytest.raises(ValueError, match="outputs differ"):
        load_code_operator(path, spec.model_copy(update={"description": "Another question."}))
    with pytest.raises(ValueError, match="code operator"):
        load_code_operator(DATA / "operators" / "fixture" / "outcome.py", spec)
