from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from aifn import Call, Worker
from aifn.engines import ReplayEngine
from pydantic import ValidationError
from test_pipeline import DATA, TOUCAN, ULTRACHAT, edit_config

from traj_analyzer.cli import main
from traj_analyzer.evaluation import load_evaluation
from traj_analyzer.extract import coverage, extract
from traj_analyzer.ingest import ingest
from traj_analyzer.mining import MINED, load_code_operator, mine
from traj_analyzer.operators.base import FeatureSpec
from traj_analyzer.operators.catalog import load_groups
from traj_analyzer.project import ConfigError, Project
from traj_analyzer.proposals import Definition, Proposal, Revision
from traj_analyzer.runtime import mailbox, policy, registry

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
        config["mining"] = {"singles": 2, "pairs": 2, "candidates": 4}

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

    def check(self, project: Project) -> None:
        pass

    def run(self, project: Project, count: int) -> None:
        box = mailbox(project)
        self.directory.mkdir(parents=True, exist_ok=True)
        for call in box.pending():
            shutil.copytree(REPLAY / call_key(call), self.directory / call.id)
        Worker(mailbox=box, registry=registry(project), engine=ReplayEngine(self.directory),
               policy=policy(project), id="replay").work()


def test_mining_round_replays_real_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = mining_project(tmp_path / "proj")
    monkeypatch.chdir(project.root)
    (report,) = mine(project, runner=MiningReplay(tmp_path / "replay"))
    assert (report.round, len(report.singles), len(report.pairs), report.proposals) == (1, 2, 2, 12)
    assert not report.failed_readings and not report.removed
    assert report.added == ["final_summary_present", "multiple_user_refinement_requests"]
    verdicts = {(v.feature, v.parent): v for v in report.verdicts}
    assert verdicts["follows_user_request_order", None].reason.startswith("it takes one value")
    # The noisy candidate was rewritten once, and the rewrite was still noisy.
    first = verdicts["assistant_response_format", None]
    rewrite = verdicts["assistant_response_format", "assistant_response_format"]
    assert (first.decision, rewrite.decision, rewrite.noise) == ("reject", "reject", 0.25)
    assert "above the limit" in first.reason and all(v.compared == 4 for v in report.verdicts)

    mined = Project.load(project.root)
    (group,) = load_groups(mined, [MINED])
    assert [f.name for f in group.features] == report.added
    assert coverage(mined)[MINED] == {"extracted": 14, "missing": 0, "ok": 14}
    assert not mined.trial_path.exists()
    assert json.loads((mined.mining_dir / "1" / "report.json").read_text(encoding="utf-8"))["added"] == report.added

    evaluation = load_evaluation(mined)
    cards = {card.feature: card for card in evaluation.features}
    assert evaluation.measured == {MINED: evaluation.evaluation_set} and len(evaluation.evaluation_set) == 4
    for name in report.added:
        assert (cards[name].kind, cards[name].compared, cards[name].coverage) == ("llm", 4, 14)
        assert cards[name].noise is not None and cards[name].noise <= 0.1 and cards[name].distinct == 2


def test_proposals_check_example_values_and_revisions() -> None:
    boolean = FeatureSpec(name="user_thanks", type="boolean", description="The user writes thanks.")
    example = {"file": "d/1.md", "steps": "#2", "value": True}
    Proposal(spec=boolean, reads="text", examples=[example])
    with pytest.raises(ValidationError, match="does not fit the feature type"):
        Proposal(spec=boolean, reads="text", examples=[{**example, "value": ["a"]}])
    count = FeatureSpec(name="turns", type="scalar", description="Number of turns.")
    with pytest.raises(ValidationError, match="boolean, category or set"):
        Definition(spec=count, reads="text")
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
