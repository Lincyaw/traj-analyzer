from __future__ import annotations

from pathlib import Path

import pytest

from traj_analyzer.cli import main
from traj_analyzer.harness import EMPTY, Harness, discover_harnesses, harness_for
from traj_analyzer.project import ConfigError, Project
from traj_analyzer.schema import Trajectory


def test_profiles_come_from_the_library_and_the_project(tmp_path: Path) -> None:
    builtin = discover_harnesses(None)
    assert set(builtin) == {"claude_code"} and builtin["claude_code"].tool_acts["bash"] == "shell"
    root = tmp_path / "proj"
    assert main(["init", str(root)]) == 0
    (root / "harness").mkdir()
    (root / "harness" / "claude_code.py").write_text(
        "from traj_analyzer.harness import Harness\n"
        "HARNESS = Harness(name='claude_code', tool_acts={'bash': 'shell'})\n", encoding="utf-8")
    (root / "harness" / "codex.py").write_text(
        "from traj_analyzer.harness import Harness\n"
        "HARNESS = Harness(name='codex', tool_acts={'exec_command': 'shell'}, command_keys=('cmd',))\n",
        encoding="utf-8")
    found = discover_harnesses(Project.load(root))
    assert set(found) == {"claude_code", "codex"} and found["claude_code"].tool_acts == {"bash": "shell"}
    assert found["codex"].command_keys == ("cmd",)
    (root / "harness" / "other.py").write_text(
        "from traj_analyzer.harness import Harness\nHARNESS = Harness(name='else')\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="named after the file"):
        discover_harnesses(Project.load(root))


def test_a_trajectory_gets_the_profile_its_adapter_names() -> None:
    found = discover_harnesses(None)
    assert harness_for(Trajectory(id="t", dataset="d"), found) is EMPTY
    assert harness_for(Trajectory(id="t", dataset="d", harness="claude_code"), found) is found["claude_code"]
    with pytest.raises(ConfigError, match="harness/codex.py"):
        harness_for(Trajectory(id="t", dataset="d", harness="codex"), found)


def test_merged_profiles_join_their_tables() -> None:
    base = Harness(name="a", tool_acts={"bash": "shell"}, rejected=("No",), error_marks={"timeout": ("timed out",)})
    extra = Harness(name="b", tool_acts={"exec": "shell"}, rejected=("No", "Never"),
                    error_marks={"timeout": ("slow",), "denied": ("Hook",)})
    merged = base.merged(extra, "c")
    assert merged.name == "c" and merged.tool_acts == {"bash": "shell", "exec": "shell"}
    assert merged.rejected == ("No", "Never")
    assert merged.error_marks == {"timeout": ("timed out", "slow"), "denied": ("Hook",)}
    assert merged.exit_code is None and merged.merged(EMPTY, "d").rejected == ("No", "Never")
