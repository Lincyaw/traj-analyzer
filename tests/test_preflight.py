from __future__ import annotations

from pathlib import Path

import pytest
from test_pipeline import edit_config

from traj_analyzer.cli import main
from traj_analyzer.preflight import CHECK, installed
from traj_analyzer.project import ConfigError, Project
from traj_analyzer.worker import registry


def test_the_check_names_what_a_worker_lacks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "proj"
    assert main(["init", str(root)]) == 0
    project = Project.load(root)
    assert CHECK in registry(project).functions
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="DEEPSEEK_API_KEY"):
        installed(project)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "unused")
    edit_config(root, lambda config: config["engine"].update(patches=["engine/missing.patch.yml"]))
    with pytest.raises(ConfigError, match="engine/missing.patch.yml"):
        installed(Project.load(root))
    edit_config(root, lambda config: config["engine"].update(patches=[], dsh_home=str(tmp_path / "dsh")))
    with pytest.raises(RuntimeError, match="python -m aifn install"):
        installed(Project.load(root))
