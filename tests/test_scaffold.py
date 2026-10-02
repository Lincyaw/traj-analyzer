from pathlib import Path

import pytest

from traj_analyzer.scaffold import InitError, init_project


def test_init_shares_complete_skill_directories_after_moving_project(tmp_path: Path) -> None:
    project = tmp_path / "analysis"
    result = init_project(project)
    source = Path(__file__).parents[1] / ".claude" / "skills"
    for skill in source.iterdir():
        if not skill.is_dir():
            continue
        shared = project / ".claude" / "skills" / skill.name
        codex = project / ".agents" / "skills" / skill.name
        assert codex.is_symlink() and not codex.readlink().is_absolute()
        assert codex.resolve() == shared.resolve()
        assert (codex / "SKILL.md").read_bytes() == (skill / "SKILL.md").read_bytes()
        assert str(codex.relative_to(project)) in result["files"]
        reference = shared / "references" / "project.md"
        reference.parent.mkdir()
        reference.write_text("Project analysis conventions.\n", encoding="utf-8")
        assert (codex / "references" / "project.md").read_bytes() == reference.read_bytes()
    moved = tmp_path / "moved"
    project.rename(moved)
    for skill in source.iterdir():
        if skill.is_dir():
            assert (moved / ".agents" / "skills" / skill.name / "SKILL.md").read_bytes() == (
                skill / "SKILL.md").read_bytes()


def test_force_reuses_existing_shared_skill_links(tmp_path: Path) -> None:
    init_project(tmp_path)
    links = {path.name: path.readlink() for path in (tmp_path / ".agents" / "skills").iterdir()}
    init_project(tmp_path, force=True)
    assert {path.name: path.readlink() for path in (tmp_path / ".agents" / "skills").iterdir()} == links


def test_init_rejects_custom_skill_conflicts_before_copying_templates(tmp_path: Path) -> None:
    custom = tmp_path / ".agents" / "skills" / "traj-report"
    custom.mkdir(parents=True)
    for force in (False, True):
        with pytest.raises(InitError, match="already exists"):
            init_project(tmp_path, force=force)
        assert custom.is_dir() and not custom.is_symlink()
        assert not (tmp_path / "traj.yaml").exists()
