from __future__ import annotations

import shutil
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any


# Packaging drops files whose names start with a dot, so templates store these without it.
_DOTTED = {"gitignore": ".gitignore", "claude": ".claude"}


class InitError(ValueError):
    pass


def init_project(directory: Path, *, force: bool = False) -> dict[str, Any]:
    """Create a project with shared skill directories for Claude Code and Codex."""
    target = directory.expanduser().resolve()
    if (target / "traj.yaml").exists() and not force:
        raise InitError(f"{target / 'traj.yaml'} already exists; use --force to overwrite")
    written = []
    with as_file(files("traj_analyzer") / "templates" / "project") as root:
        codex_links = {
            target / ".agents" / "skills" / skill.name: Path("../../.claude/skills") / skill.name
            for skill in sorted((root / "claude" / "skills").iterdir())
            if skill.is_dir() and (skill / "SKILL.md").is_file()
        }
        for destination, link in codex_links.items():
            if destination.is_symlink() and destination.readlink() == link:
                continue
            if destination.exists() or destination.is_symlink():
                raise InitError(f"{destination} already exists; keep custom skills outside the bundled skill paths")
        for path in sorted(root.rglob("*")):
            if path.is_dir() or "__pycache__" in path.parts:
                continue
            relative = Path(*(_DOTTED.get(p, p) for p in path.relative_to(root).parts))
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            written.append(str(relative))
        for destination, link in codex_links.items():
            if not destination.is_symlink():
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.symlink_to(link, target_is_directory=True)
            written.append(str(destination.relative_to(target)))
    return {"project": str(target), "files": written}
