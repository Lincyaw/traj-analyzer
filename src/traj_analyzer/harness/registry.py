from __future__ import annotations

from pathlib import Path

from traj_analyzer.files import load_module
from traj_analyzer.harness.base import EMPTY, Harness
from traj_analyzer.project import ConfigError, Project
from traj_analyzer.schema import Trajectory

LIBRARY = Path(__file__).parent / "library"


def discover_harnesses(project: Project | None) -> dict[str, Harness]:
    """The profiles of the built-in library and, when given, of the project's harness/ directory; a project profile
    overrides a built-in one of the same name.

    A profile is a file `<name>.py` that defines `HARNESS`, a `Harness` whose `name` is the file stem.
    """
    roots = [LIBRARY]
    if project is not None and (project.root / "harness").is_dir():
        roots.append(project.root / "harness")
    found: dict[str, Harness] = {}
    for root in roots:
        for path in sorted(root.glob("*.py")):
            if path.name.startswith("_"):
                continue
            harness = getattr(load_module(path), "HARNESS", None)
            if not isinstance(harness, Harness):
                raise ConfigError(f"{path} must define HARNESS as a Harness")
            if harness.name != path.stem:
                raise ConfigError(f"{path} defines the harness {harness.name}, which must be named after the file")
            found[harness.name] = harness
    return found


def harness_for(trajectory: Trajectory, harnesses: dict[str, Harness]) -> Harness:
    """The profile of a trajectory: `EMPTY` when its adapter named no harness."""
    if trajectory.harness is None:
        return EMPTY
    if trajectory.harness not in harnesses:
        raise ConfigError(f"{trajectory.key} comes from the harness {trajectory.harness!r}, and no profile "
                          f"harness/{trajectory.harness}.py defines it")
    return harnesses[trajectory.harness]
