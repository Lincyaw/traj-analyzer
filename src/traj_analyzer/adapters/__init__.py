from __future__ import annotations

import importlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from traj_analyzer.files import load_module
from traj_analyzer.project import ConfigError
from traj_analyzer.schema import Trajectory

BUILTIN = Path(__file__).parent


class Adapter(Protocol):
    """Reads one input file into trajectories; constructed as `ADAPTER(**options)` from the dataset config."""

    def read(self, path: Path, dataset: str) -> Iterable[Trajectory]: ...


@dataclass(frozen=True)
class AdapterRef:
    name: str
    origin: Literal["builtin", "project"]
    path: Path
    adapter: type[Any]


def discover_adapters(root: Path | None) -> dict[str, AdapterRef]:
    """Adapters of the package and, when given, of the project's adapters/ directory; project adapters override.

    An adapter is a file `<name>.py` that defines `ADAPTER`, the adapter class; files starting with an underscore
    hold shared code.
    """
    places: list[tuple[Literal["builtin", "project"], Path]] = [("builtin", BUILTIN)]
    if root is not None and (root / "adapters").is_dir():
        places.append(("project", root / "adapters"))
    refs: dict[str, AdapterRef] = {}
    for origin, directory in places:
        for path in sorted(directory.glob("*.py")):
            if path.name.startswith("_"):
                continue
            adapter = getattr(load_module(path), "ADAPTER", None)
            if not isinstance(adapter, type):
                raise ConfigError(f"{path} must define ADAPTER as the adapter class")
            refs[path.stem] = AdapterRef(name=path.stem, origin=origin, path=path, adapter=adapter)
    return refs


def load_adapter(spec: str, options: dict[str, Any], root: Path) -> Adapter:
    """Build the adapter named `spec`: a project or built-in adapter name, or `module:Class` of an installed package."""
    location, sep, name = spec.partition(":")
    if sep:
        cls = getattr(importlib.import_module(location), name)
    else:
        refs = discover_adapters(root)
        if spec not in refs:
            raise ConfigError(f"Unknown adapter {spec}; see `traj adapters list`")
        cls = refs[spec].adapter
    adapter: Adapter = cls(**options)
    return adapter
