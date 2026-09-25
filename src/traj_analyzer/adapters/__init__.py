from __future__ import annotations

import importlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

from traj_analyzer.files import load_module
from traj_analyzer.project import ConfigError
from traj_analyzer.schema import Trajectory


class Adapter(Protocol):
    def read(self, path: Path, dataset: str) -> Iterable[Trajectory]: ...


BUILTIN = {
    "claude_code": "traj_analyzer.adapters.claude_code:ClaudeCodeAdapter",
    "messages": "traj_analyzer.adapters.messages:MessagesAdapter",
}


def load_adapter(spec: str, options: dict[str, Any], root: Path) -> Adapter:
    """Resolve a built-in name, `module:Class`, or `path/in/project.py:Class`."""
    target = BUILTIN.get(spec, spec)
    location, sep, name = target.partition(":")
    if not sep:
        raise ConfigError(f"Adapter must be a built-in name or 'module:Class': {spec}")
    is_file = location.endswith(".py")
    module = load_module((root / location).resolve()) if is_file else importlib.import_module(location)
    adapter: Adapter = getattr(module, name)(**options)
    return adapter
