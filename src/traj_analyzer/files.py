from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

from ruamel.yaml import YAML


def read_yaml(path: Path) -> Any:
    return YAML(typ="safe").load(path)


def parse_yaml(text: str) -> Any:
    return YAML(typ="safe").load(text)


def iter_jsonl(path: Path) -> Iterator[Any]:
    """Parse one JSON value per line, naming the file and line of a corrupt line."""
    with path.open(encoding="utf-8") as lines:
        for number, line in enumerate(lines, start=1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: {error}") from error


def write_jsonl(path: Path, values: Iterable[str]) -> None:
    """Write already serialized JSON values, one per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for value in values:
            out.write(value + "\n")


def load_module(path: Path) -> ModuleType:
    """Import a Python file that is not part of an installed package, such as a project adapter or operator."""
    name = f"traj_module_{hashlib.sha256(str(path).encode()).hexdigest()[:16]}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Pydantic resolves postponed annotations through sys.modules, so the module must be registered first.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
