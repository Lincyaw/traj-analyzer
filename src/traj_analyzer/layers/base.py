from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa

from traj_analyzer.files import load_module
from traj_analyzer.harness import Harness
from traj_analyzer.layers.tables import BUILTIN, STEPS, TRAJECTORIES, Tables, description, steps_rows, trajectory_row
from traj_analyzer.project import ConfigError, Project, RenderConfig
from traj_analyzer.schema import Trajectory

LIBRARY = Path(__file__).parent / "library"
_NAME = re.compile(r"[a-z][a-z0-9_]*")


@dataclass(frozen=True)
class Layer:
    """One parsing layer: it reads a trajectory and the tables it requires, and writes rows into the tables it
    declares.

    Every layer file defines one as `LAYER`. `schemas` maps each table name to its columns, without the trajectory
    key, which the writer adds; a column carries its description in its metadata. `requires` names the tables of
    other layers the layer reads; `compute(trajectory, tables, harness)` gets `steps`, `trajectories`, those tables
    and the harness profile of the trajectory, and returns the rows of every table it declares for one trajectory.
    """

    name: str
    description: str
    schemas: dict[str, pa.Schema]
    compute: Callable[[Trajectory, Tables, Harness], dict[str, list[dict[str, Any]]]]
    requires: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _NAME.fullmatch(self.name):
            raise ValueError(f"layer name {self.name!r} must be snake_case")
        for table in self.schemas:
            if not _NAME.fullmatch(table):
                raise ValueError(f"{self.name}: table name {table!r} must be snake_case")
            if table in BUILTIN:
                raise ValueError(f"{self.name}: {table} is a table every project has")
        own = set(self.requires) & set(self.schemas)
        if own:
            raise ValueError(f"{self.name}: requires the tables it writes: {sorted(own)}")


@dataclass(frozen=True)
class LayerRef:
    name: str
    origin: Literal["library", "project"]
    path: Path
    layer: Layer


def discover_layers(project: Project | None) -> dict[str, LayerRef]:
    """The layers of the built-in library and, when given, of the project's layers/ directory; project layers
    override."""
    roots: list[tuple[Literal["library", "project"], Path]] = [("library", LIBRARY)]
    if project is not None and (project.root / "layers").is_dir():
        roots.append(("project", project.root / "layers"))
    refs: dict[str, LayerRef] = {}
    for origin, root in roots:
        for path in sorted(root.glob("*.py")):
            if path.name.startswith("_"):
                continue
            layer = getattr(load_module(path), "LAYER", None)
            if not isinstance(layer, Layer):
                raise ConfigError(f"{path} must define LAYER as a Layer")
            if layer.name != path.stem:
                raise ConfigError(f"{path} defines the layer {layer.name}, which must be named after the file")
            refs[layer.name] = LayerRef(name=layer.name, origin=origin, path=path, layer=layer)
    owners: dict[str, str] = {}
    for ref in refs.values():
        for table in ref.layer.schemas:
            if table in owners:
                raise ConfigError(f"Layers {owners[table]} and {ref.name} both write the table {table}")
            owners[table] = ref.name
    return refs


def _owners(refs: dict[str, LayerRef]) -> dict[str, str]:
    return {table: ref.name for ref in refs.values() for table in ref.layer.schemas}


def enabled_layers(project: Project, tables: set[str] | frozenset[str] = frozenset()) -> list[LayerRef]:
    """The layers `traj.yaml` enables, those that write the `tables` the enabled operators read, and the layers
    those require, in the order they run."""
    refs = discover_layers(project)
    unknown = sorted(set(project.config.layers) - set(refs))
    if unknown:
        raise ConfigError(f"Unknown layers in traj.yaml: {unknown}; see `traj layers list`")
    owners = _owners(refs)
    wanted = set(project.config.layers)
    for table in sorted(tables):
        if table in BUILTIN:
            continue
        if table not in owners:
            raise ConfigError(f"An enabled operator reads the table {table}, which no layer writes")
        wanted.add(owners[table])
    pending = sorted(wanted)
    while pending:
        name = pending.pop()
        for table in refs[name].layer.requires:
            if table in BUILTIN:
                continue
            if table not in owners:
                raise ConfigError(f"Layer {name} requires the table {table}, which no layer writes")
            if owners[table] not in wanted:
                wanted.add(owners[table])
                pending.append(owners[table])
    return ordered([refs[name] for name in sorted(wanted)])


def ordered(layers: list[LayerRef]) -> list[LayerRef]:
    """The layers in an order where every layer runs after the layers writing the tables it requires."""
    owners = _owners({ref.name: ref for ref in layers})
    placed: list[LayerRef] = []
    done: set[str] = set()
    visiting: set[str] = set()
    by_name = {ref.name: ref for ref in layers}

    def place(ref: LayerRef) -> None:
        if ref.name in done:
            return
        if ref.name in visiting:
            raise ConfigError(f"Layer {ref.name} requires a table that depends on its own tables")
        visiting.add(ref.name)
        for table in ref.layer.requires:
            if table in BUILTIN:
                continue
            if table not in owners:
                raise ConfigError(f"Layer {ref.name} requires the table {table}, which no enabled layer writes")
            place(by_name[owners[table]])
        visiting.discard(ref.name)
        done.add(ref.name)
        placed.append(ref)

    for ref in sorted(layers, key=lambda ref: ref.name):
        place(ref)
    return placed


def schemas_of(layers: list[LayerRef]) -> dict[str, pa.Schema]:
    return {table: schema for ref in layers for table, schema in ref.layer.schemas.items()}


def compute_tables(
    trajectory: Trajectory, layers: list[LayerRef], render: RenderConfig, harness: Harness
) -> tuple[dict[str, list[dict[str, Any]]], Tables]:
    """The rows of the steps and trajectories tables and of every layer for one trajectory, and the same as the
    `Tables` an operator reads.

    The layers run in the order of `ordered`, each seeing the tables of the layers before it, with the profile of
    the harness the trajectory comes from.
    """
    rows: dict[str, list[dict[str, Any]]] = {"steps": steps_rows(trajectory, render),
                                             "trajectories": [trajectory_row(trajectory)]}
    schemas = {"steps": STEPS, "trajectories": TRAJECTORIES}
    tables = Tables.of(rows, schemas)
    for ref in ordered(layers):
        rows.update(ref.layer.compute(trajectory, tables, harness))
        schemas.update(ref.layer.schemas)
        tables = Tables.of(rows, schemas)
    return rows, tables


def describe_tables(schemas: dict[str, pa.Schema]) -> str:
    """The tables and their columns as Markdown, for the people and agents that query them."""
    lines = []
    for name, schema in schemas.items():
        lines.append(f"- `{name}.parquet`: one row per {_row_name(name)}, with `key` and")
        lines += [f"  - `{field_.name}` ({field_.type}): {description(field_)}" for field_ in schema]
    return "\n".join(lines) + "\n"


def _row_name(table: str) -> str:
    names = {"steps": "step", "trajectories": "trajectory", "actions": "tool action", "relations": "relation between "
             "two actions", "queries": "SQL query", "turns": "message the user typed, with what followed it",
             "evidence": "step a feature value rests on", "events": "event", "judgments": "judged unit",
             "contents": "step, with its whole text", "hunt_findings": "trajectory a hunt confirmed"}
    return names.get(table, table.rstrip("s"))
