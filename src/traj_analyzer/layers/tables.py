from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from traj_analyzer.project import ConfigError, Project, RenderConfig
from traj_analyzer.schema import Trajectory

KEY = "key"
FLUSH_ROWS = 4096
"""Rows a writer holds as Python objects before it turns them into an Arrow chunk."""


def column(name: str, type_: pa.DataType, description: str) -> pa.Field:
    """A column with the description every table of the project carries for the people and agents that query it."""
    return pa.field(name, type_, metadata={"description": description})


def description(field_: pa.Field) -> str:
    metadata = field_.metadata or {}
    return metadata.get(b"description", b"").decode()


STEPS = pa.schema([
    column("step", pa.int64(), "The number after # in the step heading of the trajectory file."),
    column("role", pa.string(), "user, assistant, tool or system."),
    column("kind", pa.string(), "message, thinking, tool_call, tool_result or notice."),
    column("name", pa.string(), "The tool of a tool call or tool result, or the name of a notice."),
    column("is_error", pa.bool_(), "Whether a tool result is marked as an error."),
    column("turn", pa.int64(), "How many messages the user had typed before this step."),
    column("timestamp", pa.string(), "When the step happened, when the adapter knows."),
    column("chars", pa.int64(), "Length of the whole content."),
    column("content", pa.string(), "The text, cut at the length the trajectory files cut it."),
])
TRAJECTORIES = pa.schema([
    column("dataset", pa.string(), "The dataset the trajectory was ingested into."),
    column("id", pa.string(), "The trajectory id inside its dataset."),
    column("steps", pa.int64(), "Number of steps."),
    column("turns", pa.int64(), "Number of messages the user typed."),
    column("metadata", pa.string(), "The metadata of the trajectory as JSON text."),
])
EVIDENCE = pa.schema([
    column("feature", pa.string(), "The feature whose value the steps support."),
    column("step", pa.int64(), "A step the value rests on."),
    column("note", pa.string(), "What the step shows, when the operator says."),
])
EVENTS = pa.schema([
    column("rule", pa.string(), "The event feature, named after the rule that found the event."),
    column("step", pa.int64(), "The step where the expectation was broken."),
    column("end", pa.int64(), "The last step of the event, when it spans several."),
    column("edge", pa.string(), "The interaction the event lies on: owner, tool, local_env, external_env, context, "
                                "memory, peer, grader or third_party."),
    column("objects", pa.list_(pa.string()), "What the event concerns, such as files or commands."),
    column("recovered", pa.bool_(), "Whether a later action made up for the event."),
    column("recovered_at", pa.int64(), "The step of the action that made up for it."),
    column("note", pa.string(), "What was expected and what happened instead."),
])
JUDGMENTS = pa.schema([
    column("group", pa.string(), "The call group of judgment operators that answered."),
    column("unit", pa.string(), "turn or window."),
    column("index", pa.int64(), "The position of the unit in the trajectory, counting from 0."),
    column("start", pa.int64(), "The first step of the unit."),
    column("end", pa.int64(), "The last step of the unit."),
    column("feature", pa.string(), "The per-unit output."),
    column("value", pa.string(), "The value as JSON text."),
    column("evidence", pa.string(), "The steps the model cited, when the group keeps evidence."),
])
BUILTIN = {"steps": STEPS, "trajectories": TRAJECTORIES, "evidence": EVIDENCE, "events": EVENTS,
           "judgments": JUDGMENTS}
"""Tables every project has, besides those of its parsing layers."""


def with_key(schema: pa.Schema) -> pa.Schema:
    return pa.schema([column(KEY, pa.string(), "Trajectory key, <dataset>/<id>."), *schema])


def steps_rows(trajectory: Trajectory, render: RenderConfig) -> list[dict[str, Any]]:
    rows = []
    turn = 0
    for step in trajectory.steps:
        if step.is_user_message:
            turn += 1
        rows.append({"step": step.index, "role": step.role, "kind": step.kind, "name": step.name,
                     "is_error": step.is_error, "turn": turn, "timestamp": step.timestamp,
                     "chars": len(step.content), "content": step.content[:render.max_step_chars]})
    return rows


def trajectory_row(trajectory: Trajectory) -> dict[str, Any]:
    return {"dataset": trajectory.dataset, "id": trajectory.id, "steps": len(trajectory.steps),
            "turns": sum(1 for step in trajectory.steps if step.is_user_message),
            "metadata": json.dumps(trajectory.metadata, ensure_ascii=False, sort_keys=True, default=str)}


@dataclass(frozen=True)
class Tables:
    """The rows of every table for one trajectory, one Arrow table per table name.

    `tables[name]` gives the rows as a pandas frame, where a missing value reads as NaN or None by the column type;
    `tables.records(name)` gives them as dicts, where a missing value is None.
    """

    arrow: dict[str, pa.Table] = field(default_factory=dict)
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)

    @classmethod
    def of(cls, rows: dict[str, list[dict[str, Any]]], schemas: dict[str, pa.Schema]) -> Tables:
        return cls({name: pa.Table.from_pylist(rows.get(name, []), schema=schema) for name, schema in schemas.items()})

    def _table(self, name: str) -> pa.Table:
        if name not in self.arrow:
            raise KeyError(f"no table {name!r} is computed for this trajectory; the operator declares the tables "
                           f"it reads in `tables`, and traj.yaml enables the layer that writes them")
        return self.arrow[name]

    def __getitem__(self, name: str) -> pd.DataFrame:
        if name not in self.frames:
            self.frames[name] = self._table(name).to_pandas()
        return self.frames[name]

    def __contains__(self, name: str) -> bool:
        return name in self.arrow

    def records(self, name: str) -> list[dict[str, Any]]:
        return self._table(name).to_pylist()


def table_path(project: Project, name: str) -> Path:
    return project.tables_dir / f"{name}.parquet"


def read_table(project: Project, name: str) -> pa.Table:
    path = table_path(project, name)
    if not path.is_file():
        raise ConfigError(f"No table {name}; run `traj extract`")
    return pq.read_table(path)


def table_names(project: Project) -> list[str]:
    return sorted(path.stem for path in project.tables_dir.glob("*.parquet")) if project.tables_dir.is_dir() else []


class TableWriter:
    """Collects the rows of several tables over one run and writes each table once, replacing the rows of the
    trajectories the run covered and keeping the rows of the others."""

    def __init__(self, project: Project, schemas: dict[str, pa.Schema]) -> None:
        self.project = project
        self.schemas = {name: with_key(schema) for name, schema in schemas.items()}
        self.chunks: dict[str, list[pa.Table]] = {name: [] for name in schemas}
        self.pending: dict[str, list[dict[str, Any]]] = {name: [] for name in schemas}
        self.keys: set[str] = set()

    def add(self, key: str, rows: dict[str, list[dict[str, Any]]]) -> None:
        self.keys.add(key)
        for name, table_rows in rows.items():
            if name not in self.schemas:
                raise ValueError(f"rows for a table without a schema: {name}")
            pending = self.pending[name]
            pending.extend({KEY: key, **row} for row in table_rows)
            if len(pending) >= FLUSH_ROWS:
                self._flush(name)

    def _flush(self, name: str) -> None:
        if self.pending[name]:
            self.chunks[name].append(pa.Table.from_pylist(self.pending[name], schema=self.schemas[name]))
            self.pending[name] = []

    def finish(self, scopes: dict[str, tuple[str, set[str]]] | None = None) -> None:
        """Write every table. A table named in `scopes` keeps the old rows of the covered trajectories whose value
        of the scope column is not among the values this run produced, since other runs write such rows too."""
        self.project.tables_dir.mkdir(parents=True, exist_ok=True)
        for name, schema in self.schemas.items():
            self._flush(name)
            new = pa.concat_tables(self.chunks[name]) if self.chunks[name] else schema.empty_table()
            path = table_path(self.project, name)
            if path.is_file():
                old = pq.read_table(path)
                if old.schema.names == schema.names:
                    covered = pc.is_in(old[KEY], value_set=pa.array(sorted(self.keys), pa.string()))
                    if scopes and name in scopes:
                        column, values = scopes[name]
                        covered = pc.and_(covered, pc.is_in(old[column], value_set=pa.array(sorted(values),
                                                                                              pa.string())))
                    kept = old.filter(pc.invert(pc.fill_null(covered, False)))
                    new = pa.concat_tables([kept.cast(schema), new])
            pq.write_table(new, path)
