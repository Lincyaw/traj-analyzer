from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from traj_analyzer.ingest import load_trajectory
from traj_analyzer.layers.tables import column, with_key
from traj_analyzer.project import Project
from traj_analyzer.schema import Trajectory

CONTENTS = "contents"
"""The table of a hunt's workspace holding the whole text of every step, which the steps table cuts."""
CONTENTS_SCHEMA = pa.schema([
    column("step", pa.int64(), "The number after # in the step heading of the trajectory file."),
    column("role", pa.string(), "user, assistant, tool or system."),
    column("kind", pa.string(), "message, thinking, tool_call, tool_result or notice."),
    column("name", pa.string(), "The tool of a tool call or tool result, or the name of a notice."),
    column("content", pa.string(), "The whole text of the step, never cut."),
    column("arguments", pa.string(), "The JSON arguments of a tool call, when the adapter parsed the call."),
])


def contents_rows(trajectory: Trajectory) -> list[dict[str, Any]]:
    """The whole text of every step of the trajectory, which the steps table and the trajectory files cut."""
    return [{"step": step.index, "role": step.role, "kind": step.kind, "name": step.name, "content": step.content,
             "arguments": step.arguments} for step in trajectory.steps]


def write_contents(project: Project, keys: list[str], path: Path) -> None:
    """The whole text of every step of these trajectories, as the table of a hunt's workspace."""
    rows = [{"key": key, **row} for key in keys for row in contents_rows(load_trajectory(project, key))]
    pq.write_table(pa.Table.from_pylist(rows, schema=with_key(CONTENTS_SCHEMA)), path)
