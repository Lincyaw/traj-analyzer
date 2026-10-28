from __future__ import annotations

import re
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import sqlglot
from sqlglot import exp

from traj_analyzer.hunting.ledger import ProbeState
from traj_analyzer.layers import discover_layers
from traj_analyzer.layers.tables import column
from traj_analyzer.mining.operators import compute_operator, load_code_operator
from traj_analyzer.project import Project
from traj_analyzer.schema import Trajectory

COLUMNS = ["key", "step", "note"]
"""What a probe finds, one row per position."""
HITS = ["probe", *COLUMNS]
HITS_SCHEMA = pa.schema([("probe", pa.string()), ("key", pa.string()), ("step", pa.int64()), ("note", pa.string())])
"""Every position the probes of a round found, as `rounds/<round>/hits.parquet` of the workspace holds them."""
CONTENTS = "contents"
CONTENTS_SCHEMA = pa.schema([
    column("step", pa.int64(), "The number after # in the step heading of the trajectory file."),
    column("role", pa.string(), "user, assistant, tool or system."),
    column("kind", pa.string(), "message, thinking, tool_call, tool_result or notice."),
    column("name", pa.string(), "The tool of a tool call or tool result, or the name of a notice."),
    column("content", pa.string(), "The whole text of the step, never cut."),
    column("arguments", pa.string(), "The JSON arguments of a tool call, when the adapter parsed the call."),
])
TABLE = re.compile(r"tables/([a-z][a-z0-9_]*)\.parquet")
"""How a SQL probe names a table."""


class ProbeError(ValueError):
    """A probe whose result is not one row per position of a trajectory of the hunt."""


def probe_path(workspace: Path, probe: ProbeState) -> Path:
    return workspace / "probes" / f"{probe.name}.{'sql' if probe.kind == 'sql' else 'py'}"


def contents_rows(trajectory: Trajectory) -> list[dict[str, object]]:
    """The whole text of every step of the trajectory, which the steps table and the trajectory files cut."""
    return [{"step": step.index, "role": step.role, "kind": step.kind, "name": step.name, "content": step.content,
             "arguments": step.arguments} for step in trajectory.steps]


def readable_tables(project: Project) -> set[str]:
    """The tables a SQL probe reads: the steps, the trajectories, the whole text of the steps and the tables of the
    parsing layers, which the marks of a hunt compute for one trajectory at a time."""
    return {"steps", "trajectories", CONTENTS} | {table for ref in discover_layers(project).values()
                                                   for table in ref.layer.schemas}


def tables_read(text: str) -> set[str]:
    """The tables a SQL probe reads, by name; a probe names a table as `'tables/<name>.parquet'` and in no other way,
    such as a table function."""
    tree = sqlglot.parse_one(text, dialect="duckdb")
    ctes = {cte.alias_or_name for cte in tree.find_all(exp.CTE)}
    names = set()
    for table in tree.find_all(exp.Table):
        if isinstance(table.this, exp.Identifier) and not table.args.get("db") and not table.args.get("catalog"):
            if table.name in ctes:
                continue
            match = TABLE.fullmatch(table.name)
            if match:
                names.add(match.group(1))
                continue
        raise ProbeError(f"a SQL probe names its tables as 'tables/<name>.parquet' and nothing else; it reads "
                         f"{table.sql(dialect='duckdb')}")
    return names


def _one_select(connection: duckdb.DuckDBPyConnection, text: str) -> None:
    statements = connection.extract_statements(text)
    if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
        raise ProbeError("a SQL probe is one SELECT statement")


def _positions(connection: duckdb.DuckDBPyConnection, text: str) -> pd.DataFrame:
    frame = connection.sql(text).df()
    if sorted(frame.columns) != sorted(COLUMNS):
        raise ProbeError(f"a SQL probe returns the columns {COLUMNS}, this one returns {list(frame.columns)}")
    return frame[COLUMNS]


def run_sql(workspace: Path, text: str, keys: set[str], readable: set[str]) -> pd.DataFrame:
    """The positions one SQL probe finds over every trajectory of the hunt, with the workspace as the directory its
    table paths start from."""
    connection = duckdb.connect()
    connection.execute(f"SET file_search_path = '{workspace}'")
    _one_select(connection, text)
    unknown = sorted(tables_read(text) - readable)
    if unknown:
        raise ProbeError(f"a SQL probe reads only the tables {sorted(readable)}; this one reads {unknown}")
    return checked(_positions(connection, text), keys)


def mark_sql(text: str, relations: dict[str, pa.Table], key: str) -> pd.DataFrame:
    """The positions one SQL probe finds in one trajectory, whose tables `relations` holds with their key column;
    the probe reads nothing else."""
    connection = duckdb.connect()
    for name, table in relations.items():
        connection.register(f"tables/{name}.parquet", table)
    connection.execute("SET enable_external_access = false")
    _one_select(connection, text)
    missing = sorted(tables_read(text) - set(relations))
    if missing:
        raise ProbeError(f"the probe reads the tables {missing}, which are not computed for {key}; traj.yaml enables "
                         f"the layers that write them")
    return checked(_positions(connection, text), {key})


def run_code(project: Project, path: Path, name: str, keys: set[str]) -> pd.DataFrame:
    """The positions one code probe finds: the events of its one output on every trajectory of the hunt."""
    operator = load_code_operator(path)
    (feature,) = operator.outputs(operator.params())
    if feature.type != "event" or feature.name != name:
        raise ProbeError(f"a code probe has one output of type event named {name}; this one has {feature.type} "
                         f"{feature.name}")
    values, _ = compute_operator(project, operator, sorted(keys))
    rows = [{"key": key, "step": event["step"], "note": event["note"]}
            for key, events in values.items() for event in events]
    return checked(pd.DataFrame(rows, columns=COLUMNS), keys)


def checked(frame: pd.DataFrame, keys: set[str]) -> pd.DataFrame:
    if frame["key"].isna().any() or frame["step"].isna().any():
        raise ProbeError("every position has a key and a step")
    unknown = sorted(set(frame["key"]) - keys)
    if unknown:
        raise ProbeError(f"keys of no trajectory of the hunt: {unknown[:5]}")
    steps = pd.to_numeric(frame["step"])
    if not (steps == steps.round()).all():
        raise ProbeError("steps are whole numbers")
    found = pd.DataFrame({"key": frame["key"].astype(str), "step": steps.astype("int64"),
                          "note": frame["note"].fillna("").astype(str)})
    return found.drop_duplicates().sort_values(["key", "step", "note"]).reset_index(drop=True)


def run_probes(
    project: Project, workspace: Path, probes: list[ProbeState], keys: set[str]
) -> tuple[pd.DataFrame, dict[str, str]]:
    """The positions every probe finds, and the error of every probe that failed to run."""
    found = []
    failed: dict[str, str] = {}
    readable = readable_tables(project)
    for probe in probes:
        path = probe_path(workspace, probe)
        # A probe written by a model can fail in any way; every failure is recorded for the next proposal.
        try:
            if probe.kind == "sql":
                frame = run_sql(workspace, path.read_text(encoding="utf-8"), keys, readable)
            else:
                frame = run_code(project, path, probe.name, keys)
        except Exception as error:
            failed[probe.name] = f"{type(error).__name__}: {error}"
            continue
        found.append(frame.assign(probe=probe.name)[HITS])
    hits = pd.concat(found, ignore_index=True) if found else HITS_SCHEMA.empty_table().to_pandas()
    return hits, failed
