from __future__ import annotations

import json
from typing import Any

import pyarrow as pa
from sqlglot import exp

from traj_analyzer.harness import Harness
from traj_analyzer.layers.base import Layer
from traj_analyzer.layers.tables import Tables, column
from traj_analyzer.schema import Trajectory
from traj_analyzer.sql import Canonical, literal_template, parse, resolve, shape

DIALECT = "duckdb"
SQL_KEYS = ("sql", "query", "statement", "sql_query")
"""Argument keys whose text is read as SQL when it parses."""
QUERIES = pa.schema([
    column("step", pa.int64(), "The step of the tool call that holds the query."),
    column("tool", pa.string(), "The tool called."),
    column("argument", pa.string(), "The argument key the query was written in."),
    column("parsed", pa.bool_(), "Whether sqlglot parsed the query; the other columns are empty when it did not."),
    column("kind", pa.string(), "The statement kind, such as select, with or union."),
    column("tables", pa.list_(pa.string()), "What the query reads: table names and the file paths of table functions."),
    column("scope", pa.list_(pa.string()), "filter, group and join columns, as <role>:<column>."),
    column("measures", pa.list_(pa.string()), "Columns inside aggregates."),
    column("aggregates", pa.list_(pa.string()), "The aggregates, as <aggregate>:<column>."),
    column("shown", pa.list_(pa.string()), "Columns shown or ordered by, as <role>:<column>."),
    column("values", pa.list_(pa.string()), "String values filtered on, as <column> <op> <value>."),
    column("template", pa.string(), "The query with literals replaced by placeholders."),
    column("canonical", pa.string(), "The query without names and the order of commutative parts; equal text asks "
                                     "the same thing."),
    column("text", pa.string(), "The query as written."),
])


def _sql_arguments(content: str) -> list[tuple[str, str]]:
    text = content.lstrip()
    if not text.startswith("{"):
        return []
    try:
        arguments = json.loads(content)
    except json.JSONDecodeError:
        return []
    if not isinstance(arguments, dict):
        return []
    return [(key, arguments[key]) for key in SQL_KEYS if isinstance(arguments.get(key), str) and arguments[key].strip()]


def compute(trajectory: Trajectory, tables: Tables, harness: Harness) -> dict[str, list[dict[str, Any]]]:
    rows = []
    for step in trajectory.steps:
        if step.kind != "tool_call":
            continue
        for argument, text in _sql_arguments(step.content):
            tree = parse(text, DIALECT)
            row: dict[str, Any] = {"step": step.index, "tool": step.name, "argument": argument,
                                   "parsed": tree is not None, "text": text}
            if tree is None or not isinstance(tree, exp.Query):
                row["parsed"] = False
                rows.append(row)
                continue
            found = shape(tree)
            row.update(kind=type(tree).__name__.lower(), tables=list(found.tables), scope=list(found.scope),
                       measures=list(found.measures), aggregates=list(found.aggregates), shown=list(found.shown),
                       values=list(found.values), template=literal_template(tree, DIALECT),
                       canonical=Canonical(resolve(tree, DIALECT)).text)
            rows.append(row)
    return {"queries": rows}


LAYER = Layer(
    name="sql",
    description="Every SQL query a tool call carries, parsed into what it reads, filters, aggregates and shows.",
    schemas={"queries": QUERIES},
    compute=compute,
)
