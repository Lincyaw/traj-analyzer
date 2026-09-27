from __future__ import annotations

import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import duckdb
import pyarrow as pa
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from traj_analyzer.features.table import KEY, read_group, status_column
from traj_analyzer.ingest import load_index
from traj_analyzer.operators.base import FeatureSpec
from traj_analyzer.operators.catalog import Group, load_groups
from traj_analyzer.project import Project

FEATURES = "features"
"""The table joining the values of every extracted group on the trajectory key."""

STATIC = Path(__file__).parent / "static"


class Sorter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    dir: Literal["asc", "desc"]


class Filter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    type: Literal["like"]
    value: str


class RowsRequest(BaseModel):
    """What Tabulator sends for one page, plus the text search and the SQL condition typed above the table."""

    model_config = ConfigDict(extra="forbid")

    table: str
    page: int = Field(ge=1)
    size: int = Field(ge=1, le=1000)
    sorters: list[Sorter] = []
    filters: list[Filter] = []
    search: str = ""
    where: str = ""


@dataclass
class Source:
    name: str
    sql: str
    columns: dict[str, str]
    """Column name to DuckDB type."""
    rows: int


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


class Database:
    """An in-memory DuckDB holding the index and every extracted group, with file access switched off afterwards."""

    def __init__(self, project: Project) -> None:
        self.connection = duckdb.connect()
        index = load_index(project)
        self._load("trajectories", pa.table({KEY: [r.key for r in index], "dataset": [r.dataset for r in index]}))
        self.connection.execute("CREATE SCHEMA groups")
        groups = [g for g in load_groups(project) if (project.table_dir / f"{g.name}.parquet").is_file()]
        for group in groups:
            self._load(f"groups.{_quote(group.name)}", read_group(project, group))
        values = [f"{_quote(g.name)}.{_quote(f.name)}" for g in groups for f in g.features]
        joins = [f"LEFT JOIN groups.{_quote(g.name)} AS {_quote(g.name)} USING ({KEY})" for g in groups]
        self.connection.execute(f"CREATE VIEW {FEATURES} AS SELECT {', '.join(['trajectories.*', *values])} "
                                f"FROM trajectories {' '.join(joins)}")
        self.connection.execute("SET enable_external_access = false")
        self.connection.execute("SET lock_configuration = true")
        self.notes = {KEY: "Trajectory key: <dataset>/<id>.", "dataset": "Dataset the trajectory was ingested into."}
        for group in groups:
            for feature in group.features:
                self.notes.update(_notes(group, feature))
        self.sources = {FEATURES: self._source(FEATURES, FEATURES)}
        for group in groups:
            self.sources[group.name] = self._source(group.name, f"groups.{_quote(group.name)}")

    def _load(self, sql_name: str, table: pa.Table) -> None:
        self.connection.register("incoming", table)
        self.connection.execute(f"CREATE TABLE {sql_name} AS SELECT * FROM incoming")
        self.connection.unregister("incoming")

    def _source(self, name: str, sql: str) -> Source:
        columns = {row[0]: row[1] for row in self.connection.execute(f"DESCRIBE {sql}").fetchall()}
        (rows,) = self.connection.execute(f"SELECT count(*) FROM {sql}").fetchone() or (0,)
        return Source(name=name, sql=sql, columns=columns, rows=rows)

    def rows(self, request: RowsRequest) -> dict[str, Any]:
        source = self.sources.get(request.table)
        if source is None:
            raise HTTPException(404, f"Unknown table {request.table}")
        unknown = sorted({item.field for item in (*request.sorters, *request.filters)} - set(source.columns))
        if unknown:
            raise HTTPException(400, f"Unknown columns {unknown}")
        text = {name: f"lower(CAST({_quote(name)} AS VARCHAR))" for name in source.columns}
        conditions: list[str] = []
        params: list[Any] = []
        for item in request.filters:
            conditions.append(f"contains({text[item.field]}, lower(?))")
            params.append(item.value)
        if request.search:
            conditions.append(f"contains(concat_ws(' ', {', '.join(text.values())}), lower(?))")
            params.append(request.search)
        if request.where.strip():
            conditions.append(f"({request.where})")
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        order = [f"{_quote(s.field)} {s.dir.upper()} NULLS LAST" for s in request.sorters] + [_quote(KEY)]
        cursor = self.connection.cursor()
        try:
            (total,) = cursor.execute(f"SELECT count(*) FROM {source.sql}{where}", params).fetchone() or (0,)
            result = cursor.execute(
                f"SELECT * FROM {source.sql}{where} ORDER BY {', '.join(order)} LIMIT ? OFFSET ?",
                [*params, request.size, (request.page - 1) * request.size])
            names = [column[0] for column in result.description or []]
            data = [dict(zip(names, row, strict=True)) for row in result.fetchall()]
        except duckdb.Error as error:
            raise HTTPException(400, str(error)) from error
        return {"last_page": max(1, math.ceil(total / request.size)), "last_row": total, "data": data}


def _notes(group: Group, feature: FeatureSpec) -> dict[str, str]:
    """Header notes for a feature's value column and the columns that accompany it in its group's table."""
    lines = [feature.description.strip(), "", f"Type: {feature.type}"]
    if feature.range:
        lines.append(f"Range: {feature.range[0]:g} to {feature.range[1]:g}")
    if feature.labels:
        lines += ["Labels:", *(f"  {label}: {text}" for label, text in feature.labels.items())]
    if feature.thresholds:
        lines.append("Thresholds: " + ", ".join(f"{name} = {value}" for name, value in feature.thresholds.items()))
    operators = ", ".join(instance.id for instance in group.instances if feature in instance.features)
    lines.append(f"Group: {group.name} ({group.kind}), operator: {operators}")
    name = feature.name
    return {
        name: "\n".join(lines),
        status_column(name): f"Status of {name}: ok, refused, failed or invalid_length; empty when not extracted.",
        f"{name}__detail": f"Why {name} has no value: not applicable, or the refusal or failure message.",
        f"{name}__evidence": f"Steps of the trajectory the LLM cited for {name}.",
    }


def create_app(project: Project) -> FastAPI:
    database = Database(project)
    app = FastAPI(title="traj viewer")

    @app.middleware("http")
    async def revalidate(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        # Browsers check the page files again on every load, so an upgraded package is picked up at once.
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/api/project")
    def project_info() -> dict[str, Any]:
        return {"name": project.root.name, "root": str(project.root)}

    @app.get("/api/tables")
    def tables() -> list[dict[str, Any]]:
        return [{"name": s.name, "rows": s.rows,
                 "columns": [{"name": n, "type": t, "note": database.notes[n]} for n, t in s.columns.items()]}
                for s in database.sources.values()]

    @app.post("/api/rows")
    def rows(request: RowsRequest) -> dict[str, Any]:
        return database.rows(request)

    app.mount("/", StaticFiles(directory=STATIC, html=True))
    return app
