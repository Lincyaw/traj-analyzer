from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import duckdb
import pyarrow as pa
from fastapi import HTTPException, Response
from pydantic import Field

from traj_analyzer.viewer.query import QueryRequest
from traj_analyzer.viewer.query import quote as _quote

if TYPE_CHECKING:
    from traj_analyzer.viewer.server import Database

MAX_ROWS = 250_000
MAX_BYTES = 128 * 1024 * 1024


class AnalysisRequest(QueryRequest):
    feature: str | None = None
    columns: list[str] | None = Field(default=None, min_length=1)


class ProfileRequest(AnalysisRequest):
    column: str
    bins: int = Field(default=40, ge=2, le=200)


def nested(dtype: str) -> bool:
    return dtype.endswith("[]") or dtype.startswith("MAP")


def analysis_query(database: Database, request: AnalysisRequest) -> tuple[str, list[Any], dict[str, str]]:
    source, sql, params = database.query(request)
    columns = request.columns
    if columns is None:
        columns = list(source.columns) if request.feature is None else [
            name for name, dtype in source.columns.items()
            if dtype == "VARCHAR" and not name.endswith(("__status", "__detail", "__evidence"))
        ]
    unknown = sorted(set(columns) - set(source.columns))
    if unknown:
        raise HTTPException(400, f"Unknown columns {unknown}")
    if len(columns) != len(set(columns)):
        raise HTTPException(400, "Duplicate analysis columns")
    types = {name: source.columns[name] for name in columns}
    select = [f"CAST(to_json({_quote(name)}) AS VARCHAR) AS {_quote(name)}" if nested(dtype) else _quote(name)
              for name, dtype in types.items()]
    types = {name: "VARCHAR" if nested(dtype) else dtype for name, dtype in types.items()}
    join = ""
    if request.feature is not None:
        dtype = source.columns.get(request.feature, "")
        if not nested(dtype):
            raise HTTPException(400, "Labels require a set, vector or map feature")
        feature = _quote(request.feature)
        if dtype.startswith("MAP"):
            join = f" CROSS JOIN UNNEST(map_entries({feature})) AS expanded(item)"
            select += ["item.key AS __label", "item.value AS __value"]
            types.update(__label="VARCHAR", __value="DOUBLE")
        elif dtype == "VARCHAR[][]":
            # A paths feature: one label record per event, its levels joined as in the wide table.
            join = f" CROSS JOIN UNNEST({feature}) AS expanded(item)"
            select += ["array_to_string(item, ' | ') AS __label", "1.0 AS __value"]
            types.update(__label="VARCHAR", __value="DOUBLE")
        elif dtype == "VARCHAR[]":
            join = f" CROSS JOIN UNNEST({feature}) AS expanded(item)"
            select += ["item AS __label", "1.0 AS __value"]
            types.update(__label="VARCHAR", __value="DOUBLE")
        else:
            join = f" CROSS JOIN UNNEST({feature}) WITH ORDINALITY AS expanded(item, position)"
            select += ["CAST(position - 1 AS VARCHAR) AS __label", "item AS __value"]
            types.update(__label="VARCHAR", __value="DOUBLE")
    select.append("1.0 AS __count")
    types["__count"] = "DOUBLE"
    return f"SELECT {', '.join(select)} FROM ({sql}) AS data{join}", params, types


def arrow_response(database: Database, request: AnalysisRequest) -> Response:
    sql, params, _ = analysis_query(database, request)
    with database.connection.cursor() as cursor:
        try:
            (count,) = cursor.execute(f"SELECT count(*) FROM ({sql})", params).fetchone()
            if count > MAX_ROWS:
                raise HTTPException(413, f"Analysis has {count:,} rows; add filters to stay below {MAX_ROWS:,} rows")
            table = cursor.execute(sql, params).to_arrow_table()
        except duckdb.Error as error:
            raise HTTPException(400, str(error)) from error
    if table.nbytes > MAX_BYTES:
        raise HTTPException(413, "Analysis exceeds 128 MiB; select fewer columns or add filters")
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return Response(sink.getvalue().to_pybytes(), media_type="application/vnd.apache.arrow.stream",
                    headers={"X-Analysis-Rows": str(count)})


def profile(database: Database, request: ProfileRequest) -> dict[str, Any]:
    sql, params, types = analysis_query(database, request)
    if request.column not in types:
        raise HTTPException(400, f"Unknown analysis column {request.column}")
    column = _quote(request.column)
    numeric = types[request.column] in {"DOUBLE", "FLOAT", "BIGINT", "INTEGER"}
    fields = ["count(*) AS total", f"count({column}) AS valid", f"count(DISTINCT {column}) AS distinct"]
    if numeric:
        fields += [f"count(*) FILTER (WHERE {column} = 0) AS zeros", f"min({column}) AS min",
                   f"max({column}) AS max", f"avg({column}) AS mean", f"quantile_cont({column}, 0.5) AS median",
                   f"quantile_cont({column}, 0.9) AS p90"]
    with database.connection.cursor() as cursor:
        try:
            result = cursor.execute(f"SELECT {', '.join(fields)} FROM ({sql})", params)
            stats = dict(zip([c[0] for c in result.description], result.fetchone(), strict=True))
            if numeric and stats["valid"]:
                low, high = stats["min"], stats["max"]
                if not math.isfinite(low) or not math.isfinite(high):
                    raise HTTPException(400, "Distribution requires finite numeric values")
                width = (high - low) / request.bins if high > low else 1.0
                if not math.isfinite(width) or width <= 0:
                    raise HTTPException(400, "Numeric range cannot be represented with these bins")
                stats.update(bin_start=low, bin_width=width, bins=request.bins)
            stats.update(nulls=stats["total"] - stats["valid"], column=request.column, numeric=numeric)
        except duckdb.Error as error:
            raise HTTPException(400, str(error)) from error
    return stats
