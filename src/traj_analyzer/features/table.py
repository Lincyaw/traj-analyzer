from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from traj_analyzer.operators.base import FeatureSpec
from traj_analyzer.operators.catalog import Group
from traj_analyzer.project import Project

Status = Literal["ok", "refused", "failed", "invalid_length"]

KEY = "key"

_ARROW = {
    "scalar": pa.float64(),
    "boolean": pa.bool_(),
    "category": pa.string(),
    "set": pa.list_(pa.string()),
    "vector": pa.list_(pa.float64()),
    "map": pa.map_(pa.string(), pa.float64()),
}


@dataclass(frozen=True)
class FeatureRow:
    key: str
    feature: str
    value: Any = None
    evidence: str | None = None
    status: Status = "ok"
    detail: str | None = None
    """Why a value is missing: the operator does not apply, or the call was refused or failed."""


def status_column(feature: str) -> str:
    return f"{feature}__status"


def _fields(feature: FeatureSpec, evidence: bool) -> list[pa.Field]:
    fields = [pa.field(feature.name, _ARROW[feature.type]), pa.field(status_column(feature.name), pa.string()),
              pa.field(f"{feature.name}__detail", pa.string())]
    if evidence:
        fields.append(pa.field(f"{feature.name}__evidence", pa.string()))
    return fields


def schema(group: Group) -> pa.Schema:
    """One row per trajectory; per feature its value and status, the detail, and the evidence when the group cites it.

    A null status means the trajectory has no value for that feature in the table.
    """
    return pa.schema([pa.field(KEY, pa.string()), *(f for feature in group.features
                                                     for f in _fields(feature, group.evidence))])


def write_group(project: Project, group: Group, rows: Iterable[FeatureRow]) -> None:
    """Replace this group's rows for the keys present in `rows` and keep rows for other keys."""
    records: dict[str, dict[str, Any]] = {}
    for row in rows:
        record = records.setdefault(row.key, {KEY: row.key})
        record[row.feature] = row.value
        record[status_column(row.feature)] = row.status
        record[f"{row.feature}__detail"] = row.detail
        record[f"{row.feature}__evidence"] = row.evidence
    table = pa.Table.from_pylist(list(records.values()), schema=schema(group))
    old = read_group(project, group)
    kept = old.filter(pc.invert(pc.is_in(old[KEY], value_set=table[KEY])))
    path = project.table_dir / f"{group.name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.concat_tables([kept, table]), path)


def read_group(project: Project, group: Group) -> pa.Table:
    """The group's table in the schema of its current features.

    A feature the file lacks, or holds with another value type, reads as null for every trajectory.
    """
    path = project.table_dir / f"{group.name}.parquet"
    target = schema(group)
    if not path.is_file():
        return target.empty_table()
    table = pq.read_table(path)
    columns = [table[KEY]]
    for feature in group.features:
        stored = table.schema.field(feature.name).type if feature.name in table.column_names else None
        for field in _fields(feature, group.evidence):
            if stored == _ARROW[feature.type] and field.name in table.column_names:
                columns.append(table[field.name])
            else:
                columns.append(pa.nulls(len(table), field.type))
    return pa.Table.from_arrays(columns, schema=target)


def feature_values(table: pa.Table, feature: FeatureSpec) -> list[Any]:
    """A feature's values as Python objects, with maps as dicts."""
    column = table[feature.name].to_pylist()
    if feature.type == "map":
        return [None if v is None else dict(v) for v in column]
    return column
