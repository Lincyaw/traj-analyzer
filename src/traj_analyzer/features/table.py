from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel

from traj_analyzer.files import iter_jsonl, write_jsonl
from traj_analyzer.project import Project

Status = Literal["ok", "refused", "failed", "invalid_length"]


class FeatureRow(BaseModel):
    key: str
    group: str
    feature: str
    value: Any = None
    evidence: str | None = None
    status: Status = "ok"
    detail: str | None = None
    """Why a value is missing: the operator does not apply, or the call was refused or failed."""


def write_group(project: Project, group: str, rows: Iterable[FeatureRow]) -> int:
    """Replace this group's rows for the keys present in `rows` and keep rows for other keys."""
    new = list(rows)
    keys = {row.key for row in new}
    kept = [row for row in read_group(project, group) if row.key not in keys]
    write_jsonl(project.table_dir / f"{group}.jsonl", (row.model_dump_json() for row in (*kept, *new)))
    return len(new)


def read_group(project: Project, group: str) -> list[FeatureRow]:
    path = project.table_dir / f"{group}.jsonl"
    if not path.is_file():
        return []
    return [FeatureRow.model_validate(value) for value in iter_jsonl(path)]
