from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Sorter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    dir: Literal["asc", "desc"]


class Filter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    type: Literal["like"]
    value: str


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table: str
    filters: list[Filter] = []
    search: str = ""
    where: str = ""


class RowsRequest(QueryRequest):
    page: int = Field(ge=1)
    size: int = Field(ge=1, le=1000)
    sorters: list[Sorter] = []


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'
