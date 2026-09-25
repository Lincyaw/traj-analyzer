from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from traj_analyzer.schema import Trajectory

FeatureType = Literal["scalar", "boolean", "category", "set", "vector", "distribution"]
_NAME = re.compile(r"[a-z][a-z0-9_]{0,47}")


class FeatureSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: FeatureType
    description: str
    range: tuple[float, float] | None = None
    labels: dict[str, str] | None = None
    per: Literal["chunk"] | None = None
    length: int | None = Field(default=None, ge=1)
    thresholds: dict[str, float] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        if not _NAME.fullmatch(value) or "__" in value:
            raise ValueError("feature name must be snake_case without '__'")
        return value

    @model_validator(mode="after")
    def _shape(self) -> FeatureSpec:
        if self.type in ("category", "distribution") and not self.labels:
            raise ValueError(f"{self.name}: type {self.type} needs labels")
        if self.type == "vector" and (self.per is None) == (self.length is None):
            raise ValueError(f"{self.name}: vector needs exactly one of per and length")
        if self.type != "vector" and (self.per or self.length):
            raise ValueError(f"{self.name}: per and length apply only to vectors")
        if self.range and self.range[0] > self.range[1]:
            raise ValueError(f"{self.name}: range is reversed")
        return self


class NoParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class Operator:
    """One reusable feature extractor; every operator file defines one as `OPERATOR`.

    A code operator computes its outputs with `compute`, which returns a value for every output name.
    An LLM operator contributes its outputs and `guidance` to the instruction of its call group.
    `requires` marks the trajectories the operator applies to; the others get empty values.
    """

    kind: Literal["code", "llm"]
    description: str
    outputs: Callable[[Any], list[FeatureSpec]]
    params: type[BaseModel] = NoParams
    tags: tuple[str, ...] = ()
    compute: Callable[[Trajectory, Any], dict[str, Any]] | None = None
    guidance: Callable[[Any], str] | None = None
    requires: Callable[[Trajectory], bool] | None = None

    def __post_init__(self) -> None:
        if (self.kind == "code") != (self.compute is not None):
            raise ValueError("compute is required for code operators and not allowed for llm operators")
        if self.kind == "code" and self.guidance is not None:
            raise ValueError("guidance applies only to llm operators")


def has_user_messages(trajectory: Trajectory) -> bool:
    return any(s.role == "user" and s.kind == "message" for s in trajectory.steps)


def has_tool_calls(trajectory: Trajectory) -> bool:
    return any(s.kind == "tool_call" for s in trajectory.steps)
