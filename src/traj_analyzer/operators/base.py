from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from traj_analyzer.schema import Trajectory

if TYPE_CHECKING:
    from traj_analyzer.layers.tables import Tables

FeatureType = Literal["boolean", "scalar", "category", "set", "vector", "map", "paths", "event"]
Edge = Literal["owner", "grader", "third_party", "context", "memory", "tool", "peer", "local_env", "external_env"]
"""The interaction an event lies on: with the person who gave the task, a grader, a third party, the context given,
memory, a tool, a peer agent or subagent, the local environment, or an external environment."""
Unit = Literal["trajectory", "turn", "window"]
"""What one call of a judgment operator covers: the whole trajectory, every turn of the user at once, or one window
of steps."""
_NAME = re.compile(r"[a-z][a-z0-9_]{0,47}")


class EventMark(BaseModel):
    """What makes a value an event: the edge it lies on, and for a judged boolean or category output the values that
    count, such as `["true"]` or `["correct", "reject"]`."""

    model_config = ConfigDict(extra="forbid")

    edge: Edge
    when: list[str] = Field(default_factory=list)


class Event(BaseModel):
    """One expectation broken at a position of the trajectory, as a feature of type `event` holds it."""

    model_config = ConfigDict(extra="forbid")

    step: int
    end: int | None = None
    objects: list[str] = Field(default_factory=list)
    recovered: bool = False
    recovered_at: int | None = None
    note: str = ""


class FeatureSpec(BaseModel):
    """One feature: a boolean, a number, a label, a set of labels, a vector of numbers, a map from label to number,
    a list of paths, or a list of events.

    `labels` fixes the allowed labels of a category, the items of a set, or the keys of a map; without it they are
    free strings. `range` bounds a scalar, the elements of a vector, or the values of a map.
    A paths feature holds one path per event of the trajectory, such as a tool call, in order; a path has one
    label per entry of `levels`, each level refining the one before it.
    An event feature holds the positions where a rule found an expectation broken; `event` names the edge.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    type: FeatureType
    description: str
    range: tuple[float, float] | None = None
    labels: dict[str, str] | None = None
    per: Literal["chunk"] | None = None
    length: int | None = Field(default=None, ge=1)
    levels: list[str] | None = None
    """Names of the levels of a paths feature, from the coarsest to the finest."""
    thresholds: dict[str, float] = Field(default_factory=dict)
    scope: str | None = None
    """For a feature a model reads: a condition over extracted code features, in the syntax of a sampler's `where`.
    The model is asked only about trajectories that meet it, and the feature has no value on the others."""
    event: EventMark | None = None
    """For an event feature, its edge; for a judged boolean or category output, the values that make an event."""

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        if not _NAME.fullmatch(value) or "__" in value:
            raise ValueError("feature name must be snake_case without '__'")
        return value

    @model_validator(mode="after")
    def _shape(self) -> FeatureSpec:
        if self.labels and self.type not in ("category", "set", "map"):
            raise ValueError(f"{self.name}: labels apply only to category, set and map")
        if self.range and self.type not in ("scalar", "vector", "map"):
            raise ValueError(f"{self.name}: range applies only to scalar, vector and map")
        if self.type == "vector" and (self.per is None) == (self.length is None):
            raise ValueError(f"{self.name}: vector needs exactly one of per and length")
        if self.type != "vector" and (self.per or self.length):
            raise ValueError(f"{self.name}: per and length apply only to vectors")
        if self.range and self.range[0] > self.range[1]:
            raise ValueError(f"{self.name}: range is reversed")
        if (self.type == "paths") != bool(self.levels):
            raise ValueError(f"{self.name}: levels are required for paths and apply to no other type")
        if self.levels and len(set(self.levels)) != len(self.levels):
            raise ValueError(f"{self.name}: level names must be distinct")
        if self.type == "event" and (self.event is None or self.event.when):
            raise ValueError(f"{self.name}: an event feature names its edge in `event` and takes no `when`")
        if self.event is not None and self.type not in ("event", "boolean", "category"):
            raise ValueError(f"{self.name}: `event` applies to event, boolean and category features")
        if self.event is not None and self.type in ("boolean", "category") and not self.event.when:
            raise ValueError(f"{self.name}: `event.when` lists the values that make an event")
        if self.event is not None and self.type == "category" and self.labels:
            unknown = sorted(set(self.event.when) - set(self.labels))
            if unknown:
                raise ValueError(f"{self.name}: `event.when` names labels the feature lacks: {unknown}")
        return self


@dataclass(frozen=True)
class Evidenced:
    """A value with the steps it rests on, which a code operator returns for a feature that keeps evidence."""

    value: Any
    steps: tuple[int, ...] = ()
    note: str = ""


AggregateKind = Literal["share", "any", "count", "mean", "last", "first", "set"]


class Aggregate(BaseModel):
    """A trajectory feature computed from the per-unit outputs of a judgment operator.

    `share`, `any` and `count` take the units whose output `of` is one of `values`; `mean` takes a scalar output;
    `last` and `first` copy the output of the last or first unit; `set` collects the distinct values of a category
    output over the units.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    of: str
    kind: AggregateKind
    values: list[str] = Field(default_factory=list)
    description: str

    def spec(self, outputs: dict[str, FeatureSpec]) -> FeatureSpec:
        if self.of not in outputs:
            raise ValueError(f"{self.name}: aggregates output {self.of}, which the operator lacks")
        source = outputs[self.of]
        if self.kind in ("share", "any", "count"):
            if source.type not in ("boolean", "category") or not self.values:
                raise ValueError(f"{self.name}: {self.kind} takes a boolean or category output and `values`")
            type_: FeatureType = "scalar" if self.kind != "any" else "boolean"
            range_ = (0.0, 1.0) if self.kind == "share" else None
            return FeatureSpec(name=self.name, type=type_, range=range_, description=self.description)
        if self.kind == "mean":
            if source.type != "scalar":
                raise ValueError(f"{self.name}: mean takes a scalar output")
            return FeatureSpec(name=self.name, type="scalar", range=source.range, description=self.description)
        if self.kind == "set":
            if source.type != "category":
                raise ValueError(f"{self.name}: set takes a category output")
            return FeatureSpec(name=self.name, type="set", labels=source.labels, description=self.description)
        if source.type not in ("boolean", "category", "scalar"):
            raise ValueError(f"{self.name}: {self.kind} takes a boolean, category or scalar output")
        return FeatureSpec(name=self.name, type=source.type, labels=source.labels, range=source.range,
                           description=self.description)


class NoParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class Operator:
    """One reusable feature extractor; every operator file defines one as `OPERATOR`.

    A code operator computes its outputs with `compute(trajectory, tables, params)`, which returns a value for every
    output name; `tables` names the tables of the parsing layers it reads, which are enabled with it, and
    `tables_of(params)` the tables it reads besides those with these params.
    A judgment operator contributes its outputs and `guidance` to the instruction of its call group; `unit` says
    what one call covers, and for `turn` and `window` units `aggregates` turn the per-unit outputs into features.
    `requires` marks the trajectories the operator applies to; the others get empty values.
    """

    kind: Literal["code", "judgment"]
    description: str
    outputs: Callable[[Any], list[FeatureSpec]]
    params: type[BaseModel] = NoParams
    tags: tuple[str, ...] = ()
    compute: Callable[[Trajectory, Tables, Any], dict[str, Any]] | None = None
    guidance: Callable[[Any], str] | None = None
    requires: Callable[[Trajectory], bool] | None = None
    tables: tuple[str, ...] = ()
    tables_of: Callable[[Any], tuple[str, ...]] | None = None
    unit: Unit = "trajectory"
    aggregates: Callable[[Any], list[Aggregate]] | None = None

    def __post_init__(self) -> None:
        if (self.kind == "code") != (self.compute is not None):
            raise ValueError("compute is required for code operators and not allowed for judgment operators")
        if self.kind == "code" and (self.guidance is not None or self.aggregates is not None):
            raise ValueError("guidance and aggregates apply only to judgment operators")
        if self.kind == "code" and self.unit != "trajectory":
            raise ValueError("a code operator computes per trajectory")
        if self.kind == "judgment" and (self.unit != "trajectory") != (self.aggregates is not None):
            raise ValueError("a judgment operator over turns or windows declares aggregates, one per trajectory "
                             "does not")
        if self.kind == "judgment" and (self.tables or self.tables_of is not None):
            raise ValueError("tables apply only to code operators")

    def tables_with(self, params: Any) -> tuple[str, ...]:
        """Every table the operator reads with these params."""
        extra = self.tables_of(params) if self.tables_of is not None else ()
        return tuple(sorted({*self.tables, *extra}))


def has_user_messages(trajectory: Trajectory) -> bool:
    return any(s.is_user_message for s in trajectory.steps)


def has_tool_calls(trajectory: Trajectory) -> bool:
    return any(s.kind == "tool_call" for s in trajectory.steps)
