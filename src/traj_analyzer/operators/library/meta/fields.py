from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from traj_analyzer.operators.base import FeatureSpec, Operator
from traj_analyzer.schema import Trajectory


class MetaField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    type: Literal["scalar", "boolean", "category", "set"]
    description: str = ""
    labels: dict[str, str] | None = None
    range: tuple[float, float] | None = None
    thresholds: dict[str, float] = Field(default_factory=dict)
    required: bool = True
    """A missing key is an error when required, and an empty value otherwise."""


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fields: dict[str, MetaField] = Field(default_factory=dict)
    """Feature name to the metadata field it copies."""


def outputs(params: Params) -> list[FeatureSpec]:
    return [
        FeatureSpec(name=name, type=spec.type, labels=spec.labels, range=spec.range, thresholds=spec.thresholds,
                    description=spec.description or f"Metadata field {spec.key}.")
        for name, spec in params.fields.items()
    ]


def _convert(spec: MetaField, value: Any) -> Any:
    if value is None:
        return None
    match spec.type:
        case "scalar":
            return float(value)
        case "boolean":
            if not isinstance(value, bool):
                raise TypeError(f"metadata {spec.key} must be a boolean, got {value!r}")
            return value
        case "category":
            return str(value)
        case "set":
            return sorted({str(v) for v in value})


def compute(trajectory: Trajectory, params: Params) -> dict[str, Any]:
    values = {}
    for name, spec in params.fields.items():
        if spec.key not in trajectory.metadata and spec.required:
            raise KeyError(f"{trajectory.key} has no metadata {spec.key}")
        values[name] = _convert(spec, trajectory.metadata.get(spec.key))
    return values


OPERATOR = Operator(
    kind="code",
    description="Copies chosen trajectory metadata, such as evaluation results or the model name, into features.",
    tags=("chat", "agent"),
    params=Params,
    outputs=outputs,
    compute=compute,
)
