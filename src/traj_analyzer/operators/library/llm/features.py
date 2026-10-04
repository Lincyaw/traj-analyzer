from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from traj_analyzer.operators.base import FeatureSpec, Operator


class LlmFeature(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["boolean", "category", "set"]
    description: str
    labels: dict[str, str] | None = None
    guidance: str = ""
    """Judging criteria for this feature, written into the instruction."""


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: dict[str, LlmFeature] = Field(default_factory=dict)
    """Feature name to its definition."""


def outputs(params: Params) -> list[FeatureSpec]:
    return [FeatureSpec(name=name, type=spec.type, description=spec.description, labels=spec.labels)
            for name, spec in params.features.items()]


def guidance(params: Params) -> str:
    return "\n".join(f"- `{name}`: {spec.guidance.strip()}" for name, spec in params.features.items() if spec.guidance)


OPERATOR = Operator(
    kind="llm",
    description="Features defined in traj.yaml: each one is a literal question about what the trajectory shows.",
    tags=("chat", "agent"),
    params=Params,
    outputs=outputs,
    guidance=guidance,
)
