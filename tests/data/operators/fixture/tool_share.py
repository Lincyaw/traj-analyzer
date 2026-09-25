from __future__ import annotations

from typing import Any

from traj_analyzer.operators.base import FeatureSpec, NoParams, Operator, has_tool_calls
from traj_analyzer.schema import Trajectory


def outputs(params: NoParams) -> list[FeatureSpec]:
    return [FeatureSpec(name="tool_share", type="scalar", range=(0, 1), description="Share of steps that call tools.")]


def compute(trajectory: Trajectory, params: NoParams) -> dict[str, Any]:
    calls = sum(1 for s in trajectory.steps if s.kind == "tool_call")
    return {"tool_share": calls / len(trajectory.steps)}


OPERATOR = Operator(
    kind="code",
    description="Share of steps that call tools, for trajectories with tool calls.",
    outputs=outputs,
    compute=compute,
    requires=has_tool_calls,
)
