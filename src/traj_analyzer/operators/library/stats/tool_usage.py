from __future__ import annotations

from typing import Any

from traj_analyzer.operators.base import FeatureSpec, NoParams, Operator
from traj_analyzer.schema import Trajectory


def outputs(params: NoParams) -> list[FeatureSpec]:
    return [
        FeatureSpec(name="n_tool_calls", type="scalar", description="Number of tool calls."),
        FeatureSpec(name="tool_error_rate", type="scalar", range=(0, 1),
                    description="Share of tool results flagged as errors; empty without tool results."),
        FeatureSpec(name="tools_used", type="set", description="Distinct names of the tools called."),
    ]


def compute(trajectory: Trajectory, params: NoParams) -> dict[str, Any]:
    results = [s for s in trajectory.steps if s.kind == "tool_result"]
    calls = [s for s in trajectory.steps if s.kind == "tool_call"]
    return {
        "n_tool_calls": float(len(calls)),
        "tool_error_rate": sum(s.is_error for s in results) / len(results) if results else None,
        "tools_used": sorted({s.name for s in calls if s.name}),
    }


OPERATOR = Operator(
    kind="code",
    description="How much the trajectory calls tools, which tools, and how often they fail.",
    tags=("agent",),
    outputs=outputs,
    compute=compute,
)
