from __future__ import annotations

from collections import Counter
from typing import Any

from traj_analyzer.operators.base import FeatureSpec, NoParams, Operator
from traj_analyzer.schema import Trajectory


def outputs(params: NoParams) -> list[FeatureSpec]:
    return [
        FeatureSpec(name="n_tool_calls", type="scalar", description="Number of tool calls."),
        FeatureSpec(name="tool_error_rate", type="scalar", range=(0, 1),
                    description="Share of tool results flagged as errors; empty without tool results."),
        FeatureSpec(name="tool_calls", type="map",
                    description="Number of calls per tool name; steps whose data names no tool count as `unnamed`."),
        FeatureSpec(name="tool_errors", type="map",
                    description="Number of error results per tool name; results whose data names no tool count as "
                                "`unnamed`."),
    ]


def compute(trajectory: Trajectory, params: NoParams) -> dict[str, Any]:
    calls = [s for s in trajectory.steps if s.kind == "tool_call"]
    results = [s for s in trajectory.steps if s.kind == "tool_result"]
    return {
        "n_tool_calls": float(len(calls)),
        "tool_error_rate": sum(s.is_error for s in results) / len(results) if results else None,
        "tool_calls": dict(Counter(s.name or "unnamed" for s in calls)),
        "tool_errors": dict(Counter(s.name or "unnamed" for s in results if s.is_error)),
    }


OPERATOR = Operator(
    kind="code",
    description="How often the trajectory calls each tool, and how often each tool fails.",
    tags=("agent",),
    outputs=outputs,
    compute=compute,
)
