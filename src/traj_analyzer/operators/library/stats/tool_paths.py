from __future__ import annotations

from typing import Any

from traj_analyzer.operators.base import FeatureSpec, NoParams, Operator, has_tool_calls
from traj_analyzer.schema import Step, Trajectory


def outputs(params: NoParams) -> list[FeatureSpec]:
    return [FeatureSpec(
        name="tool_paths", type="paths", levels=["tool", "outcome"],
        description="One path per tool call in order: the tool name, then how the call ended, as `ok`, `error`, "
                    "or `unanswered` when the trajectory holds no result for it. A result answers the earliest call "
                    "without one, of the same tool when the result names its tool. Calls whose data names no tool "
                    "count as `unnamed`.",
    )]


def compute(trajectory: Trajectory, params: NoParams) -> dict[str, Any]:
    calls: list[tuple[Step, list[str]]] = []
    for step in trajectory.steps:
        if step.kind == "tool_call":
            calls.append((step, [step.name or "unnamed", "unanswered"]))
        elif step.kind == "tool_result":
            # A result answers the earliest call without one, of the same tool when the result names its tool.
            for call, path in calls:
                if step.name in (None, call.name) and path[1] == "unanswered":
                    path[1] = "error" if step.is_error else "ok"
                    break
    return {"tool_paths": [path for _, path in calls]}


OPERATOR = Operator(
    kind="code",
    description="Every tool call as a path of tool name and outcome, for the tree of what the agent calls.",
    tags=("agent",),
    outputs=outputs,
    compute=compute,
    requires=has_tool_calls,
)
