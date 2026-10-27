from __future__ import annotations

from typing import Any

from traj_analyzer.layers.tables import Tables
from traj_analyzer.operators.base import EventMark, FeatureSpec, NoParams, Operator
from traj_analyzer.schema import Trajectory


def outputs(params: NoParams) -> list[FeatureSpec]:
    return [FeatureSpec(name="weather_calls", type="event", event=EventMark(edge="tool"),
                        description="A tool call whose tool name holds `weather`.")]


def compute(trajectory: Trajectory, tables: Tables, params: NoParams) -> dict[str, Any]:
    return {"weather_calls": [{"step": step.index, "objects": [], "recovered": False, "recovered_at": None,
                               "note": step.name} for step in trajectory.steps
                              if step.kind == "tool_call" and "weather" in (step.name or "")]}


OPERATOR = Operator(kind="code", description="Tool calls of the weather tools.", outputs=outputs, compute=compute)
