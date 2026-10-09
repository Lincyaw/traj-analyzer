from __future__ import annotations

from typing import Any

from traj_analyzer.events import RULES, events_of
from traj_analyzer.layers.tables import Tables
from traj_analyzer.operators.base import FeatureSpec, NoParams, Operator, has_tool_calls
from traj_analyzer.schema import Trajectory


def outputs(params: NoParams) -> list[FeatureSpec]:
    return [rule.spec() for rule in RULES]


def compute(trajectory: Trajectory, tables: Tables, params: NoParams) -> dict[str, Any]:
    return events_of(RULES, tables)


OPERATOR = Operator(
    kind="code",
    description="Expectations about tool use that hold in every scenario: a call gets a result, a failed action is "
                "not repeated unchanged, a side effect follows a word from the user, failures do not run on, and a "
                "turn the user interrupts had produced something and was not stopped at its plan.",
    tags=("agent",),
    outputs=outputs,
    compute=compute,
    requires=has_tool_calls,
    tables=("actions", "relations", "turns"),
)
