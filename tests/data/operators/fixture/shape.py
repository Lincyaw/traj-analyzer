from __future__ import annotations

from typing import Any

from traj_analyzer.operators.base import FeatureSpec, NoParams, Operator
from traj_analyzer.schema import Trajectory


def outputs(params: NoParams) -> list[FeatureSpec]:
    return [
        FeatureSpec(name="source", type="category", description="Dataset the trajectory comes from."),
        FeatureSpec(name="turn_band", type="category", labels={"one": "One user turn", "few": "Two or three",
                                                               "many": "Four or more"},
                    description="Number of user turns, in bands."),
    ]


def compute(trajectory: Trajectory, params: NoParams) -> dict[str, Any]:
    turns = sum(1 for s in trajectory.steps if s.is_user_message)
    return {"source": trajectory.dataset, "turn_band": "one" if turns <= 1 else "few" if turns <= 3 else "many"}


OPERATOR = Operator(
    kind="code",
    description="Where a trajectory comes from and how many user turns it has, for grouping in studies.",
    outputs=outputs,
    compute=compute,
)
