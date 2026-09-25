from __future__ import annotations

from datetime import datetime
from typing import Any

from traj_analyzer.operators.base import FeatureSpec, NoParams, Operator
from traj_analyzer.schema import Trajectory


def outputs(params: NoParams) -> list[FeatureSpec]:
    return [
        FeatureSpec(name="n_steps", type="scalar", description="Number of steps."),
        FeatureSpec(name="n_user_turns", type="scalar", description="Number of messages typed by the user."),
        FeatureSpec(name="total_chars", type="scalar", description="Characters of content over all steps."),
        FeatureSpec(name="duration_minutes", type="scalar",
                    description="Minutes from the first to the last timestamped step."),
    ]


def compute(trajectory: Trajectory, params: NoParams) -> dict[str, Any]:
    stamps = [datetime.fromisoformat(s.timestamp) for s in trajectory.steps if s.timestamp]
    return {
        "n_steps": float(len(trajectory.steps)),
        "n_user_turns": float(sum(1 for s in trajectory.steps if s.is_user_message)),
        "total_chars": float(sum(len(s.content) for s in trajectory.steps)),
        "duration_minutes": (stamps[-1] - stamps[0]).total_seconds() / 60 if len(stamps) > 1 else None,
    }


OPERATOR = Operator(
    kind="code",
    description="Size and duration of a trajectory.",
    tags=("chat", "agent"),
    outputs=outputs,
    compute=compute,
)
