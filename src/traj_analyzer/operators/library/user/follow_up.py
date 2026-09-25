from __future__ import annotations

from traj_analyzer.operators.base import FeatureSpec, NoParams, Operator, has_user_messages


def outputs(params: NoParams) -> list[FeatureSpec]:
    return [FeatureSpec(
        name="user_follow_up", type="boolean",
        description="After an assistant answer, a user message asks a further question or makes a further request "
                    "on the same task.",
    )]


OPERATOR = Operator(
    kind="llm",
    description="Whether the user follows up on an assistant answer.",
    tags=("chat", "agent"),
    outputs=outputs,
    requires=has_user_messages,
)
