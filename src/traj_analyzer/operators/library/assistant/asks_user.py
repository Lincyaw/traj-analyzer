from __future__ import annotations

from traj_analyzer.operators.base import FeatureSpec, NoParams, Operator, has_user_messages


def outputs(params: NoParams) -> list[FeatureSpec]:
    return [FeatureSpec(
        name="assistant_asks_user", type="boolean",
        description="An assistant message asks the user a question: for clarification, for a decision, or for "
                    "permission to go on.",
    )]


OPERATOR = Operator(
    kind="llm",
    description="Whether the assistant asks the user something.",
    tags=("chat", "agent"),
    outputs=outputs,
    requires=has_user_messages,
)
