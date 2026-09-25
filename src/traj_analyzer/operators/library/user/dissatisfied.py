from __future__ import annotations

from traj_analyzer.operators.base import FeatureSpec, NoParams, Operator, has_user_messages


def outputs(params: NoParams) -> list[FeatureSpec]:
    return [FeatureSpec(
        name="user_dissatisfied", type="boolean",
        description="A user message says the user is unhappy with the assistant or its work, e.g. 'this is wrong "
                    "again', 'you are not listening', 'still not working'.",
    )]


OPERATOR = Operator(
    kind="llm",
    description="Whether the user writes that they are unhappy with the assistant.",
    tags=("chat", "agent"),
    outputs=outputs,
    requires=has_user_messages,
)
