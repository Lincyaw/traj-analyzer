from __future__ import annotations

from traj_analyzer.operators.base import FeatureSpec, NoParams, Operator, has_user_messages


def outputs(params: NoParams) -> list[FeatureSpec]:
    return [FeatureSpec(
        name="user_approves", type="boolean",
        description="A user message accepts the assistant's result, e.g. 'works now', 'looks good', 'thanks, done', "
                    "or asks it to commit or ship the result.",
    )]


OPERATOR = Operator(
    kind="llm",
    description="Whether the user accepts the assistant's result.",
    tags=("chat", "agent"),
    outputs=outputs,
    requires=has_user_messages,
)
