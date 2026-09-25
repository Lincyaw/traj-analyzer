from __future__ import annotations

from traj_analyzer.operators.base import FeatureSpec, NoParams, Operator, has_user_messages


def outputs(params: NoParams) -> list[FeatureSpec]:
    return [FeatureSpec(
        name="task_completed", type="boolean",
        description="By the last step, the user's final request is fulfilled and the user has not rejected the "
                    "result. False when the trajectory ends with the user pointing out a remaining problem, "
                    "with an open question from the user, or with work the assistant announced but did not do.",
    )]


OPERATOR = Operator(
    kind="llm",
    description="Whether the user's final request was fulfilled.",
    tags=("chat", "agent"),
    outputs=outputs,
    requires=has_user_messages,
)
