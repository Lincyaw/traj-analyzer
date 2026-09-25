from __future__ import annotations

from traj_analyzer.operators.base import FeatureSpec, NoParams, Operator


def outputs(params: NoParams) -> list[FeatureSpec]:
    return [FeatureSpec(
        name="assistant_claims_done", type="boolean",
        description="An assistant message says the task is finished, fixed or working, e.g. 'done', 'this fixes "
                    "the bug', 'all tests pass'.",
    )]


OPERATOR = Operator(
    kind="llm",
    description="Whether the assistant says the task is done.",
    tags=("chat", "agent"),
    outputs=outputs,
)
