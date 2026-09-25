from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from traj_analyzer.operators.base import FeatureSpec, Operator, has_user_messages

DEFAULT_LABELS = {
    "bugfix": "Fixing a defect in existing code or configuration.",
    "feature": "Building new functionality.",
    "analysis": "Understanding, reviewing or explaining existing material, without producing a deliverable.",
    "writing": "Producing or revising prose, such as documentation or papers.",
    "ops": "Running, deploying or operating systems.",
    "other": "None of the above.",
}


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid")

    labels: dict[str, str] = DEFAULT_LABELS


def outputs(params: Params) -> list[FeatureSpec]:
    return [FeatureSpec(
        name="task_type", type="category", labels=params.labels,
        description="The main kind of work the user asked for; when the request changes, the one that took most "
                    "of the trajectory.",
    )]


OPERATOR = Operator(
    kind="llm",
    description="The main kind of work the user asked for.",
    tags=("chat", "agent"),
    params=Params,
    outputs=outputs,
    requires=has_user_messages,
)
