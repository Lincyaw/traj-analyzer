from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from traj_analyzer.operators.base import FeatureSpec, Operator, has_user_messages

DEFAULT_LABELS = {
    "fix": "Fix a bug or an error.",
    "build": "Build or add something new.",
    "change": "Change or refactor something that exists.",
    "explain": "Explain or analyse something, without changing it.",
    "write": "Write or revise prose, such as documentation or a paper.",
    "operate": "Run, deploy or operate a system.",
    "other": "Any other request.",
}


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid")

    labels: dict[str, str] = DEFAULT_LABELS


def outputs(params: Params) -> list[FeatureSpec]:
    return [FeatureSpec(
        name="request_kinds", type="set", labels=params.labels,
        description="The kinds of request the user's messages make; give the kind of every request a user message "
                    "states.",
    )]


OPERATOR = Operator(
    kind="llm",
    description="What kinds of request the user makes.",
    tags=("chat", "agent"),
    params=Params,
    outputs=outputs,
    requires=has_user_messages,
)
