from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from traj_analyzer.operators.base import Aggregate, FeatureSpec, Operator, has_user_messages

DEFAULT_LABELS = {
    "fix": "Fix a bug or an error.",
    "build": "Build or add something new.",
    "change": "Change or refactor something that exists.",
    "explain": "Explain or analyse something, without changing it.",
    "write": "Write or revise prose, such as documentation or a paper.",
    "operate": "Run, deploy or operate a system.",
    "other": "Any other request.",
    "none": "No request: a reply to the assistant or a confirmation.",
}


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid")

    labels: dict[str, str] = DEFAULT_LABELS


def outputs(params: Params) -> list[FeatureSpec]:
    return [FeatureSpec(
        name="intent", type="category", labels=params.labels,
        description="The kind of request the message the user typed at the start of this turn makes, judged from "
                    "the message alone; `none` when it makes no request.",
    )]


def aggregates(params: Params) -> list[Aggregate]:
    return [
        Aggregate(name="intents", of="intent", kind="set",
                  description="The kinds of request the user's messages make over the trajectory."),
        Aggregate(name="first_intent", of="intent", kind="first",
                  description="The kind of request the first message of the user makes."),
    ]


OPERATOR = Operator(
    kind="judgment",
    description="What kind of request each message the user typed makes, read from the message alone, so the call "
                "group needs no context before the message (`context: 0`); the kinds serve as strata.",
    tags=("chat", "agent"),
    params=Params,
    outputs=outputs,
    aggregates=aggregates,
    unit="turn",
    requires=has_user_messages,
)
