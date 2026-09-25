from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from traj_analyzer.operators.base import FeatureSpec, Operator, has_user_messages

DEFAULT_LABELS = {
    "misread_request": "Did something other than what the user asked, as shown by the user correcting it.",
    "content_error": "Produced content the user or a later step shows to be wrong, such as a false claim or an "
                     "inconsistent sentence.",
    "repeated_failure": "Retried the same failing approach several times.",
    "premature_claim": "Claimed success or completion without having verified it.",
    "overreach": "Changed or added things the user did not ask for.",
    "stalled": "Stopped or handed work back to the user when it could have continued.",
}


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid")

    labels: dict[str, str] = DEFAULT_LABELS


def outputs(params: Params) -> list[FeatureSpec]:
    return [FeatureSpec(
        name="failure_modes", type="set", labels=params.labels,
        description="Every problem in the assistant's behaviour that the trajectory shows at least once. "
                    "A user correction is evidence of a problem even when the assistant fixes it afterwards.",
    )]


OPERATOR = Operator(
    kind="llm",
    description="Which kinds of mistakes the assistant made.",
    tags=("chat", "agent"),
    params=Params,
    outputs=outputs,
    requires=has_user_messages,
)
