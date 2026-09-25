from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from traj_analyzer.operators.base import FeatureSpec, Operator, has_user_messages

SCALE = """\
Score how much the user had to push back on the assistant, using this scale.
- 0: every user message is a new request, a follow up question, or approval.
- 0.3: the user corrects the assistant once, such as pointing out an error, a misunderstanding or an unwanted change.
- 0.6: the user corrects the assistant several times, repeats a request that was not followed, or states the same \
problem again.
- 1: the user expresses anger or gives up on the assistant.
A correction counts even when it is phrased politely or as a question, such as "this sentence still has a problem" \
or "why did you change that file".
Messages that only refine the user's own request, without the assistant having done something wrong, do not count.
"""


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid")

    high: float = 0.5


def outputs(params: Params) -> list[FeatureSpec]:
    return [
        FeatureSpec(name="user_frustration", type="scalar", range=(0, 1), thresholds={"high": params.high},
                    description="How much the user had to push back on the assistant over the whole trajectory, "
                                "on the scale in the guidance."),
        FeatureSpec(name="frustration_curve", type="vector", per="chunk", range=(0, 1),
                    description="user_frustration judged separately within each chunk."),
    ]


def guidance(params: Params) -> str:
    return SCALE


OPERATOR = Operator(
    kind="llm",
    description="How much the user had to correct or push back on the assistant, overall and per chunk.",
    tags=("chat", "agent"),
    params=Params,
    outputs=outputs,
    guidance=guidance,
    requires=has_user_messages,
)
