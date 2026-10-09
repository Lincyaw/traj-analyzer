from __future__ import annotations

from traj_analyzer.operators.base import FeatureSpec, NoParams, Operator, has_user_messages

LABELS = {
    "done": "The last task the user gave is completed as far as the trajectory shows, and the user raised no "
            "objection after.",
    "partial": "Part of the last task is done and part is left open, or the assistant stopped with work "
               "outstanding.",
    "abandoned": "The last task is left open: the user stopped answering, gave up, or the assistant stopped "
                 "without finishing.",
    "unclear": "The trajectory does not show whether the last task was completed.",
}


def outputs(params: NoParams) -> list[FeatureSpec]:
    return [
        FeatureSpec(name="end_state", type="category", labels=LABELS,
                    description="The state of the last task the user gave when the trajectory ends: done, partial, "
                                "abandoned or unclear."),
        FeatureSpec(name="agent_claims_done", type="boolean",
                    description="The assistant's last message says the task is finished, fixed or working."),
    ]


OPERATOR = Operator(
    kind="judgment",
    description="How the trajectory ends: whether the last task was completed, and whether the assistant says so.",
    tags=("chat", "agent"),
    outputs=outputs,
    requires=has_user_messages,
)
