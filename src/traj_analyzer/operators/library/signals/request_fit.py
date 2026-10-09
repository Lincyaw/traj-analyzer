from __future__ import annotations

from traj_analyzer.operators.base import Aggregate, EventMark, FeatureSpec, NoParams, Operator, has_user_messages

FIT = {
    "yes": "Everything the assistant does in the window serves the request.",
    "partly": "Some of what the assistant does serves the request and some does not.",
    "no": "What the assistant does in the window does not serve the request: it works on something else, "
          "undoes what was asked, or acts against an instruction of the request.",
}
EDGES = {
    "none": "Nothing went wrong.",
    "owner": "The assistant misread or ignored what the user asked.",
    "context": "The assistant acted on context or instructions that contradict the request.",
    "memory": "The assistant lost or confused what it had learned earlier in the trajectory.",
    "tool": "A tool result was misread, or a tool was used for something it does not do.",
    "peer": "A subagent or peer agent did something else than asked.",
    "local_env": "The local environment behaved otherwise than the assistant assumed.",
    "external_env": "An external service behaved otherwise than the assistant assumed.",
}


def outputs(params: NoParams) -> list[FeatureSpec]:
    return [
        FeatureSpec(name="request_fit", type="category", labels=FIT,
                    description="Whether what the assistant does in the window serves the request in the section "
                                "`## Request`: yes, partly or no. Judge the work against the request, not against "
                                "whether it succeeded.",
                    event=EventMark(edge="owner", when=["no"])),
        FeatureSpec(name="unfit_edge", type="category", labels=EDGES,
                    description="When request_fit is partly or no: the interaction on which it went wrong; none "
                                "otherwise."),
    ]


def aggregates(params: NoParams) -> list[Aggregate]:
    return [
        Aggregate(name="unfit_share", of="request_fit", kind="share", values=["no"],
                  description="Share of the windows in which the assistant's work does not serve the request."),
        Aggregate(name="partly_fit_share", of="request_fit", kind="share", values=["partly"],
                  description="Share of the windows in which only part of the assistant's work serves the "
                              "request."),
    ]


OPERATOR = Operator(
    kind="judgment",
    description="Whether the assistant's work in each window of the trajectory serves what the user asked, and "
                "on which interaction it goes wrong when it does not.",
    tags=("agent",),
    outputs=outputs,
    aggregates=aggregates,
    unit="window",
    requires=has_user_messages,
)
