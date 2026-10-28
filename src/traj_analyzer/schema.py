from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["user", "assistant", "tool", "system"]
Kind = Literal["message", "thinking", "tool_call", "tool_result", "notice"]
"""A notice is text the harness wrote into the conversation under the user's role, such as the notice that the
user interrupted the assistant; `Step.name` says which notice."""
INTERRUPTION = "interruption"
"""The name of the notice that the user interrupted the assistant."""
INJECTED = "injected"
"""The name of a notice the harness or the dataset inserted that is no message the user typed."""


class Step(BaseModel):
    index: int
    role: Role
    kind: Kind = "message"
    content: str = ""
    name: str | None = None
    is_error: bool = False
    timestamp: str | None = None
    arguments: str | None = None
    """The JSON text of the arguments of a `tool_call` step, set by an adapter that parses the call; None when the
    content holds the arguments as written."""
    source_step: int | None = None
    """The step of the source this step was parsed from, for a source that numbers its own steps, such as the
    `step_id` of an ATIF trajectory. Several steps of one source step carry the same number, since a source step
    can hold a message, its reasoning, its tool calls and their results. None when the source numbers nothing,
    and the number is the source's own, which says nothing about the order of the steps here."""

    @property
    def is_user_message(self) -> bool:
        return self.role == "user" and self.kind == "message"


class Trajectory(BaseModel):
    id: str
    dataset: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    steps: list[Step] = Field(default_factory=list)
    harness: str | None = None
    """The harness that ran the agent, naming the profile `harness/<name>.py` that reads its tool calls; None when
    the adapter knows no harness."""
    recorded_calls: list[str] | None = None
    """The tools of the calls the source reports apart from the conversation, such as the calls a recorder saw in the
    responses of the model; they can be a part of the agent's calls, where the source reports one form of call and
    not another. `traj ingest` stops at a trajectory whose steps lack one of them. None when the source reports
    none."""

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.id}"
