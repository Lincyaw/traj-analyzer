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

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.id}"
