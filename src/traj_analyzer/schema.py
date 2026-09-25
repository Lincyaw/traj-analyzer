from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["user", "assistant", "tool", "system"]
Kind = Literal["message", "thinking", "tool_call", "tool_result"]


class Step(BaseModel):
    index: int
    role: Role
    kind: Kind = "message"
    content: str = ""
    name: str | None = None
    is_error: bool = False
    timestamp: str | None = None


class Trajectory(BaseModel):
    id: str
    dataset: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    steps: list[Step] = Field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.id}"
