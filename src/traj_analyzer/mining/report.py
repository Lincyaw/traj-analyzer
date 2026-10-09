from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Verdict(BaseModel):
    feature: str
    reads: Literal["fields", "text"]
    hypothesis: str | None = None
    parent: str | None = None
    """The feature this one revises."""
    noise: float | None = None
    compared: int = 0
    distinct: int = 0
    varying_share: float | None = None
    batch_explained: float | None = None
    """Share of the variance the batches explain inside the strata, when batches are declared."""
    tail_precision: float | None = None
    """Share of the tail the tail check found worth a look."""
    lift: float | None = None
    """For an event feature, how much more often a consequence follows its events than any tool call."""
    alert: bool = False
    decision: Literal["add", "reject", "remove"]
    reason: str = ""


class RoundReport(BaseModel):
    round: int
    starting: list[str] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list)
    findings: dict[str, int] = Field(default_factory=dict)
    """Number of findings by status."""
    candidates: int = 0
    reviews: dict[str, int] = Field(default_factory=dict)
    """Number of candidates the review kept and rejected."""
    verdicts: list[Verdict] = Field(default_factory=list)
    added: list[str] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)
    extracted: bool = True
    """Whether the enabled mined judgment features were extracted from every trajectory."""
