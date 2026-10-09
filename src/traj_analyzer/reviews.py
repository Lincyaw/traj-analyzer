from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from traj_analyzer.files import iter_jsonl
from traj_analyzer.project import Project


class Verdict(BaseModel):
    """What a person found when they opened a trajectory a feature pointed them to."""

    model_config = ConfigDict(extra="forbid")

    key: str
    feature: str
    step: int | None = None
    worth: bool
    """Whether the trajectory was worth their look because of what the feature found there."""
    note: str = ""
    at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))


def append_verdict(project: Project, verdict: Verdict) -> None:
    """Add one verdict to `reviews/verdicts.jsonl`, which lives in git like the reports."""
    path = project.reviews_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as out:
        out.write(verdict.model_dump_json() + "\n")


def read_verdicts(project: Project) -> list[Verdict]:
    path = project.reviews_path
    if not path.is_file():
        return []
    return [Verdict.model_validate(value) for value in iter_jsonl(path)]


def precision_by_feature(verdicts: list[Verdict]) -> dict[str, tuple[float, int]]:
    """Per feature, the share of its verdicts that found the trajectory worth a look, and how many there are.

    A later verdict on the same trajectory and feature replaces an earlier one.
    """
    latest: dict[tuple[str, str], bool] = {}
    for verdict in verdicts:
        latest[verdict.feature, verdict.key] = verdict.worth
    counts: dict[str, list[bool]] = {}
    for (feature, _), worth in latest.items():
        counts.setdefault(feature, []).append(worth)
    return {feature: (sum(found) / len(found), len(found)) for feature, found in counts.items()}
