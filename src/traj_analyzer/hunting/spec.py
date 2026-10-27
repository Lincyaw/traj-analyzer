from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from traj_analyzer.files import read_yaml
from traj_analyzer.project import ConfigError, Project

NAME = re.compile(r"[a-z][a-z0-9_-]{0,39}")
"""Name of a hunt; it becomes part of the names of its aifn functions."""


class HuntSpec(BaseModel):
    """One hunt, `hunts/<name>.yaml`: what to look for in the trajectories, and how much to read per round."""

    model_config = ConfigDict(extra="forbid")

    need: str = Field(min_length=1)
    """What the trajectories worth finding show, in the words of the person who asks."""
    datasets: Literal["all"] | list[str] = "all"
    patience: int = Field(default=3, ge=1)
    """Rounds in a row without a new finding that end the search."""
    max_rounds: int = Field(default=12, ge=1)
    """Most rounds, which bounds the model calls."""
    reads_per_round: int = Field(default=60, ge=2)
    """Trajectories `hunt-verify` reads per round."""
    hit_share: float = Field(default=0.6, ge=0, le=1)
    """Share of the reads of a round drawn from the hits of the probes; the rest explores the trajectories no probe
    hits."""
    min_reads_per_probe: int = Field(default=3, ge=1)
    """Reads every active probe with unread hits gets before the rest of the hit reads is shared out."""
    cases_per_call: int = Field(default=6, ge=1)
    """Trajectories one `hunt-verify` call reads."""
    clusters: int = Field(default=40, ge=1)
    """Clusters of the trajectories that the coverage reads go through, the least read first."""
    probes_max: int = Field(default=20, ge=1)
    """Most active probes."""
    min_precision: float = Field(default=0.5, ge=0, le=1)
    """Precision below which the ledger asks for a probe to be revised."""
    final_reads: int = Field(default=120, ge=0)
    """Reads of the closing check, over the hits of the active probes no round read."""
    seed: int = 0
    model: str | None = None
    """Model of the hunt's functions; defaults to mining.model, then engine.model."""

    @model_validator(mode="after")
    def _rounds(self) -> HuntSpec:
        if self.patience > self.max_rounds:
            raise ValueError("patience is larger than max_rounds, so the search could never end on it")
        return self


def hunt_names(project: Project) -> list[str]:
    """The hunts of the project."""
    if not project.hunts_dir.is_dir():
        return []
    return sorted(path.stem for path in project.hunts_dir.glob("*.yaml"))


def load_hunt(project: Project, name: str) -> HuntSpec:
    if not NAME.fullmatch(name):
        raise ConfigError(f"Hunt name {name!r} must match {NAME.pattern}")
    path = project.hunts_dir / f"{name}.yaml"
    if not path.is_file():
        raise ConfigError(f"No hunt {name}: {path} does not exist")
    spec = HuntSpec.model_validate(read_yaml(path))
    if spec.datasets != "all":
        unknown = sorted(set(spec.datasets) - set(project.config.datasets))
        if unknown:
            raise ConfigError(f"Hunt {name} names datasets traj.yaml does not configure: {unknown}")
    return spec


def hunt_model(project: Project, spec: HuntSpec) -> str:
    return spec.model or project.config.mining.model or project.config.engine.model
