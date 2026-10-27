from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from traj_analyzer.anomalies import Spread
from traj_analyzer.calls import WorkspaceCalls
from traj_analyzer.ingest import IndexRow
from traj_analyzer.mining.functions import Definition, Example, ReviseRequest, StepRequest, TailRequest, mining_model
from traj_analyzer.mining.report import Verdict

WORKSPACE = "workspace"
"""Directory of a round that its functions share."""


@dataclass(kw_only=True)
class Session(WorkspaceCalls):
    """What every step of one round shares."""

    number: int
    directory: Path
    rows: dict[str, IndexRow]
    reference: list[str]
    chosen: list[str]
    """The evaluation set."""
    readable: list[str]
    """The reference trajectories the functions may read: those no measurement of noise rests on."""
    context: pd.DataFrame
    """Per trajectory: the context features, the dataset, the batch and the stratum."""

    def step(self, name: str, instruction: str, files: list[str]) -> Any:
        """Run one function of the sequence in the workspace, alone, and return what it submitted.

        The request holds the digest of every file the function builds on, so the call is answered from the mailbox
        when a round runs again over the same files. A function stopped before it submitted runs again from the files
        the driver wrote: its notes and the operator files the driver did not place go first.
        """
        present = [file for file in files if (self.workspace / file).is_file()]
        digests = {file: hashlib.sha256((self.workspace / file).read_bytes()).hexdigest() for file in present}
        request = StepRequest(round=self.number, files=digests, model=mining_model(self.project))
        leftovers = [self.workspace / "notes" / f"{name}.md", *self.unplaced("operators")]
        return self.alone(name, instruction, request, leftovers)

    def link_case(self, key: str) -> str:
        """Make the trajectory readable as `cases/<key>.md` and return that path; cases come from the evaluation
        set too, which `trajectories/` leaves out."""
        file = f"cases/{key}.md"
        link = self.workspace / file
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.is_symlink():
            link.symlink_to(self.project.trajectory_path(key, ".md"))
        return file


@dataclass
class Item:
    """One definition on trial: a candidate, or the revision of a candidate or of an enabled mined feature."""

    definition: Definition
    examples: list[Example] = field(default_factory=list)
    hypothesis: str | None = None
    parent: str | None = None
    replaces: str | None = None
    """The enabled mined feature this definition takes the place of."""
    noisy: bool = False
    """An enabled mined feature above the noise limit, which is removed when no revision of it is accepted."""
    revised: bool = False
    source: Path | None = None
    """The operator file of a code candidate."""
    noise: float | None = None
    compared: int = 0
    spread: Spread | None = None
    values: dict[str, Any] = field(default_factory=dict)
    """The value on every reference trajectory the trial measured."""
    steps: dict[str, tuple[int, ...]] = field(default_factory=dict)
    """The steps each value rests on, when the operator gives them."""
    tail_precision: float | None = None
    lift: float | None = None
    alert: bool = False
    decision: Literal["add", "reject", "revise"] | None = None
    reason: str = ""
    revise_request: ReviseRequest | None = None
    tail_request: TailRequest | None = None

    @property
    def name(self) -> str:
        return self.definition.spec.name

    def settle(self, decision: Literal["add", "reject", "revise"], reason: str = "") -> None:
        self.decision, self.reason = decision, reason

    def verdict(self) -> Verdict:
        assert self.decision in ("add", "reject")
        return Verdict(
            feature=self.name, reads=self.definition.reads, hypothesis=self.hypothesis, parent=self.parent,
            noise=self.noise, compared=self.compared, distinct=self.spread.distinct if self.spread else 0,
            varying_share=self.spread.varying_share if self.spread else None,
            batch_explained=self.spread.batch_explained if self.spread else None,
            tail_precision=self.tail_precision, lift=self.lift, alert=self.alert, decision=self.decision,
            reason=self.reason)
