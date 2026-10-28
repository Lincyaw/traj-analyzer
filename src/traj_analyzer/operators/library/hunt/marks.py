from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, PrivateAttr, ValidationInfo, model_validator

from traj_analyzer.hunting.kept import Marks, load_marks, mark
from traj_analyzer.layers.tables import Tables
from traj_analyzer.operators.base import EventMark, FeatureSpec, Operator
from traj_analyzer.operators.catalog import ROOT
from traj_analyzer.schema import Trajectory


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hunt: str | None = None
    """The hunt whose kept probes mark the trajectories, `hunts/<hunt>/`; a finished run of the hunt enables the
    operator with it."""
    _marks: Marks | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _load(self, info: ValidationInfo) -> Params:
        if self.hunt is None:
            return self
        if not info.context or ROOT not in info.context:
            raise ValueError("hunt.marks reads the probes a hunt kept in its project; enable it in traj.yaml")
        self._marks = load_marks(Path(info.context[ROOT]), self.hunt)
        return self

    @property
    def marks(self) -> Marks:
        if self._marks is None:
            raise ValueError("hunt.marks needs params.hunt, the hunt whose kept probes mark the trajectories")
        return self._marks


def outputs(params: Params) -> list[FeatureSpec]:
    if params.hunt is None:
        return []
    marks = params.marks
    names = {facet.name: [entry.name for entry, _ in [*marks.sql, *marks.code] if entry.facet == facet.name]
             for facet in marks.facets}
    return [FeatureSpec(name=facet.name, type="event", event=EventMark(edge=facet.edge),
                        description=f"Hunt {marks.hunt}: {facet.definition.strip()}"
                                    + (f" Does not count: {facet.excludes.strip()}" if facet.excludes.strip() else "")
                                    + f" Marked where its probes hit: {', '.join(names[facet.name]) or 'none'}.")
            for facet in marks.facets]


def compute(trajectory: Trajectory, tables: Tables, params: Params) -> dict[str, Any]:
    return mark(params.marks, trajectory, tables)


def tables_of(params: Params) -> tuple[str, ...]:
    return params.marks.tables() if params.hunt is not None else ()


OPERATOR = Operator(
    kind="code",
    description="The marks of a hunt: one event feature per facet the hunt kept, whose events are the positions its "
                "kept probes hit in a trajectory, each note naming the probe. Every trajectory is marked on its own, "
                "with no model call; `traj hunt` enables it as `hunt-<name>` when a run finishes.",
    tags=("hunt",),
    outputs=outputs,
    compute=compute,
    params=Params,
    tables_of=tables_of,
)
