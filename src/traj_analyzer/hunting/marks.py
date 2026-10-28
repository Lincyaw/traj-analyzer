from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa

from traj_analyzer.hunting.functions import Facet
from traj_analyzer.hunting.kept import KeptProbe, read_kept
from traj_analyzer.hunting.probes import CONTENTS, CONTENTS_SCHEMA, contents_rows, mark_sql, tables_read
from traj_analyzer.hunting.spec import load_hunt
from traj_analyzer.ingest import visible_metadata
from traj_analyzer.layers.tables import KEY, Tables, with_key
from traj_analyzer.mining.operators import load_code_operator
from traj_analyzer.operators.base import Event, Operator
from traj_analyzer.project import ConfigError, Project
from traj_analyzer.schema import Trajectory

OPERATOR = "hunt.marks"
"""The operator that marks every trajectory with the probes a hunt kept."""


def marks_instance(hunt: str) -> str:
    """The instance of the marks operator of a hunt in traj.yaml, which is also its execution group."""
    return f"hunt-{hunt}"


@dataclass(frozen=True)
class Marks:
    """What marks the trajectories of a hunt: its kept facets and probes, the code probes loaded, the datasets it
    covers, and the metadata keys `render.hide_metadata` keeps from the probes, as from every model."""

    hunt: str
    facets: list[Facet]
    sql: list[tuple[KeptProbe, str]]
    code: list[tuple[KeptProbe, Operator]]
    datasets: Literal["all"] | list[str]
    hidden: list[str]

    def tables(self) -> tuple[str, ...]:
        """The tables of the parsing layers the probes read."""
        found = {table for _, text in self.sql for table in tables_read(text)} - {"steps", "trajectories", CONTENTS}
        found |= {table for _, operator in self.code for table in operator.tables_with(operator.params())}
        return tuple(sorted(found))

    def covers(self, trajectory: Trajectory) -> bool:
        return self.datasets == "all" or trajectory.dataset in self.datasets


def load_marks(root: Path, hunt: str) -> Marks:
    project = Project.load(root)
    spec = load_hunt(project, hunt)
    directory = project.hunts_dir / hunt
    kept = read_kept(directory)
    if kept is None:
        raise ConfigError(f"Hunt {hunt} has kept no probes; a finished run of it keeps them in {directory}")
    return Marks(hunt=hunt, facets=kept.facets,
                 sql=[(entry, text) for entry, text in kept.probes if entry.kind == "sql"],
                 code=[(entry, load_code_operator(directory / entry.file)) for entry, _ in kept.probes
                       if entry.kind == "code"],
                 datasets=spec.datasets, hidden=project.config.render.hide_metadata)


def trajectory_relations(trajectory: Trajectory, tables: Tables, hidden: list[str]) -> dict[str, pa.Table]:
    """The tables of one trajectory as a SQL probe reads them: with their key column, the whole text of the steps as
    `contents`, and the metadata `hidden` names left out."""
    found = dict(tables.arrow)
    trajectories = found["trajectories"].to_pylist()
    for row in trajectories:
        row["metadata"] = visible_metadata(row["metadata"], hidden)
    found["trajectories"] = pa.Table.from_pylist(trajectories, schema=found["trajectories"].schema)
    found[CONTENTS] = pa.Table.from_pylist(contents_rows(trajectory), schema=CONTENTS_SCHEMA)
    return {name: table.add_column(0, with_key(table.schema).field(KEY), pa.array([trajectory.key] * table.num_rows,
                                                                                    pa.string()))
            for name, table in found.items()}


def mark_trajectory(marks: Marks, trajectory: Trajectory, tables: Tables) -> dict[str, Any]:
    """Per facet, the events of its probes in one trajectory, each note naming the probe; None for every facet on a
    trajectory of a dataset the hunt does not cover."""
    if not marks.covers(trajectory):
        return {facet.name: None for facet in marks.facets}
    found: dict[str, list[dict[str, Any]]] = {facet.name: [] for facet in marks.facets}
    if marks.sql:
        shown = trajectory_relations(trajectory, tables, marks.hidden)
        for entry, text in marks.sql:
            for row in mark_sql(text, shown, trajectory.key).itertuples():
                found[entry.facet].append(Event(step=int(row.step), note=f"{entry.name}: {row.note}")
                                          .model_dump(mode="json"))
    for entry, operator in marks.code:
        for value in operator.compute(trajectory, tables, operator.params())[entry.name] or []:
            event = Event.model_validate(value)
            found[entry.facet].append(event.model_copy(update={"note": f"{entry.name}: {event.note}"})
                                      .model_dump(mode="json"))
    return {name: sorted(events, key=lambda event: (event["step"], event["note"])) for name, events in found.items()}
