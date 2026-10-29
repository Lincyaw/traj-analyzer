from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa

from traj_analyzer.hunting.contents import CONTENTS, CONTENTS_SCHEMA, contents_rows
from traj_analyzer.hunting.functions import Facet
from traj_analyzer.hunting.kept import KeptProbe, read_kept
from traj_analyzer.hunting.probes import mark_connection, mark_sql, tables_read
from traj_analyzer.hunting.spec import load_hunt
from traj_analyzer.ingest import shown_keys, visible_metadata
from traj_analyzer.layers.tables import KEY, Tables, with_key
from traj_analyzer.mining.operators import load_code_operator
from traj_analyzer.operators.base import Event, Operator
from traj_analyzer.operators.catalog import edit_operators
from traj_analyzer.project import ConfigError, Project
from traj_analyzer.schema import Trajectory

OPERATOR = "hunt.marks"
"""The operator that marks every trajectory with the probes a hunt kept."""


def marks_instance(hunt: str) -> str:
    """The instance of the marks operator of a hunt in traj.yaml, which is also its execution group."""
    return f"hunt-{hunt}"


def enable_marks(project: Project, hunt: str) -> Project:
    """Enable a hunt's marks with its own feature prefix, preserving any existing registration."""
    group = marks_instance(hunt)

    def enable(operators: list[Any]) -> None:
        if not any(operator.get("as") == group for operator in operators):
            operators.append({"use": OPERATOR, "as": group, "prefix": hunt.replace("-", "_"),
                              "params": {"hunt": hunt}})

    return edit_operators(project, enable)


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
    found["trajectories"] = _shown_trajectories(found["trajectories"], hidden)
    found[CONTENTS] = pa.Table.from_pylist(contents_rows(trajectory), schema=CONTENTS_SCHEMA)
    return {name: table.add_column(0, with_key(table.schema).field(KEY), pa.array([trajectory.key] * table.num_rows,
                                                                                    pa.string()))
            for name, table in found.items()}


def _shown_trajectories(table: pa.Table, hidden: list[str]) -> pa.Table:
    rows = table.to_pylist()
    for row in rows:
        row["metadata"] = visible_metadata(row["metadata"], hidden)
    return pa.Table.from_pylist(rows, schema=table.schema)


def mark_trajectory(marks: Marks, trajectory: Trajectory, tables: Tables) -> dict[str, Any]:
    """Per facet, the events of its probes in one trajectory, each note naming the probe; None for every facet on a
    trajectory of a dataset the hunt does not cover.

    Both kinds of probe read the trajectory with the metadata `render.hide_metadata` names left out, as the tables
    of the workspace leave it out for the probes of a round.
    """
    if not marks.covers(trajectory):
        return {facet.name: None for facet in marks.facets}
    found: dict[str, list[dict[str, Any]]] = {facet.name: [] for facet in marks.facets}
    if marks.sql:
        relations = trajectory_relations(trajectory, tables, marks.hidden)
        connection = mark_connection(relations)
        names = set(relations)
        for entry, text in marks.sql:
            for row in mark_sql(connection, text, names, trajectory.key).itertuples():
                found[entry.facet].append(Event(step=int(row.step), note=f"{entry.name}: {row.note}")
                                          .model_dump(mode="json"))
    if marks.code:
        shown = trajectory.model_copy(update={"metadata": shown_keys(trajectory.metadata, marks.hidden)})
        inside = Tables({**tables.arrow, "trajectories": _shown_trajectories(tables.arrow["trajectories"],
                                                                            marks.hidden)})
        for entry, operator in marks.code:
            for value in operator.compute(shown, inside, operator.params())[entry.name] or []:
                event = Event.model_validate(value)
                found[entry.facet].append(event.model_copy(update={"note": f"{entry.name}: {event.note}"})
                                          .model_dump(mode="json"))
    return {name: sorted(events, key=lambda event: (event["step"], event["note"])) for name, events in found.items()}
