from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from traj_analyzer.batches import assign
from traj_analyzer.hunting.draw import clusters_of
from traj_analyzer.hunting.kept import read_kept
from traj_analyzer.hunting.ledger import HuntState, ProbeState
from traj_analyzer.hunting.spec import HuntSpec, hunt_model, load_hunt
from traj_analyzer.ingest import load_index
from traj_analyzer.layers.tables import table_path
from traj_analyzer.operators.catalog import load_groups
from traj_analyzer.project import ConfigError, Project
from traj_analyzer.vectorize import Matrix, build_frame

WORKSPACE = "workspace"
STATE = "state"
REPORT = "report.json"
CLUSTERS = "clusters.json"
TIMELINE = "timeline.jsonl"
"""When every step of a run was opened and done, one JSON line each."""
PENDING = "pending.json"
"""A round of an agent session that waits for its verdicts (section 18.9 of the design)."""
FINAL = "final"
"""Directory of the closing check under `rounds/` of the workspace."""
Step = Literal["propose", "read", "close"]


@dataclass(frozen=True)
class Hunt:
    """What every round of one run of a hunt shares."""

    spec: HuntSpec
    project: Project
    directory: Path
    keys: list[str]
    matrix: Matrix
    clusters: dict[str, int]
    batches: dict[str, tuple[str, str]]
    """Per trajectory, its batch and stratum under `batches` and `strata` of traj.yaml."""

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def workspace(self) -> Path:
        return self.directory / WORKSPACE

    @property
    def model(self) -> str:
        return hunt_model(self.project, self.spec)


def open_hunt(project: Project, name: str, *, create: bool, max_rounds: int | None = None,
              new_run: bool = False) -> Hunt:
    """The run of a hunt with what its rounds share; `create` starts a new run when the last one is finished."""
    if new_run and not create:
        raise ConfigError("A new hunt run requires create=True")
    spec = load_hunt(project, name)
    if max_rounds is not None:
        spec = HuntSpec.model_validate({**spec.model_dump(), "max_rounds": max_rounds})
    rows = sorted(load_index(project, None if spec.datasets == "all" else spec.datasets), key=lambda row: row.key)
    if not rows:
        raise ConfigError(f"Hunt {name} has no trajectories; run `traj ingest`")
    if not table_path(project, "steps").is_file():
        raise ConfigError("No tables yet; run `traj extract`")
    frame = build_frame(project, load_groups(project), rows, project.config.sampling.vector_length)
    if not frame.columns:
        raise ConfigError("No extracted features tell the trajectories apart; run `traj extract`")
    keys = [row.key for row in rows]
    directory = run_directory(project, name, new_run=new_run) if create else open_run(project, name)
    matrix = frame.matrix(dict.fromkeys(frame.columns, 1.0))
    batch = assign(project, frame, rows, project.config.batches)
    stratum = assign(project, frame, rows, project.config.strata)
    return Hunt(spec=spec, project=project, directory=directory, keys=keys, matrix=matrix,
                clusters=_clusters(directory, keys, matrix, spec),
                batches={key: (str(batch[key]), str(stratum[key])) for key in keys})


def over(spec: HuntSpec, state: HuntState) -> bool:
    """Whether the search has ended: `patience` rounds in a row without a new finding, or `max_rounds` rounds."""
    return state.quiet >= spec.patience or state.round >= spec.max_rounds


def run_numbers(project: Project, name: str) -> list[int]:
    """The runs of the hunt, by their directory number, in order."""
    root = project.hunting_dir / name
    return sorted(int(path.name) for path in root.glob("[0-9]*") if path.is_dir() and path.name.isdigit())


def finished_runs(project: Project, name: str) -> list[int]:
    """The runs of the hunt that have a report, in order."""
    return [number for number in run_numbers(project, name)
            if (project.hunting_dir / name / str(number) / REPORT).is_file()]


def run_directory(project: Project, name: str, *, new_run: bool = False) -> Path:
    """The directory of the run to continue: the last one when it has no report yet, a new one otherwise."""
    numbers = run_numbers(project, name)
    number = numbers[-1] if numbers else 1
    if numbers and (new_run or (project.hunting_dir / name / str(number) / REPORT).is_file()):
        number += 1
    directory = project.hunting_dir / name / str(number)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def last_run(project: Project, name: str) -> Path | None:
    numbers = run_numbers(project, name)
    return project.hunting_dir / name / str(numbers[-1]) if numbers else None


def open_run(project: Project, name: str) -> Path:
    """The run of the hunt in progress."""
    directory = last_run(project, name)
    if directory is None or (directory / REPORT).is_file():
        raise ConfigError(f"Hunt {name} has no run in progress; start one with `traj hunt start {name}`")
    return directory


def _clusters(directory: Path, keys: list[str], matrix: Matrix, spec: HuntSpec) -> dict[str, int]:
    """The clusters of the run, computed when it starts and kept for every round of it."""
    path = directory / CLUSTERS
    if path.is_file():
        clusters: dict[str, int] = json.loads(path.read_text(encoding="utf-8"))
        if sorted(clusters) != keys:
            raise ConfigError(f"The trajectories of the hunt changed since its run in {directory} started; "
                              f"start a separate run with `traj hunt run {spec.name} --new-run`")
        return clusters
    clusters = clusters_of(keys, matrix, spec.clusters, spec.seed)
    path.write_text(json.dumps(clusters, indent=1, sort_keys=True), encoding="utf-8")
    return clusters


def start(project: Project, name: str) -> HuntState:
    """The state a first round starts from: the facets and probes the hunt kept from its last run, when it has."""
    kept = read_kept(project.hunts_dir / name)
    if kept is None:
        return HuntState()
    probes = [ProbeState(name=entry.name, facet=entry.facet, kind=entry.kind, why=entry.why, since=0, text=text)
              for entry, text in kept.probes]
    return HuntState(facets=kept.facets, probes=probes)


def resume(directory: Path) -> HuntState | None:
    """The state after the last finished round of a run; None before its first round finishes."""
    states = sorted((directory / STATE).glob("*.json"), key=lambda path: int(path.stem))
    if not states:
        return None
    return HuntState.model_validate_json(states[-1].read_text(encoding="utf-8"))


def save_state(hunt: Hunt, state: HuntState) -> None:
    (hunt.directory / STATE).mkdir(exist_ok=True)
    (hunt.directory / STATE / f"{state.round}.json").write_text(state.model_dump_json(indent=1), encoding="utf-8")


def note_step(hunt: Hunt, number: int, step: Step, event: Literal["open", "done"]) -> None:
    """Record in the timeline of the run that a step of a round was opened or done."""
    line = {"round": number, "step": step, "event": event, "at": datetime.now(UTC).isoformat(timespec="seconds")}
    with (hunt.directory / TIMELINE).open("a", encoding="utf-8") as out:
        out.write(json.dumps(line) + "\n")
