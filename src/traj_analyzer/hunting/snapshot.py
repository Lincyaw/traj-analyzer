from __future__ import annotations

import json
import shutil
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from traj_analyzer.hunting.contents import contents_rows
from traj_analyzer.hunting.run import CLUSTERS
from traj_analyzer.hunting.summary import finished_run
from traj_analyzer.ingest import load_trajectory, visible_metadata
from traj_analyzer.layers.tables import read_table, steps_rows, trajectory_row
from traj_analyzer.project import ConfigError, Project


def validate_extracted_trajectories(project: Project, keys: list[str]) -> None:
    """Require the extracted trajectory table to cover every trajectory the hunt will search."""
    extracted = set(read_table(project, "trajectories")["key"].to_pylist())
    missing = sorted(set(keys) - extracted)
    if missing:
        raise ConfigError(f"{len(missing)} hunt trajectories have no extracted tables, starting at {missing[0]}; "
                          "run `traj extract` before starting the hunt")


def repair_snapshot_tables(project: Project, name: str, run: int | None, *, dry_run: bool = False) -> dict[str, Any]:
    """Complete missing core table rows only when the source still matches the run's saved whole step contents."""
    directory = finished_run(project, name, run)
    keys = set(json.loads((directory / CLUSTERS).read_text(encoding="utf-8")))
    tables = directory / "workspace" / "tables"
    trajectories = pq.read_table(tables / "trajectories.parquet")
    steps = pq.read_table(tables / "steps.parquet")
    present = trajectories["key"].to_pylist()
    if len(present) != len(set(present)) or set(present) - keys:
        raise ConfigError("The saved trajectory table has duplicate keys or keys outside the hunt")
    missing = sorted(keys - set(present))
    if set(steps["key"].to_pylist()) - set(present):
        raise ConfigError("The saved steps include trajectories absent from the trajectory table")
    contents = pq.read_table(tables / "contents.parquet")
    added_trajectories, added_steps = [], []
    for key in missing:
        trajectory = load_trajectory(project, key)
        saved = contents.filter(pc.equal(contents["key"], key)).drop(["key"]).sort_by("step").to_pylist()
        if contents_rows(trajectory) != saved:
            raise ConfigError(f"{key} no longer matches the whole step contents saved by the hunt")
        row = trajectory_row(trajectory)
        row["metadata"] = visible_metadata(row["metadata"], project.config.render.hide_metadata)
        added_trajectories.append({"key": key, **row})
        added_steps.extend({"key": key, **step} for step in steps_rows(trajectory, project.config.render))
    replacements = {
        "trajectories": pa.concat_tables([
            trajectories, pa.Table.from_pylist(added_trajectories, schema=trajectories.schema)]),
        "steps": pa.concat_tables([steps, pa.Table.from_pylist(added_steps, schema=steps.schema)]),
    }
    counts = replacements["steps"].group_by("key").aggregate([("step", "count")]).to_pylist()
    observed = {row["key"]: row["step_count"] for row in counts}
    if any(observed.get(row["key"], 0) != row["steps"] for row in replacements["trajectories"].to_pylist()):
        raise ConfigError("The saved step counts do not match the trajectory table")
    if missing and not dry_run:
        for table in replacements:
            path = tables / f"{table}.parquet"
            backup = path.with_suffix(".before-repair.parquet")
            if backup.exists():
                raise ConfigError(f"Snapshot backup already exists: {backup}")
        for table, replacement in replacements.items():
            path = tables / f"{table}.parquet"
            shutil.copyfile(path, path.with_suffix(".before-repair.parquet"))
            pq.write_table(replacement, path)
    return {"hunt": name, "run": int(directory.name), "dry_run": dry_run,
            "trajectories": len(keys), "added": missing, "added_steps": len(added_steps)}
