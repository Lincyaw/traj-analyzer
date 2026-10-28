from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from traj_analyzer.hunting.ledger import HuntState
from traj_analyzer.hunting.probes import readable_tables, run_code, run_sql
from traj_analyzer.hunting.run import CLUSTERS, WORKSPACE, last_run, resume
from traj_analyzer.hunting.spec import load_hunt
from traj_analyzer.project import ConfigError, Project

LISTED = 10
"""Trajectories listed per part of what a try prints."""
SAMPLE = 20


def try_probe(project: Project, name: str, file: Path) -> dict[str, Any]:
    """Run a probe file over every trajectory of the last run of the hunt, as a round runs it, and compare what it hits
    with the verdicts so far and with the hits of the active probes of the last round that ran its probes.

    The probe is the file's stem; a `.sql` file is a SQL probe, a `.py` file a code probe.
    """
    load_hunt(project, name)
    directory = last_run(project, name)
    if directory is None:
        raise ConfigError(f"Hunt {name} has no run; start one with `traj hunt run {name}` or `traj hunt start {name}`")
    workspace = directory / WORKSPACE
    keys = set(json.loads((directory / CLUSTERS).read_text(encoding="utf-8")))
    probe = file.stem
    if file.suffix not in (".sql", ".py"):
        raise ConfigError(f"{file} is neither a .sql file nor a .py file")
    # A probe being written can fail in any way; the failure is what the writer needs to see.
    try:
        if file.suffix == ".sql":
            found = run_sql(workspace, file.read_text(encoding="utf-8"), keys, readable_tables(project))
        else:
            found = run_code(project, file.resolve(), probe, keys, project.config.render.hide_metadata)
    except Exception as error:
        raise ConfigError(f"the probe does not run: {type(error).__name__}: {error}") from error
    state = resume(directory) or HuntState()
    last = _last_hits(workspace)
    hit = sorted(set(found["key"]))
    readings = state.readings
    read = [readings[key] for key in hit if key in readings]
    rejected = [reading for reading in read if not reading.found]
    steps = found.groupby("key")["step"].apply(list).to_dict()
    covered = set(last["key"])
    new = [key for key in hit if key not in covered]
    out: dict[str, Any] = {
        "probe": probe, "kind": "sql" if file.suffix == ".sql" else "code",
        "positions": len(found), "trajectories": len(hit),
        "read": {"read": len(read), "found": len(read) - len(rejected),
                 "facets": dict(Counter(reading.facet for reading in read if reading.found)),
                 "precision": round((len(read) - len(rejected)) / len(read), 4) if read else None},
        "rejected": {"count": len(rejected), "listed": [
            {"key": reading.key, "steps": steps[reading.key], "why": reading.why} for reading in rejected[:LISTED]]},
        "misses": [{"key": reading.key, "facet": reading.facet, "steps": reading.steps}
                   for reading in readings.values() if reading.found and not reading.hits and reading.key in steps],
        "new": {"trajectories": len(new), "unread": sum(key not in readings for key in new),
                "listed": [{"key": key, "steps": steps[key], "verdict": _verdict_of(state, key)}
                           for key in new[:LISTED]]},
        "same_name": None,
        "sample": [{"key": row.key, "step": int(row.step), "note": row.note, "verdict": _verdict_of(state, row.key)}
                   for row in found.drop_duplicates("key").head(SAMPLE).itertuples()],
    }
    if probe in set(last["probe"]):
        before = set(last.loc[last["probe"] == probe, "key"])
        lost = sorted(before - set(hit))
        out["same_name"] = {"trajectories": len(before), "gained": len(set(hit) - before), "lost": len(lost),
                            "lost_found": [key for key in lost if key in readings and readings[key].found]}
    return out


def _last_hits(workspace: Path) -> pd.DataFrame:
    """The hits of the active probes of the last round that ran its probes."""
    rounds = sorted((int(path.parent.name) for path in (workspace / "rounds").glob("*/hits.parquet")
                     if path.parent.name.isdigit()), reverse=True)
    if not rounds:
        return pd.DataFrame(columns=["probe", "key", "step", "note"])
    return pq.read_table(workspace / "rounds" / str(rounds[0]) / "hits.parquet").to_pandas()


def _verdict_of(state: HuntState, key: str) -> str:
    reading = state.readings.get(key)
    if reading is None:
        return "unread"
    return f"found {reading.facet}" if reading.found else "rejected"
