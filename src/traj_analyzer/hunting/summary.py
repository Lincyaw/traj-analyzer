from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from traj_analyzer.hunting.driver import REPORT, TIMELINE, VERDICTS_TABLE
from traj_analyzer.hunting.spec import load_hunt
from traj_analyzer.layers.tables import table_path
from traj_analyzer.project import ConfigError, Project
from traj_analyzer.reviews import read_verdicts

LISTED = 20
"""Trajectories listed per part of the summary."""


def finished_runs(project: Project, name: str) -> list[int]:
    """The runs of the hunt that have a report, in order."""
    root = project.hunting_dir / name
    return sorted(int(path.parent.name) for path in root.glob(f"*/{REPORT}") if path.parent.name.isdigit())


def finished_run(project: Project, name: str, run: int | None) -> Path:
    """The directory of a finished run of the hunt: `run`, or the last one with a report."""
    load_hunt(project, name)
    root = project.hunting_dir / name
    if run is not None:
        directory = root / str(run)
        if not (directory / REPORT).is_file():
            raise ConfigError(f"Run {run} of hunt {name} has no report; see `traj hunt status {name}`")
        return directory
    finished = finished_runs(project, name)
    if not finished:
        raise ConfigError(f"Hunt {name} has no finished run; see `traj hunt status {name}`")
    return root / str(finished[-1])


def timeline_minutes(path: Path) -> dict[tuple[int, str], float]:
    """Minutes from the opening of every step of a run to its end, by round and step; a step opened again after a
    stop counts from its last opening."""
    opened: dict[tuple[int, str], datetime] = {}
    minutes: dict[tuple[int, str], float] = {}
    if not path.is_file():
        return minutes
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        step = (entry["round"], entry["step"])
        at = datetime.fromisoformat(entry["at"])
        if entry["event"] == "open":
            opened[step] = at
        elif step in opened:
            minutes[step] = round((at - opened[step]).total_seconds() / 60, 1)
    return minutes


def review_agreement(project: Project, name: str, features: dict[str, str]) -> dict[str, Any]:
    """How the verdicts people gave on the marks of the hunt agree with the hunt's own verdicts: a person who found a
    trajectory worth a look for the feature of a facet agrees with a hunt verdict that found that facet in it."""
    path = table_path(project, VERDICTS_TABLE)
    read = {}
    if path.is_file():
        table = pq.read_table(path).to_pandas()
        read = {row.key: (bool(row.found), row.facet) for row in table[table["hunt"] == name].itertuples()}
    latest = {(verdict.feature, verdict.key): verdict for verdict in read_verdicts(project)
              if verdict.feature in features}
    agree, unread = 0, 0
    differ = []
    for (feature, key), verdict in sorted(latest.items()):
        if key not in read:
            unread += 1
            continue
        found, facet = read[key]
        if verdict.worth == (found and facet == features[feature]):
            agree += 1
        else:
            differ.append({"key": key, "feature": feature, "worth": verdict.worth, "found": found, "facet": facet,
                           "note": verdict.note})
    return {"verdicts": len(latest), "agree": agree, "unread": unread, "differ": differ}


def summarize_run(project: Project, name: str, run: int | None = None) -> dict[str, Any]:
    """What a finished run of the hunt found, in brief: the rounds with their reads and minutes, the facets, the
    batches, the active probes, the coverage, how blind reads and people's verdicts agree, and where the files are."""
    directory = finished_run(project, name, run)
    report = json.loads((directory / REPORT).read_text(encoding="utf-8"))
    minutes = timeline_minutes(directory / TIMELINE)
    rounds = [{"round": record["round"], "reads": record["reads"], "blind": record["blind"],
               "found": record["found"], "new": len(record["new"]), "unanswered": len(record["unanswered"]),
               "minutes": {step: minutes.get((record["round"], step)) for step in ("propose", "read")}}
              for record in report["per_round"]]
    closing = report["rounds"] + 1
    calibration = report["calibration"]
    probes = report["probes"]
    return {
        "hunt": name, "run": report["run"], "rounds": report["rounds"], "converged": report["converged"],
        "found": report["found"],
        "per_round": rounds,
        "final": {**report["final"], "unanswered": len(report["final"]["unanswered"]),
                  "minutes": minutes.get((closing, "close"))},
        "facets": [{"name": facet["name"], "edge": facet["edge"], "found": facet["found"],
                    "definition": facet["definition"]} for facet in report["facets"]],
        "batches": report["batches"],
        "probes": [{field: row[field] for field in ("name", "facet", "hits", "read", "found", "precision")}
                   for row in probes if row["status"] == "active"],
        "inactive_probes": sum(row["status"] != "active" for row in probes),
        "coverage": report["coverage"],
        "calibration": {**{key: value for key, value in calibration.items() if key != "differ"},
                        "differ": [item["key"] for item in calibration["differ"]][:LISTED]},
        "reviews": review_agreement(project, name, report["marks"]["features"]),
        "unverified": {probe: len(keys) for probe, keys in report["unverified"].items()},
        "marks": report["marks"],
        "files": {"report": str((directory / REPORT).relative_to(project.root)),
                  "verdicts": str(table_path(project, VERDICTS_TABLE).relative_to(project.root)),
                  "kept": str((project.hunts_dir / name).relative_to(project.root))},
    }
