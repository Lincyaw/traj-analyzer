from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa

from traj_analyzer.hunting.ledger import ProbeState
from traj_analyzer.mining.operators import compute_operator, load_code_operator
from traj_analyzer.project import Project

COLUMNS = ["key", "step", "note"]
"""What a probe finds, one row per position."""
HITS = ["probe", *COLUMNS]
HITS_SCHEMA = pa.schema([("probe", pa.string()), ("key", pa.string()), ("step", pa.int64()), ("note", pa.string())])
"""Every position the probes of a round found, as `rounds/<round>/hits.parquet` of the workspace holds them."""


class ProbeError(ValueError):
    """A probe whose result is not one row per position of a trajectory of the hunt."""


def probe_path(workspace: Path, probe: ProbeState) -> Path:
    return workspace / "probes" / f"{probe.name}.{'sql' if probe.kind == 'sql' else 'py'}"


def run_sql(workspace: Path, text: str, keys: set[str]) -> pd.DataFrame:
    """The positions one SQL probe finds, with the workspace as the directory its table paths start from."""
    connection = duckdb.connect()
    connection.execute(f"SET file_search_path = '{workspace}'")
    statements = connection.extract_statements(text)
    if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
        raise ProbeError("a SQL probe is one SELECT statement")
    frame = connection.sql(text).df()
    if sorted(frame.columns) != sorted(COLUMNS):
        raise ProbeError(f"a SQL probe returns the columns {COLUMNS}, this one returns {list(frame.columns)}")
    return checked(frame[COLUMNS], keys)


def run_code(project: Project, path: Path, name: str, keys: set[str]) -> pd.DataFrame:
    """The positions one code probe finds: the events of its one output on every trajectory of the hunt."""
    operator = load_code_operator(path)
    (feature,) = operator.outputs(operator.params())
    if feature.type != "event" or feature.name != name:
        raise ProbeError(f"a code probe has one output of type event named {name}; this one has {feature.type} "
                         f"{feature.name}")
    values, _ = compute_operator(project, operator, sorted(keys))
    rows = [{"key": key, "step": event["step"], "note": event["note"]}
            for key, events in values.items() for event in events]
    return checked(pd.DataFrame(rows, columns=COLUMNS), keys)


def checked(frame: pd.DataFrame, keys: set[str]) -> pd.DataFrame:
    if frame["key"].isna().any() or frame["step"].isna().any():
        raise ProbeError("every position has a key and a step")
    unknown = sorted(set(frame["key"]) - keys)
    if unknown:
        raise ProbeError(f"keys of no trajectory of the hunt: {unknown[:5]}")
    steps = pd.to_numeric(frame["step"])
    if not (steps == steps.round()).all():
        raise ProbeError("steps are whole numbers")
    found = pd.DataFrame({"key": frame["key"].astype(str), "step": steps.astype("int64"),
                          "note": frame["note"].fillna("").astype(str)})
    return found.drop_duplicates().sort_values(["key", "step", "note"]).reset_index(drop=True)


def run_probes(
    project: Project, workspace: Path, probes: list[ProbeState], keys: set[str]
) -> tuple[pd.DataFrame, dict[str, str]]:
    """The positions every probe finds, and the error of every probe that failed to run."""
    found = []
    failed: dict[str, str] = {}
    for probe in probes:
        path = probe_path(workspace, probe)
        # A probe written by a model can fail in any way; every failure is recorded for the next proposal.
        try:
            if probe.kind == "sql":
                frame = run_sql(workspace, path.read_text(encoding="utf-8"), keys)
            else:
                frame = run_code(project, path, probe.name, keys)
        except Exception as error:
            failed[probe.name] = f"{type(error).__name__}: {error}"
            continue
        found.append(frame.assign(probe=probe.name)[HITS])
    hits = pd.concat(found, ignore_index=True) if found else HITS_SCHEMA.empty_table().to_pandas()
    return hits, failed
