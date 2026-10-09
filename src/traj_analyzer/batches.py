from __future__ import annotations

import numpy as np
import pandas as pd

from traj_analyzer.ingest import IndexRow
from traj_analyzer.project import ConfigError, Project
from traj_analyzer.vectorize import Frame

META = "meta."
"""Prefix of a batch or stratum column read from the trajectory metadata."""
ALL = "all"
"""The one batch or stratum when traj.yaml declares none."""
QUARTILES = 4
SEPARATOR = " | "


def assign(project: Project, frame: Frame, rows: list[IndexRow], names: list[str]) -> pd.Series:
    """One label per trajectory of the frame from the columns `names`: the values joined by ` | `.

    A name is a feature column of the query view or `meta.<key>` of the metadata. A numeric column is cut into
    quartiles over the trajectories that have it, labelled `<name>:q1` to `<name>:q4`; a missing value reads `none`.
    """
    index = frame.query.index
    if not names:
        return pd.Series(ALL, index=index, dtype=object)
    parts = []
    for name in names:
        if name.startswith(META):
            key = name[len(META):]
            series = pd.Series({row.key: row.metadata.get(key) for row in rows}, dtype=object).reindex(index)
        elif name in frame.query.columns:
            series = frame.query[name]
        else:
            raise ConfigError(f"{name} in batches or strata is neither an extracted feature nor meta.<key>")
        parts.append(_labels(name, series))
    joined = parts[0]
    for part in parts[1:]:
        joined = joined + SEPARATOR + part
    return joined


def _labels(name: str, series: pd.Series) -> pd.Series:
    values = series.astype(object)
    numeric = pd.to_numeric(values.where(values.notna()), errors="coerce")
    is_number = numeric.notna() | values.isna()
    if is_number.all() and not all(isinstance(v, bool) for v in values.dropna()):
        known = numeric.dropna()
        edges = np.unique(np.quantile(known, np.linspace(0, 1, QUARTILES + 1))) if len(known) else np.array([])
        if len(edges) >= 2:
            bins = pd.cut(numeric, edges, labels=[f"{name}:q{i}" for i in range(1, len(edges))], include_lowest=True)
            return pd.Series(bins.astype(object), index=series.index).map(lambda v: "none" if pd.isna(v) else str(v))
    return values.map(lambda v: "none" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
