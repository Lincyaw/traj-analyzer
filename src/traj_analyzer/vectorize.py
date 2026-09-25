from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from traj_analyzer.features.table import read_group
from traj_analyzer.ingest import IndexRow
from traj_analyzer.operators.base import FeatureSpec
from traj_analyzer.operators.catalog import Group
from traj_analyzer.project import ConfigError, Project

MAX_FREE_LABELS = 100
"""Columns for category, set and map features without labels: one per label, for the most frequent labels."""


@dataclass
class Frame:
    """Two views of the feature table, both indexed by trajectory key.

    `query` holds readable columns for filtering, and `distance` holds numeric columns for clustering.
    `columns` maps each feature to its columns in `distance`, and `extracted` to the keys it was extracted for.
    """

    query: pd.DataFrame
    distance: pd.DataFrame
    columns: dict[str, list[str]] = field(default_factory=dict)
    extracted: dict[str, set[str]] = field(default_factory=dict)

    def complete(self, features: list[str]) -> Frame:
        """Keep the trajectories that have a row for every one of `features` in the feature table.

        A row counts even when its value is empty: the operator did not apply, the value was None, or the LLM call
        was refused or failed.
        """
        unknown = sorted(set(features) - set(self.columns))
        if unknown:
            raise ConfigError(f"No extracted values for features {unknown}")
        mask = pd.Series(True, index=self.distance.index)
        for feature in features:
            mask &= self.distance.index.isin(list(self.extracted[feature]))
        return Frame(query=self.query[mask], distance=self.distance[mask], columns=self.columns,
                     extracted=self.extracted)

    def matrix(self, weights: dict[str, float]) -> np.ndarray:
        """Standardize every column, fill missing values with the column mean, and apply weights.

        A feature's weight is divided by the square root of its column count.
        """
        unknown = sorted(set(weights) - set(self.columns))
        if unknown:
            raise ConfigError(f"No extracted values for sampler features {unknown}")
        parts = []
        for feature, weight in weights.items():
            if weight == 0:
                continue
            cols = self.columns[feature]
            block = self.distance[cols].astype(float)
            std = block.std(ddof=0).replace(0, 1.0)
            block = ((block - block.mean()) / std).fillna(0.0)
            parts.append(block.to_numpy() * weight / np.sqrt(len(cols)))
        if not parts:
            raise ConfigError("The sampler has no features with extracted values")
        return np.hstack(parts)


def ident(label: str) -> str:
    cleaned = re.sub(r"\W", "_", str(label)).strip("_")
    if not cleaned:
        raise ConfigError(f"Label {label!r} has no characters usable in a column name")
    return cleaned


def build_frame(
    project: Project, groups: list[Group], trajectories: list[IndexRow], vector_length: int
) -> Frame:
    """Read the feature table as the last `traj extract` left it, for the given trajectories and enabled features."""
    query: dict[str, pd.Series] = {}
    distance: dict[str, pd.Series] = {}
    columns: dict[str, list[str]] = {}
    extracted: dict[str, set[str]] = {}
    index = pd.Index([t.key for t in trajectories], name="key")
    wanted = set(index)
    for group in groups:
        values: dict[str, dict[str, object]] = {f.name: {} for f in group.features}
        present: dict[str, set[str]] = {f.name: set() for f in group.features}
        for row in read_group(project, group.name):
            if row.key in wanted and row.feature in values:
                present[row.feature].add(row.key)
                if row.status == "ok":
                    values[row.feature][row.key] = row.value
        for feature in group.features:
            if not present[feature.name]:
                continue
            q, d = _expand(feature, values[feature.name], index, vector_length)
            query.update(q)
            distance.update(d)
            columns[feature.name] = list(d)
            extracted[feature.name] = present[feature.name]
    return Frame(
        query=pd.DataFrame(query, index=index),
        distance=pd.DataFrame(distance, index=index),
        columns=columns,
        extracted=extracted,
    )


def _expand(
    feature: FeatureSpec, raw: dict[str, object], index: pd.Index, length: int
) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    name = feature.name
    series = pd.Series(raw, dtype=object).reindex(index)
    match feature.type:
        case "scalar" | "boolean":
            col = pd.to_numeric(series.map(lambda v: None if v is None else float(v)))
            return {name: col}, {name: col}
        case "category":
            labels = list(feature.labels or ()) or _top_values(series)
            onehot = {f"{name}__{ident(lb)}": (series == lb).astype(float).where(series.notna()) for lb in labels}
            return {name: series.astype("string")}, onehot
        case "set":
            labels = list(feature.labels or ()) or _top_values(series)
            multihot = {
                f"{name}__{ident(lb)}": series.map(
                    lambda v, lb=lb: float(lb in v) if isinstance(v, list) else None)
                for lb in labels
            }
            count = series.map(lambda v: len(v) if isinstance(v, list) else None)
            return {**multihot, f"{name}__count": count}, multihot
        case "map":
            keys = list(feature.labels or ()) or _top_values(
                series.map(lambda v: list(v) if isinstance(v, dict) else None))
            values = {
                f"{name}__{ident(key)}": series.map(
                    lambda v, key=key: float(v.get(key, 0.0)) if isinstance(v, dict) else None)
                for key in keys
            }
            return values, values
        case "vector":
            aggregates = {
                f"{name}__{agg}": series.map(lambda v, fn=fn: fn(v) if isinstance(v, list) else None)
                for agg, fn in _AGGREGATES.items()
            }
            points = series.map(lambda v: _resample(v, length) if isinstance(v, list) else None)
            curve = {
                f"{name}__t{i}": points.map(lambda v, i=i: None if v is None else v[i])
                for i in range(length)
            }
            return aggregates, {**curve, **aggregates}
    raise AssertionError(feature.type)


_AGGREGATES = {
    "max": lambda v: float(max(v)),
    "min": lambda v: float(min(v)),
    "mean": lambda v: float(np.mean(v)),
    "first": lambda v: float(v[0]),
    "last": lambda v: float(v[-1]),
}


def _resample(values: list[float], length: int) -> list[float]:
    if len(values) == 1:
        return [float(values[0])] * length
    source = np.linspace(0, 1, len(values))
    return [float(v) for v in np.interp(np.linspace(0, 1, length), source, values)]


def _top_values(series: pd.Series) -> list[str]:
    """Most frequent values of a category column, or of the items of a set column."""
    counts = series.explode().dropna().value_counts()
    return [str(v) for v in counts.index[:MAX_FREE_LABELS]]
