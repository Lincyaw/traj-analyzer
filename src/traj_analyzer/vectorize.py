from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from traj_analyzer.features.table import KEY, feature_values, read_group, status_column
from traj_analyzer.ingest import IndexRow
from traj_analyzer.operators.base import FeatureSpec
from traj_analyzer.operators.catalog import Group
from traj_analyzer.project import ConfigError, Project

MAX_FREE_LABELS = 100
"""Columns for category, set and map features without labels: one per label, for the most frequent labels."""
LABEL_TYPES = ("category", "set", "map")
"""Feature types whose distance columns are one per label, kept on their own scale of 0 to 1."""


@dataclass
class Matrix:
    """The space strategies measure: `x` holds the scaled columns, `raw` the value of each column before scaling."""

    x: np.ndarray
    columns: list[str]
    raw: np.ndarray


@dataclass
class Frame:
    """Two views of the feature table, both indexed by trajectory key.

    `query` holds readable columns for filtering, and `distance` holds numeric columns for clustering.
    `columns` maps each feature to its columns in `distance`, `types` to its feature type, and `extracted` to the keys
    it was extracted for.
    """

    query: pd.DataFrame
    distance: pd.DataFrame
    columns: dict[str, list[str]] = field(default_factory=dict)
    types: dict[str, str] = field(default_factory=dict)
    extracted: dict[str, set[str]] = field(default_factory=dict)

    def rows(self, index: pd.Index) -> Frame:
        return Frame(query=self.query.loc[index], distance=self.distance.loc[index], columns=self.columns,
                     types=self.types, extracted=self.extracted)

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
        return self.rows(self.distance.index[mask])

    def matrix(self, weights: dict[str, float]) -> Matrix:
        """Scale each feature as a whole so its columns' variances sum to its weight squared.

        Scalar, boolean and vector columns are standardized first. Label columns of category, set and map features
        keep their 0 to 1 values, so a label few trajectories have adds little to distances. A trajectory without a
        value for a feature has 0 in its label columns or the mean in its other columns, and 1 in the column
        `<feature>__missing`, which exists when some trajectory lacks the value.
        """
        unknown = sorted(set(weights) - set(self.columns))
        if unknown:
            raise ConfigError(f"No extracted values for sampler features {unknown}")
        scaled, raw = [], []
        for feature, weight in weights.items():
            if weight == 0:
                continue
            values = self.distance[self.columns[feature]].astype(float)
            missing = values.isna().all(axis=1)
            if self.types[feature] in LABEL_TYPES:
                block = values.fillna(0.0)
            else:
                block = ((values - values.mean()) / values.std(ddof=0).replace(0, 1.0)).fillna(0.0)
            if missing.any():
                name = missing_column(feature)
                if name in values.columns:
                    raise ConfigError(f"{feature}: the label 'missing' would share its column with {name}")
                block[name] = missing.astype(float)
                values[name] = missing.astype(float)
            total = float(block.var(ddof=0).sum())
            scaled.append(block * weight / np.sqrt(total) if total > 0 else block)
            raw.append(values)
        if not scaled:
            raise ConfigError("The sampler has no features with extracted values")
        x = pd.concat(scaled, axis=1)
        return Matrix(x=x.to_numpy(), columns=list(x.columns), raw=pd.concat(raw, axis=1).to_numpy())


def missing_column(feature: str) -> str:
    return f"{feature}__missing"


def label_column(feature: str, label: object) -> str:
    """Column of one label of a category, set or map feature: the label verbatim, so distinct labels never share one."""
    return f"{feature}__{label}"


def build_frame(
    project: Project, groups: list[Group], trajectories: list[IndexRow], vector_length: int
) -> Frame:
    """Read the feature table as the last `traj extract` left it, for the given trajectories and enabled features."""
    query: dict[str, pd.Series] = {}
    distance: dict[str, pd.Series] = {}
    columns: dict[str, list[str]] = {}
    types: dict[str, str] = {}
    extracted: dict[str, set[str]] = {}
    index = pd.Index([t.key for t in trajectories], name="key")
    wanted = set(index)
    for group in groups:
        table = read_group(project, group)
        keys = table[KEY].to_pylist()
        for feature in group.features:
            statuses = table[status_column(feature.name)].to_pylist()
            present = {key for key, status in zip(keys, statuses, strict=True) if status is not None and key in wanted}
            if not present:
                continue
            raw = {key: value for key, status, value in zip(keys, statuses, feature_values(table, feature), strict=True)
                   if status == "ok" and key in wanted}
            q, d = _expand(feature, raw, index, vector_length)
            query.update(q)
            distance.update(d)
            columns[feature.name] = list(d)
            types[feature.name] = feature.type
            extracted[feature.name] = present
    return Frame(
        query=pd.DataFrame(query, index=index),
        distance=pd.DataFrame(distance, index=index),
        columns=columns,
        types=types,
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
            onehot = {label_column(name, lb): (series == lb).astype(float).where(series.notna()) for lb in labels}
            return {name: series.astype("string")}, onehot
        case "set":
            labels = list(feature.labels or ()) or _top_values(series)
            if "count" in labels:
                raise ConfigError(f"{name}: the label 'count' would share its column with {name}__count")
            multihot = {
                label_column(name, lb): series.map(
                    lambda v, lb=lb: float(lb in v) if isinstance(v, list) else None)
                for lb in labels
            }
            count = series.map(lambda v: len(v) if isinstance(v, list) else None)
            return {**multihot, f"{name}__count": count}, multihot
        case "map":
            keys = list(feature.labels or ()) or _top_values(
                series.map(lambda v: list(v) if isinstance(v, dict) else None))
            values = {
                label_column(name, key): series.map(
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
