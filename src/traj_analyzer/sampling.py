from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from traj_analyzer.files import read_yaml
from traj_analyzer.operators.catalog import Group
from traj_analyzer.project import ConfigError, Project
from traj_analyzer.vectorize import Frame, Matrix

OUTLIER_NEIGHBOURS = 10
"""An outlier's score is its mean distance to this many nearest neighbours."""
EXPLAINED_COLUMNS = 5
"""Columns named in each pick's reason: those where it stands furthest from what it is compared with."""


class StrategySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["target", "outlier", "diversity", "random"]
    quota: float | int | Literal["rest"]
    label: str | None = None
    where: str | None = None
    within: str = "diversity"
    relative_to: str | None = None
    """outlier only: a query column such as model; trajectories are scored among those sharing its value."""

    @model_validator(mode="after")
    def _relative_to_outliers(self) -> StrategySpec:
        if self.relative_to is not None and self.kind != "outlier":
            raise ValueError("relative_to applies only to outlier strategies")
        return self


class SamplerSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    budget: int = Field(ge=1)
    seed: int = 0
    datasets: list[str] | None = None
    features: Literal["all"] | list[str] | dict[str, float] = "all"
    population: Literal["complete", "all"] = "complete"
    """`complete` samples only trajectories with a feature-table row for every sampler feature."""
    strategies: list[StrategySpec] = Field(min_length=1)


class Pick(BaseModel):
    key: str
    strategy: str
    reason: dict[str, Any]


class StrategyReport(BaseModel):
    label: str
    kind: str
    quota: int
    picked: int
    matched: int | None = None
    unscored: int | None = None
    """outlier with relative_to: trajectories left out because their group has fewer than 3 or no value."""


class Sampling(BaseModel):
    population: int
    strategies: list[StrategyReport]
    picks: list[Pick]


def load_sampler(project: Project, name: str) -> SamplerSpec:
    return SamplerSpec.model_validate(read_yaml(project.samplers_dir / f"{name}.yaml"))


def weights_for(spec: SamplerSpec, frame: Frame) -> dict[str, float]:
    if spec.features == "all":
        return dict.fromkeys(frame.columns, 1.0)
    if isinstance(spec.features, list):
        return dict.fromkeys(spec.features, 1.0)
    return dict(spec.features)


def sample(spec: SamplerSpec, frame: Frame, groups: list[Group]) -> Sampling:
    """Run the strategies in order; each one takes its quota from what earlier ones left."""
    weights = weights_for(spec, frame)
    if spec.population == "complete":
        frame = frame.complete([feature for feature, weight in weights.items() if weight])
    rng = np.random.default_rng(spec.seed)
    keys = list(frame.query.index)
    m = frame.matrix(weights)
    thresholds = feature_thresholds(groups)
    picks: list[Pick] = []
    reports: list[StrategyReport] = []
    chosen: set[str] = set()
    left = spec.budget
    for n, strategy in enumerate(spec.strategies):
        quota = _quota(strategy.quota, spec.budget, left)
        label = strategy.label or f"{strategy.kind}-{n}"
        matched = None
        if strategy.kind == "target":
            matched = _matches(strategy, label, frame.query, thresholds)
        free = [i for i, k in enumerate(keys) if k not in chosen]
        new, unscored = ([], None) if quota == 0 else _run(strategy, label, quota, free, keys, m, frame.query,
                                                           matched, rng, spec.seed)
        chosen.update(pick.key for pick in new)
        picks += new
        left -= len(new)
        reports.append(StrategyReport(label=label, kind=strategy.kind, quota=quota, picked=len(new),
                                      matched=None if matched is None else len(matched), unscored=unscored))
    return Sampling(population=len(keys), strategies=reports, picks=picks)


def _quota(quota: float | int | str, budget: int, left: int) -> int:
    if quota == "rest":
        return left
    if isinstance(quota, float):
        if not 0 < quota < 1:
            raise ConfigError(f"A fractional quota must be between 0 and 1, got {quota}")
        return min(left, round(budget * quota))
    return min(left, int(quota))


def _run(
    strategy: StrategySpec, label: str, quota: int, free: list[int], keys: list[str],
    m: Matrix, query: pd.DataFrame, matched: set[str] | None, rng: np.random.Generator, seed: int,
) -> tuple[list[Pick], int | None]:
    everyone = list(range(len(keys)))
    match strategy.kind:
        case "random":
            order = rng.permutation(free)[:quota]
            return [_pick(keys, label, m, i, m.x.mean(axis=0), {}) for i in order], None
        case "diversity":
            return diverse(label, quota, free, keys, m, seed, population=everyone), None
        case "outlier":
            return _outliers(strategy, label, quota, free, keys, m, query)
        case "target":
            assert matched is not None
            return _target(strategy, label, quota, free, keys, m, query, matched, rng, seed), None
    raise AssertionError(strategy.kind)


def _pick(keys: list[str], label: str, m: Matrix, i: int, reference: np.ndarray, reason: dict[str, Any]) -> Pick:
    """A pick whose reason also names the columns where it stands furthest from `reference`, in scaled units."""
    deviation = m.x[i] - reference
    top = np.argsort(-np.abs(deviation))[:EXPLAINED_COLUMNS]
    columns = [{"column": m.columns[j], "value": None if np.isnan(m.raw[i, j]) else round(float(m.raw[i, j]), 4),
                "deviation": round(float(deviation[j]), 3)} for j in top]
    return Pick(key=keys[i], strategy=label, reason={**reason, "columns": columns})


def diverse(
    label: str, quota: int, free: list[int], keys: list[str], m: Matrix, seed: int,
    population: list[int], extra: dict[str, Any] | None = None,
) -> list[Pick]:
    """Cluster `population` into `quota` clusters and take free members nearest each centre.

    Clusters take turns from the largest down, so a cluster whose members are all taken passes its turn on.
    Each pick's columns compare it with the mean of `population`.
    """
    x = m.x
    centre = x[population].mean(axis=0)
    if len(population) <= quota:
        return [_pick(keys, label, m, i, centre, {**(extra or {})}) for i in free if i in set(population)]
    model = KMeans(n_clusters=quota, random_state=seed, n_init="auto").fit(x[population])
    free_set = set(free)
    queues: list[tuple[int, list[int]]] = []
    for cluster in range(quota):
        members = [population[j] for j in np.flatnonzero(model.labels_ == cluster)]
        candidates = [i for i in members if i in free_set]
        dist = np.linalg.norm(x[candidates] - model.cluster_centers_[cluster], axis=1)
        queues.append((len(members), [candidates[j] for j in np.argsort(dist)]))
    order = sorted(range(quota), key=lambda c: -queues[c][0])
    picks: list[Pick] = []
    while len(picks) < quota and any(queues[c][1] for c in order):
        for cluster in order:
            size, queue = queues[cluster]
            if not queue or len(picks) >= quota:
                continue
            picks.append(_pick(keys, label, m, queue.pop(0), centre, {
                **(extra or {}), "cluster": cluster, "cluster_size": size,
                "cluster_share": round(size / len(population), 4),
            }))
    return picks


def _outliers(
    strategy: StrategySpec, label: str, quota: int, free: list[int], keys: list[str], m: Matrix,
    query: pd.DataFrame,
) -> tuple[list[Pick], int | None]:
    """Score each trajectory by its mean distance to its nearest neighbours and take the highest scores.

    With `relative_to`, neighbours come from the trajectories sharing its value, and the score that ranks is the
    distance divided by the median distance in that group, so every group's own unusual trajectories can rank.
    Each pick's columns compare it with the mean of its neighbours.
    """
    if strategy.relative_to is None:
        groups: dict[Any, list[int]] = {None: list(range(len(keys)))}
        if len(keys) < 3:
            raise ConfigError(f"{label}: outlier scoring needs at least 3 trajectories")
    else:
        if strategy.relative_to not in query:
            raise ConfigError(f"{label}: relative_to names no column: {strategy.relative_to}")
        values = query[strategy.relative_to]
        groups = {}
        for i, value in enumerate(values):
            if not pd.isna(value):
                groups.setdefault(value, []).append(i)
    score = np.full(len(keys), np.nan)
    ranking = np.full(len(keys), np.nan)
    reference = np.zeros_like(m.x)
    details: dict[int, dict[str, Any]] = {}
    for value, members in groups.items():
        if len(members) < 3:
            continue
        rows = np.array(members)
        dist, idx = NearestNeighbors(n_neighbors=min(OUTLIER_NEIGHBOURS, len(rows) - 1)).fit(m.x[rows]).kneighbors()
        own = dist.mean(axis=1)
        score[rows] = own
        reference[rows] = m.x[rows[idx]].mean(axis=1)
        if strategy.relative_to is None:
            ranking[rows] = own
            continue
        typical = float(np.median(own)) or float(own.mean()) or 1.0
        ranking[rows] = own / typical
        for i in rows:
            details[int(i)] = {"relative_to": strategy.relative_to, "group": _plain(value), "group_size": len(rows),
                               "group_median_score": round(typical, 4), "relative_score": round(ranking[i], 4)}
    scored = np.flatnonzero(~np.isnan(ranking))
    ranks = {int(i): r + 1 for r, i in enumerate(scored[np.argsort(-ranking[scored])])}
    ranked = sorted((i for i in free if i in ranks), key=lambda i: ranks[i])[:quota]
    picks = [_pick(keys, label, m, i, reference[i], {
        "score": round(float(score[i]), 4), "rank": ranks[i], "of": len(ranks), **details.get(i, {}),
    }) for i in ranked]
    return picks, None if strategy.relative_to is None else len(keys) - len(ranks)


_THRESHOLD = re.compile(r"\{(\w+)\.(\w+)\}")


def expand_where(where: str, thresholds: dict[str, dict[str, float]]) -> str:
    """Replace every `{feature.threshold}` with the threshold value from the feature definition."""

    def replace(match: re.Match[str]) -> str:
        feature, name = match.groups()
        if name not in thresholds.get(feature, {}):
            raise ConfigError(f"No threshold {name} on feature {feature}")
        return repr(thresholds[feature][name])

    return _THRESHOLD.sub(replace, where)


def feature_thresholds(groups: list[Group]) -> dict[str, dict[str, float]]:
    return {f.name: f.thresholds for g in groups for f in g.features}


def select(query: pd.DataFrame, where: str, thresholds: dict[str, dict[str, float]]) -> pd.DataFrame:
    """The rows matching a `where` condition, with `{feature.threshold}` expanded."""
    return query.query(expand_where(where, thresholds), engine="python")


def _matches(
    strategy: StrategySpec, label: str, query: pd.DataFrame, thresholds: dict[str, dict[str, float]]
) -> set[str]:
    if not strategy.where:
        raise ConfigError(f"{label}: a target strategy needs where")
    return set(select(query, strategy.where, thresholds).index)


def _target(
    strategy: StrategySpec, label: str, quota: int, free: list[int], keys: list[str],
    m: Matrix, query: pd.DataFrame, matched: set[str], rng: np.random.Generator, seed: int,
) -> list[Pick]:
    position = {k: i for i, k in enumerate(keys)}
    subset = sorted(position[k] for k in matched)
    candidates = [i for i in free if keys[i] in matched]
    extra = {"where": strategy.where, "matched": len(matched)}
    if strategy.within == "diversity":
        return diverse(label, quota, candidates, keys, m, seed, population=subset, extra=extra)
    centre = m.x[subset].mean(axis=0) if subset else m.x.mean(axis=0)
    if strategy.within == "random":
        order = rng.permutation(candidates)[:quota]
        return [_pick(keys, label, m, i, centre, extra) for i in order]
    direction, _, column = strategy.within.partition(":")
    if direction not in ("top", "bottom") or column not in query:
        raise ConfigError(f"{label}: within must be diversity, random, top:<column> "
                          f"or bottom:<column>")
    values = query[column]
    ranked = sorted(candidates, key=lambda i: values.iloc[i], reverse=direction == "top")
    return [_pick(keys, label, m, i, centre, {**extra, column: _plain(values.iloc[i])}) for i in ranked[:quota]]


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def enrich(project: Project, pick: Pick, frame: Frame) -> dict[str, Any]:
    row = frame.query.loc[pick.key]
    features = {c: _plain(v) for c, v in row.items() if not pd.isna(v)}
    return {**pick.model_dump(), "markdown": str(project.trajectory_path(pick.key, ".md")), "features": features}


def save(project: Project, spec: SamplerSpec, selection: dict[str, Any]) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = project.samples_dir / spec.name / stamp
    out.mkdir(parents=True)
    path = out / "selection.json"
    path.write_text(json.dumps({"sampler": spec.model_dump(), **selection}, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    return path
