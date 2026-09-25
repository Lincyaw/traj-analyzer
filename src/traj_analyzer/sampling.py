from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors

from traj_analyzer.files import read_yaml
from traj_analyzer.operators.catalog import Group
from traj_analyzer.project import ConfigError, Project
from traj_analyzer.vectorize import Frame


class StrategySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["target", "outlier", "diversity", "random"]
    quota: float | int | Literal["rest"]
    label: str | None = None
    where: str | None = None
    within: str = "diversity"
    method: Literal["isolation_forest", "knn"] = "isolation_forest"


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
    x = frame.matrix(weights)
    thresholds = {f.name: f.thresholds for g in groups for f in g.features}
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
        new = [] if quota == 0 else _run(strategy, label, quota, free, keys, x, frame.query, matched, rng,
                                         spec.seed)
        chosen.update(pick.key for pick in new)
        picks += new
        left -= len(new)
        reports.append(StrategyReport(label=label, kind=strategy.kind, quota=quota, picked=len(new),
                                      matched=None if matched is None else len(matched)))
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
    x: np.ndarray, query: pd.DataFrame, matched: set[str] | None, rng: np.random.Generator, seed: int,
) -> list[Pick]:
    match strategy.kind:
        case "random":
            order = rng.permutation(free)[:quota]
            return [Pick(key=keys[i], strategy=label, reason={}) for i in order]
        case "diversity":
            return diverse(label, quota, free, keys, x, seed, population=list(range(len(keys))))
        case "outlier":
            return _outliers(strategy, label, quota, free, keys, x, seed)
        case "target":
            assert matched is not None
            return _target(strategy, label, quota, free, keys, x, query, matched, rng, seed)
    raise AssertionError(strategy.kind)


def diverse(
    label: str, quota: int, free: list[int], keys: list[str], x: np.ndarray, seed: int,
    population: list[int], extra: dict[str, Any] | None = None,
) -> list[Pick]:
    """Cluster `population` into `quota` clusters and take free members nearest each centre.

    Clusters take turns from the largest down, so a cluster whose members are all taken passes its turn on.
    """
    if len(population) <= quota:
        return [Pick(key=keys[i], strategy=label, reason={**(extra or {})}) for i in free
                if i in set(population)]
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
            picks.append(Pick(key=keys[queue.pop(0)], strategy=label, reason={
                **(extra or {}), "cluster": cluster, "cluster_size": size,
                "cluster_share": round(size / len(population), 4),
            }))
    return picks


def _outliers(
    strategy: StrategySpec, label: str, quota: int, free: list[int], keys: list[str],
    x: np.ndarray, seed: int,
) -> list[Pick]:
    if len(keys) < 3:
        raise ConfigError(f"{label}: outlier scoring needs at least 3 trajectories")
    if strategy.method == "isolation_forest":
        score = -IsolationForest(random_state=seed).fit(x).score_samples(x)
    else:
        neighbours = min(10, len(keys) - 1)
        dist, _ = NearestNeighbors(n_neighbors=neighbours + 1).fit(x).kneighbors(x)
        score = dist[:, 1:].mean(axis=1)
    ranks = {int(i): r + 1 for r, i in enumerate(np.argsort(-score))}
    ranked = sorted(free, key=lambda i: ranks[i])[:quota]
    return [Pick(key=keys[i], strategy=label, reason={
        "method": strategy.method, "score": round(float(score[i]), 4), "rank": ranks[i],
        "of": len(keys),
    }) for i in ranked]


_THRESHOLD = re.compile(r"\{(\w+)\.(\w+)\}")


def expand_where(where: str, thresholds: dict[str, dict[str, float]]) -> str:
    """Replace every `{feature.threshold}` with the threshold value from the feature definition."""

    def replace(match: re.Match[str]) -> str:
        feature, name = match.groups()
        if name not in thresholds.get(feature, {}):
            raise ConfigError(f"No threshold {name} on feature {feature}")
        return repr(thresholds[feature][name])

    return _THRESHOLD.sub(replace, where)


def _matches(
    strategy: StrategySpec, label: str, query: pd.DataFrame, thresholds: dict[str, dict[str, float]]
) -> set[str]:
    if not strategy.where:
        raise ConfigError(f"{label}: a target strategy needs where")
    return set(query.query(expand_where(strategy.where, thresholds), engine="python").index)


def _target(
    strategy: StrategySpec, label: str, quota: int, free: list[int], keys: list[str],
    x: np.ndarray, query: pd.DataFrame, matched: set[str], rng: np.random.Generator, seed: int,
) -> list[Pick]:
    position = {k: i for i, k in enumerate(keys)}
    subset = sorted(position[k] for k in matched)
    candidates = [i for i in free if keys[i] in matched]
    extra = {"where": strategy.where, "matched": len(matched)}
    if strategy.within == "diversity":
        return diverse(label, quota, candidates, keys, x, seed, population=subset, extra=extra)
    if strategy.within == "random":
        order = rng.permutation(candidates)[:quota]
        return [Pick(key=keys[i], strategy=label, reason=extra) for i in order]
    direction, _, column = strategy.within.partition(":")
    if direction not in ("top", "bottom") or column not in query:
        raise ConfigError(f"{label}: within must be diversity, random, top:<column> "
                          f"or bottom:<column>")
    values = query[column]
    ranked = sorted(candidates, key=lambda i: values.iloc[i], reverse=direction == "top")
    return [Pick(key=keys[i], strategy=label, reason={**extra, column: _plain(values.iloc[i])})
            for i in ranked[:quota]]


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
