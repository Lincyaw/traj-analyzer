from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from sklearn.cluster import KMeans

from traj_analyzer.hunting.functions import Hit
from traj_analyzer.hunting.ledger import HuntState, Source
from traj_analyzer.hunting.spec import HuntSpec
from traj_analyzer.sampling import diverse
from traj_analyzer.vectorize import Matrix


class Draw(BaseModel):
    """One trajectory a round reads: why it was drawn, and where the active probes hit it."""

    model_config = ConfigDict(frozen=True)

    key: str
    source: Source
    hits: list[Hit]


def clusters_of(keys: list[str], matrix: Matrix, count: int, seed: int) -> dict[str, int]:
    """The cluster of every trajectory under the extracted features, which the coverage reads go through."""
    model = KMeans(n_clusters=min(count, len(keys)), random_state=seed, n_init="auto").fit(matrix.x)
    return {key: int(label) for key, label in zip(keys, model.labels_, strict=True)}


def hits_of(hits: pd.DataFrame, key: str) -> list[Hit]:
    rows = hits[hits["key"] == key].sort_values(["step", "probe"])
    return [Hit(probe=row.probe, step=int(row.step), note=row.note) for row in rows.itertuples()]


def apportion(total: int, weights: dict[str, int]) -> dict[str, int]:
    """`total` shared out in proportion to `weights` by the largest remainders, none above its weight."""
    available = sum(weights.values())
    total = min(total, available)
    if total <= 0:
        return {}
    exact = {name: total * weight / available for name, weight in weights.items()}
    shares = {name: int(value) for name, value in exact.items()}
    for name in sorted(exact, key=lambda n: (-(exact[n] - shares[n]), n))[:total - sum(shares.values())]:
        shares[name] += 1
    return shares


def draw(
    spec: HuntSpec, state: HuntState, hits: pd.DataFrame, keys: list[str], matrix: Matrix, clusters: dict[str, int],
    round_: int,
) -> list[Draw]:
    """The trajectories a round reads, none of them read before.

    The hit reads give every active probe with unread hits `min_reads_per_probe` reads first, the probes with the
    fewest read hits first, and share the rest in proportion to the unread hits; within a probe they take
    trajectories of different kinds. The other reads go to trajectories no probe hits: half to the clusters read
    least, half at random.
    """
    rng = np.random.default_rng([spec.seed, round_])
    position = {key: i for i, key in enumerate(keys)}
    read = set(state.readings)
    budget = round(spec.reads_per_round * spec.hit_share)
    hit_keys = {str(probe): sorted(set(group["key"])) for probe, group in hits.groupby("probe")}
    unread = {probe: [key for key in found if key not in read] for probe, found in hit_keys.items()}
    chosen: list[Draw] = []
    taken: set[str] = set()

    def free(probe: str) -> list[int]:
        return [position[key] for key in unread[probe] if key not in taken]

    def take(probe: str, count: int) -> None:
        candidates = free(probe)
        if count <= 0 or not candidates:
            return
        for pick in diverse(probe, count, candidates, keys, matrix, spec.seed + round_, population=candidates):
            taken.add(pick.key)
            chosen.append(Draw(key=pick.key, source="hit", hits=hits_of(hits, pick.key)))

    order = sorted(unread, key=lambda probe: (len(hit_keys[probe]) - len(unread[probe]), probe))
    for probe in order:
        take(probe, min(spec.min_reads_per_probe, budget - len(chosen)))
    shares = apportion(budget - len(chosen), {probe: len(free(probe)) for probe in order})
    for probe in order:
        take(probe, shares.get(probe, 0))

    hit_reads = len(chosen)
    explore = spec.reads_per_round - hit_reads
    hit = set(hits["key"])
    members: dict[int, list[str]] = {}
    sizes: dict[int, int] = {}
    reads: dict[int, int] = {}
    for key in keys:
        cluster = clusters[key]
        sizes[cluster] = sizes.get(cluster, 0) + 1
        reads[cluster] = reads.get(cluster, 0) + (key in read)
        if key not in hit and key not in read:
            members.setdefault(cluster, []).append(key)
    for _ in range((explore + 1) // 2):
        open_clusters = [cluster for cluster, found in members.items() if found]
        if not open_clusters:
            break
        cluster = min(open_clusters, key=lambda c: (reads[c] / sizes[c], c))
        key = members[cluster].pop(int(rng.integers(len(members[cluster]))))
        reads[cluster] += 1
        chosen.append(Draw(key=key, source="coverage", hits=[]))
    pool = sorted(key for found in members.values() for key in found)
    count = min(explore - (len(chosen) - hit_reads), len(pool))
    if count > 0:
        for key in sorted(rng.choice(pool, size=count, replace=False).tolist()):
            chosen.append(Draw(key=key, source="random", hits=[]))
    return chosen
