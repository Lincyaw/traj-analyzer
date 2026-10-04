from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

SEPARATOR = " | "
BELOW = "*"
"""Ends the label of a node that stands for every path below it."""

Node = tuple[str, ...]
Paths = list[list[str]]


@dataclass
class PathTree:
    """The paths of a population as a tree: a node is a prefix of a path, and its children refine it.

    `support` counts the trajectories with at least one event under a node, `events` the events under it.
    A node is frequent when its support reaches `minimum`; support never grows with depth, so the frequent nodes
    form a subtree, and every event belongs to its deepest frequent prefix, the node it is cut at.
    """

    levels: list[str]
    support: Counter[Node]
    events: Counter[Node]
    trajectories: int
    minimum: int

    @classmethod
    def build(cls, levels: list[str], values: Iterable[Paths | None], share: float) -> PathTree:
        """Count every prefix over the trajectories that have a value; `share` of them makes a node frequent."""
        support: Counter[Node] = Counter()
        events: Counter[Node] = Counter()
        trajectories = 0
        for paths in values:
            if paths is None:
                continue
            trajectories += 1
            seen: set[Node] = set()
            for path in paths:
                for depth in range(len(levels) + 1):
                    prefix = tuple(path[:depth])
                    events[prefix] += 1
                    seen.add(prefix)
            support.update(seen)
        return cls(levels=levels, support=support, events=events, trajectories=trajectories,
                   minimum=max(1, math.ceil(share * trajectories)))

    def cut(self, path: list[str]) -> Node:
        """The deepest frequent prefix of a path; the root when not even its first level is frequent."""
        depth = 0
        while depth < len(self.levels) and self.support[tuple(path[:depth + 1])] >= self.minimum:
            depth += 1
        return tuple(path[:depth])

    def label(self, node: Node) -> str:
        """A node as text; a node above the last level ends with `*`, since it stands for the paths below it."""
        return SEPARATOR.join([*node, BELOW] if len(node) < len(self.levels) else node)

    def shares(self, paths: Paths) -> dict[str, float]:
        """Share of a trajectory's events cut at each node, by node label."""
        counts = Counter(self.label(self.cut(path)) for path in paths)
        return {label: count / len(paths) for label, count in counts.items()}

    def share_frame(self, values: pd.Series) -> pd.DataFrame:
        """One row per trajectory and one column per node of the cut, most used first; rows without a value are
        empty, and a trajectory without events has zeros."""
        has_value = values.map(lambda v: isinstance(v, list)).to_numpy()
        shares = pd.DataFrame([self.shares(v) if isinstance(v, list) and v else {} for v in values],
                              index=values.index)
        shares = shares.fillna(0.0)
        shares.loc[~has_value] = np.nan
        return shares[shares.sum().sort_values(ascending=False, kind="stable").index]

    def rare(self, paths: Paths) -> float | None:
        """Share of a trajectory's events cut above the last level."""
        if not paths:
            return None
        return sum(len(self.cut(path)) < len(self.levels) for path in paths) / len(paths)

    def level_summary(self) -> list[dict[str, Any]]:
        """Per level: how many nodes it has, how many are frequent, and the bits it adds to describing an event."""
        total = self.events[()]
        summary = []
        previous = 0.0
        for depth, level in enumerate(self.levels, start=1):
            nodes = [node for node in self.events if len(node) == depth]
            counts = np.array([self.events[node] for node in nodes], dtype=float)
            entropy = float(-(counts / total * np.log2(counts / total)).sum()) if total else 0.0
            summary.append({
                "level": level, "nodes": len(nodes),
                "frequent": sum(self.support[node] >= self.minimum for node in nodes),
                "bits": round(entropy - previous, 3),
            })
            previous = entropy
        return summary

    def cut_summary(self, values: Iterable[Paths | None]) -> dict[str, Any]:
        """Where the events of these trajectories are cut: the share per depth and the number of nodes used."""
        depths: Counter[int] = Counter()
        nodes: set[Node] = set()
        for paths in values:
            for path in paths or []:
                node = self.cut(path)
                depths[len(node)] += 1
                nodes.add(node)
        total = sum(depths.values())
        names = ["(none)", *self.levels]
        return {"nodes": len(nodes),
                "events_by_depth": {names[depth]: round(depths[depth] / total, 4) if total else 0.0
                                    for depth in range(len(names))}}


def level_surprisal(levels: list[str], values: pd.Series, inside: np.ndarray) -> pd.DataFrame:
    """Per trajectory and level, the mean surprisal of its events at that level given the level above.

    An event at a node costs `-ln((c + 1) / (p + k + 1))`, where the reference events hold `c` under the node and
    `p` under its parent, which has `k` children there. Below a node the reference set does not hold, the levels
    cost nothing, so a new path is charged to the first level where it is new.
    """
    events: Counter[Node] = Counter()
    children: Counter[Node] = Counter()
    for paths in values[inside]:
        for path in paths or []:
            for depth in range(len(levels) + 1):
                prefix = tuple(path[:depth])
                if not events[prefix] and depth:
                    children[prefix[:-1]] += 1
                events[prefix] += 1
    rows = np.full((len(values), len(levels)), np.nan)
    for row, paths in enumerate(values):
        if not paths:
            continue
        total = np.zeros(len(levels))
        for path in paths:
            for depth in range(1, len(levels) + 1):
                parent = tuple(path[:depth - 1])
                seen = events.get(tuple(path[:depth]), 0)
                total[depth - 1] += -math.log((seen + 1) / (events.get(parent, 0) + children.get(parent, 0) + 1))
        rows[row] = total / len(paths)
    return pd.DataFrame(rows, index=values.index, columns=levels)


def describe(
    tree: PathTree, values: pd.Series, by: pd.Series | None = None, within: pd.Series | None = None, top: int = 40
) -> dict[str, Any]:
    """The tree of one paths feature: its levels, where events are cut, and the most used nodes of the cut.

    With `by`, every node and level also tells how much of the variation of a node's share across trajectories the
    `by` groups explain, beyond the `within` groups when given.
    """
    shares = tree.share_frame(values)
    usable = shares.notna().any(axis=1) if len(shares.columns) else pd.Series(False, index=shares.index)
    if by is not None:
        usable &= by.notna()
    if within is not None:
        usable &= within.notna()
    groups = None if by is None else by[usable].to_numpy()
    inner = None if within is None else within[usable].to_numpy()
    cut_events: Counter[Node] = Counter()
    for paths in values:
        for path in paths or []:
            cut_events[tree.cut(path)] += 1
    labels = {tree.label(node): node for node in cut_events}
    nodes = []
    for label in shares.columns:
        node = labels[label]
        row: dict[str, Any] = {"node": label, "depth": len(node), "trajectories": tree.support[node],
                               "events": cut_events[node]}
        if groups is not None:
            share = explained(shares.loc[usable, label].to_numpy(), groups, inner)
            row["explained"] = None if share is None else round(share, 4)
        nodes.append(row)
    result: dict[str, Any] = {
        "trajectories": tree.trajectories, "events": tree.events[()], "minimum_support": tree.minimum,
        "levels": tree.level_summary(), "cut": tree.cut_summary(values),
    }
    if groups is not None:
        names = ["(none)", *tree.levels]
        for depth, level in enumerate(result["levels"], start=1):
            found = [n["explained"] for n in nodes if n["depth"] == depth and n["explained"] is not None]
            level["explained_median"] = round(float(np.median(found)), 4) if found else None
        result["explained_by"] = {"by": by.name, "within": None if within is None else within.name,
                                  "levels": names[1:]}
    result["nodes"] = sorted(nodes, key=lambda n: -n["events"])[:top]
    return result


def explained(values: np.ndarray, by: np.ndarray, within: np.ndarray | None = None) -> float | None:
    """Share of the variance of `values` that the means of the `by` groups explain.

    With `within`, the means of the `within` groups are removed first, so the share is what `by` explains of the
    whole variance beyond `within`. None when the values do not vary.
    """
    total = float(values.var())
    if total == 0:
        return None
    centred = pd.Series(values)
    if within is not None:
        centred = centred - centred.groupby(within).transform("mean")
    return float(centred.groupby(by).transform("mean").var(ddof=0) / total)
