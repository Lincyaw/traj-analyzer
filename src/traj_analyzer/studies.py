from __future__ import annotations

import fnmatch
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from traj_analyzer.files import read_yaml
from traj_analyzer.operators.catalog import Group
from traj_analyzer.project import ConfigError, Project
from traj_analyzer.sampling import evaluate, feature_thresholds, select
from traj_analyzer.vectorize import Frame

GROUPING_TYPES = ("scalar", "boolean", "category")
"""Feature types with one value to group by."""


class PairSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    within: str
    """Feature whose value both trajectories of a pair share, such as a task or case id."""
    spread: str | None = None
    """Feature whose values the pairs take turns over, such as a system or task family."""
    distinct: str | None = None
    """Feature whose values appear in at most one pair, such as the model."""
    n: int = Field(default=5, ge=1)


class ScreenRules(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_mode_share: float = 0.95
    max_similarity: float = 0.8
    min_group_excess: float = 0.05
    min_within_r: float = 0.05
    min_present: int = Field(default=100, ge=1)
    """Columns with values for fewer trajectories get the verdict `too_few` and no other checks."""


class StudySpec(BaseModel):
    """One question about the trajectories: which of them succeed, and which features tell why."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    datasets: list[str] | None = None
    where: str | None = None
    """Condition selecting the trajectories the question is about."""
    target: str
    """Expression that is true for the trajectories that succeed."""
    by: list[str] = []
    """Features whose influence the screen removes, such as the model and the task."""
    outcomes: list[str] = []
    """Features, operator instance ids or glob patterns that record outcomes; they are never screened."""
    pairs: PairSpec | None = None
    rules: ScreenRules = ScreenRules()

    @property
    def grouping_features(self) -> list[str]:
        named = [self.pairs.within, self.pairs.spread, self.pairs.distinct] if self.pairs else []
        return [*self.by, *(f for f in named if f is not None)]


def load_study(project: Project, name: str) -> StudySpec:
    path = project.studies_dir / f"{name}.yaml"
    if not path.is_file():
        raise ConfigError(f"No study {name} at {path}")
    return StudySpec.model_validate({"name": name, **read_yaml(path)})


def list_studies(project: Project) -> list[str]:
    return sorted(p.stem for p in project.studies_dir.glob("*.yaml"))


def check_study(study: StudySpec, groups: list[Group]) -> set[str]:
    """Check the features a study names against the enabled ones, and return its outcome features."""
    specs = {f.name: f for g in groups for f in g.features}
    for feature in study.grouping_features:
        if feature not in specs:
            raise ConfigError(f"Study {study.name}: {feature} is not an enabled feature")
        if specs[feature].type not in GROUPING_TYPES:
            raise ConfigError(f"Study {study.name}: {feature} is a {specs[feature].type}; grouping needs one of "
                              f"{GROUPING_TYPES}")
    outcomes = set()
    for pattern in study.outcomes:
        matched = {f.name for g in groups for i in g.instances for f in i.features
                   if fnmatch.fnmatchcase(f.name, pattern) or pattern == i.id}
        if not matched:
            raise ConfigError(f"Study {study.name}: outcome {pattern!r} matches no enabled feature or operator")
        outcomes |= matched
    return outcomes


@dataclass
class Population:
    frame: Frame
    success: pd.Series


def population(study: StudySpec, frame: Frame, groups: list[Group]) -> Population:
    """The trajectories the study is about, and whether each one succeeded."""
    thresholds = feature_thresholds(groups)
    query = select(frame.query, study.where, thresholds) if study.where else frame.query
    success = evaluate(query, study.target, thresholds)
    return Population(frame=frame.rows(query.index), success=success.astype("boolean"))


def _demean(frame: pd.DataFrame, by: list[pd.Series], rounds: int = 5) -> pd.DataFrame:
    for _ in range(rounds):
        for groups in by:
            frame = frame - frame.groupby(groups).transform("mean")
    return frame


def _round(value: float) -> float | None:
    return None if value is None or math.isnan(value) else round(float(value), 3)


def screen(study: StudySpec, data: Population, features: list[str], outcomes: set[str]) -> dict[str, Any]:
    """Check each feature's numeric columns on the study population, all on percentile ranks.

    A column is constant when one value covers more than `max_mode_share` of the trajectories that have it, and
    redundant when its rank correlation with a column of another feature reaches `max_similarity`; that feature is
    one not being screened or one screened earlier in `features`. It varies by a `by` feature when the group means
    explain at least `min_group_excess` more of its variance than random groups of the same count would, and it
    explains success when its correlation with success, after removing the mean of every `by` group, reaches
    `min_within_r`.
    """
    frame, rules = data.frame, study.rules
    unknown = sorted(set(features) - set(frame.numeric_columns))
    if unknown:
        raise ConfigError(f"Features {unknown} have no extracted values")
    owner = {c: f for f, cols in frame.numeric_columns.items() for c in cols}
    order = {f: i for i, f in enumerate(features)}
    enough = frame.numeric.count() >= rules.min_present
    screened = [c for f in features for c in frame.numeric_columns[f] if enough[c]]
    pool = [c for c, f in owner.items() if f not in outcomes and f not in study.by and enough[c]]
    ranks = frame.numeric.rank(pct=True)
    varying = [c for c in dict.fromkeys(pool + screened) if ranks[c].std() > 0]
    correlation = ranks[varying].corr().abs()
    success = data.success.astype(float).rank(pct=True)
    groups = [frame.query[b].astype("string") for b in study.by]
    within = _demean(ranks[screened].assign(__success=success), groups)
    # A column without variance, before or after removing group means, has no correlation; it stays NaN.
    with np.errstate(invalid="ignore", divide="ignore"):
        r_within = within[screened].corrwith(within["__success"])
        r_success = ranks[screened].corrwith(success)
    shares = {}
    for name, g in zip(study.by, groups, strict=True):
        values = ranks[screened][g.notna()]
        share = values.groupby(g).transform("mean").var() / values.var()
        n_groups = values.notna().groupby(g[g.notna()]).any().sum()
        shares[name] = (share, (n_groups - 1) / (values.notna().sum() - 1))
    report: dict[str, Any] = {}
    for feature in features:
        entries = {}
        for column in frame.numeric_columns[feature]:
            present = frame.numeric[column].dropna()
            if not enough[column]:
                entries[column] = {"present": len(present), "too_few": True}
                continue
            mode_share = float(present.value_counts(normalize=True).iloc[0])
            similar = pd.Series(dtype=float)
            if column in correlation.index:
                compared = [c for c in varying if owner[c] != feature and order.get(owner[c], -1) < order[feature]]
                similar = correlation.loc[column, compared].dropna().sort_values(ascending=False)
            by = {}
            for name, (share, null) in shares.items():
                by[name] = {"share": _round(share[column]), "random": _round(null[column]),
                            "excess": _round(share[column] - null[column])}
            within_r = float(r_within[column])
            entries[column] = {
                "present": len(present),
                "too_few": False,
                "missing": _round(1 - len(present) / len(frame.numeric)),
                "mode_share": _round(mode_share),
                "most_similar": [similar.index[0], _round(similar.iloc[0])] if len(similar) else None,
                "by": by,
                "r_success": _round(r_success[column]),
                "r_success_within": _round(within_r),
                "constant": mode_share > rules.max_mode_share,
                "redundant": bool(len(similar) and similar.iloc[0] >= rules.max_similarity),
                "varies_by": [n for n, e in by.items()
                              if e["excess"] is not None and e["excess"] >= rules.min_group_excess],
                "explains_success": not math.isnan(within_r) and abs(within_r) >= rules.min_within_r,
            }
        report[feature] = {"verdict": _verdict(entries.values()), "columns": entries}
    return report


def _verdict(entries: Any) -> str:
    """`explains_success`, `varies_by`, `uninformative`, `redundant`, `constant` or `too_few`, from the best column."""
    entries = [e for e in entries if not e["too_few"]]
    if not entries:
        return "too_few"
    usable = [e for e in entries if not e["constant"] and not e["redundant"]]
    if not usable:
        return "constant" if all(e["constant"] for e in entries) else "redundant"
    if any(e["explains_success"] for e in usable):
        return "explains_success"
    if any(e["varies_by"] for e in usable):
        return "varies_by"
    return "uninformative"


def contrast_pairs(spec: PairSpec, data: Population, seed: int) -> list[dict[str, Any]]:
    """Pairs sharing the value of `within`, one trajectory that succeeded and one that did not.

    Groups come in a seeded random order, taking turns over the values of `spread`.
    """
    query = data.frame.query
    won = data.success.fillna(False).astype(bool)
    lost = (~data.success.fillna(True)).astype(bool)
    rng = np.random.default_rng(seed)
    queues: dict[Any, list[pd.DataFrame]] = {}
    grouped = [rows for _, rows in query.groupby(spec.within, sort=True)]
    for i in rng.permutation(len(grouped)):
        rows = grouped[i]
        if won[rows.index].any() and lost[rows.index].any():
            queues.setdefault(rows[spec.spread].iloc[0] if spec.spread else None, []).append(rows)
    used: set[Any] = set()
    pairs: list[dict[str, Any]] = []
    while len(pairs) < spec.n and any(queues.values()):
        for key, queue in queues.items():
            if len(pairs) == spec.n or not queue:
                continue
            rows = queue.pop(0)

            def free(mask: pd.Series, rows: pd.DataFrame = rows) -> list[str]:
                return [k for k in rows.index[mask[rows.index].to_numpy()]
                        if spec.distinct is None or rows.at[k, spec.distinct] not in used]

            good, bad = free(won), free(lost)
            if not good or not bad:
                continue
            succeeded, failed = good[rng.integers(len(good))], bad[rng.integers(len(bad))]
            if spec.distinct:
                used.update({rows.at[succeeded, spec.distinct], rows.at[failed, spec.distinct]})
            pairs.append({spec.within: rows[spec.within].iloc[0], **({spec.spread: key} if spec.spread else {}),
                          "succeeded": succeeded, "failed": failed})
    return pairs
