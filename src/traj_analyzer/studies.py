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
from traj_analyzer.sampling import expand_where
from traj_analyzer.vectorize import Frame


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
    min_present: int = 100
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


def load_study(project: Project, name: str) -> StudySpec:
    path = project.studies_dir / f"{name}.yaml"
    if not path.is_file():
        raise ConfigError(f"No study {name} at {path}")
    return StudySpec.model_validate({"name": name, **read_yaml(path)})


def list_studies(project: Project) -> list[str]:
    return sorted(p.stem for p in project.studies_dir.glob("*.yaml"))


def outcome_features(study: StudySpec, groups: list[Group]) -> set[str]:
    features = set()
    for group in groups:
        for instance in group.instances:
            for feature in instance.features:
                if any(fnmatch.fnmatchcase(feature.name, p) or p == instance.id for p in study.outcomes):
                    features.add(feature.name)
    return features


@dataclass
class Population:
    frame: Frame
    success: pd.Series


def population(study: StudySpec, frame: Frame, groups: list[Group]) -> Population:
    """The trajectories the study is about, and whether each one succeeded."""
    thresholds = {f.name: f.thresholds for g in groups for f in g.features}
    query = frame.query
    for feature in study.by + ([study.pairs.within] if study.pairs else []):
        if feature not in query.columns:
            raise ConfigError(f"Study {study.name}: {feature} is not a feature with one column, such as a category")
    if study.where:
        query = query.query(expand_where(study.where, thresholds), engine="python")
    success = query.eval(expand_where(study.target, thresholds), engine="python")
    if not isinstance(success, pd.Series):
        raise ConfigError(f"Study {study.name}: target {study.target!r} does not give one value per trajectory")
    kept = Frame(query=query, distance=frame.distance.loc[query.index], columns=frame.columns,
                 extracted=frame.extracted)
    return Population(frame=kept, success=success.astype("boolean"))


def numeric_columns(frame: Frame, feature: str) -> list[str]:
    """Numeric columns of a feature: its query columns, or its one-hot columns when the feature is a category."""
    if feature in frame.query.columns and not pd.api.types.is_numeric_dtype(frame.query[feature]):
        return frame.columns[feature]
    return [c for c in frame.query.columns if c == feature or c.startswith(f"{feature}__")]


def numeric_view(frame: Frame) -> pd.DataFrame:
    """Every feature's numeric columns side by side, as floats."""
    parts = []
    for feature in frame.columns:
        cols = numeric_columns(frame, feature)
        source = frame.query if cols and cols[0] in frame.query.columns else frame.distance
        parts.append(source[cols])
    return pd.concat(parts, axis=1).astype(float)


def _demean(frame: pd.DataFrame, by: list[pd.Series], rounds: int = 5) -> pd.DataFrame:
    for _ in range(rounds):
        for groups in by:
            frame = frame - frame.groupby(groups).transform("mean")
    return frame


def _corr(a: pd.Series, b: pd.Series) -> float:
    ok = a.notna() & b.notna()
    if ok.sum() < 3 or a[ok].std() == 0 or b[ok].std() == 0:
        return math.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _share_explained(values: pd.Series, groups: pd.Series) -> tuple[float, float]:
    """Variance share explained by group means, and the share random groups of the same count would explain."""
    ok = values.notna() & groups.notna()
    v, g = values[ok], groups[ok]
    if len(v) < 3 or v.var() == 0:
        return math.nan, math.nan
    return float(v.groupby(g).transform("mean").var() / v.var()), (g.nunique() - 1) / (len(v) - 1)


def _round(value: float) -> float | None:
    return None if math.isnan(value) else round(value, 3)


def screen(study: StudySpec, data: Population, features: list[str], outcomes: set[str]) -> dict[str, Any]:
    """Check each feature's columns on the study population, all on percentile ranks.

    A column is constant when one value covers more than `max_mode_share` of the trajectories that have it, and
    redundant when its rank correlation with a column of another feature reaches `max_similarity`; that feature is
    one not being screened or one screened earlier in `features`. It varies by a
    `by` feature when the group means explain at least `min_group_excess` more of its variance than random groups
    would, and it explains the target when its correlation with success, after removing the mean of every `by`
    group, reaches `min_within_r`.
    """
    frame, rules = data.frame, study.rules
    unknown = sorted(set(features) - set(frame.columns))
    if unknown:
        raise ConfigError(f"Features {unknown} have no extracted values")
    numeric = numeric_view(frame)
    owner = {c: f for f in frame.columns for c in numeric_columns(frame, f)}
    columns = {f: numeric_columns(frame, f) for f in features}
    pool = [c for c in numeric.columns if owner[c] not in outcomes and owner[c] not in study.by]
    ranks = numeric[pool].rank(pct=True)
    values = numeric[[c for cols in columns.values() for c in cols]]
    value_ranks = values.rank(pct=True)
    success = data.success.astype(float).rank(pct=True)
    groups = [frame.query[b].astype("string") for b in study.by]
    within = _demean(value_ranks.assign(__success=success), groups)
    varying = value_ranks.std() > 0
    comparable_ranks = ranks[[c for c in ranks.columns if ranks[c].std() > 0]]
    order = {f: i for i, f in enumerate(features)}
    report: dict[str, Any] = {}
    for feature, cols in columns.items():
        # Of two screened features that duplicate each other, only the later one is redundant.
        compared = [c for c in comparable_ranks.columns
                    if owner[c] != feature and order.get(owner[c], -1) < order[feature]]
        entries = {}
        for column in cols:
            present = values[column].dropna()
            mode_share = float(present.value_counts(normalize=True).iloc[0]) if len(present) else math.nan
            similarity = pd.Series(dtype=float)
            if len(present) < rules.min_present:
                entries[column] = {"present": len(present), "too_few": True}
                continue
            if varying[column]:
                others = comparable_ranks[compared]
                # Columns without variance where both have values give NaN, which dropna removes.
                with np.errstate(invalid="ignore", divide="ignore"):
                    similarity = others.corrwith(value_ranks[column]).abs().dropna().sort_values(ascending=False)
            by = {}
            for name, g in zip(study.by, groups, strict=True):
                share, null = _share_explained(value_ranks[column], g)
                by[name] = {"share": _round(share), "random": _round(null), "excess": _round(share - null)}
            r_within = _corr(within[column], within["__success"])
            entry = {
                "present": len(present),
                "too_few": False,
                "missing": _round(1 - len(present) / len(values)),
                "mode_share": _round(mode_share),
                "most_similar": [similarity.index[0], _round(similarity.iloc[0])] if len(similarity) else None,
                "by": by,
                "r_success": _round(_corr(value_ranks[column], success)),
                "r_success_within": _round(r_within),
            }
            entry["constant"] = bool(not len(present) or mode_share > rules.max_mode_share)
            entry["redundant"] = bool(len(similarity) and similarity.iloc[0] >= rules.max_similarity)
            entry["varies_by"] = [n for n, e in by.items()
                                  if e["excess"] is not None and e["excess"] >= rules.min_group_excess]
            entry["explains_success"] = bool(not math.isnan(r_within) and abs(r_within) >= rules.min_within_r)
            entries[column] = entry
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
    query, success = data.frame.query, data.success
    for feature in [spec.spread, spec.distinct]:
        if feature is not None and feature not in query.columns:
            raise ConfigError(f"Pairs: {feature} is not a feature with one column")
    rng = np.random.default_rng(seed)
    queues: dict[Any, list[pd.DataFrame]] = {}
    grouped = [rows for _, rows in query.groupby(spec.within, sort=True)]
    for i in rng.permutation(len(grouped)):
        rows = grouped[i]
        if success[rows.index].fillna(False).any() and (~success[rows.index].fillna(True)).any():
            queues.setdefault(rows[spec.spread].iloc[0] if spec.spread else None, []).append(rows)
    used: set[Any] = set()
    pairs: list[dict[str, Any]] = []
    while len(pairs) < spec.n and any(queues.values()):
        for key, queue in queues.items():
            if len(pairs) == spec.n or not queue:
                continue
            rows = queue.pop(0)
            pair = _pair(rows, success, spec.distinct, used, rng)
            if pair is None:
                continue
            succeeded, failed = pair
            if spec.distinct:
                used.update({rows.at[succeeded, spec.distinct], rows.at[failed, spec.distinct]})
            pairs.append({spec.within: rows[spec.within].iloc[0], **({spec.spread: key} if spec.spread else {}),
                          "succeeded": succeeded, "failed": failed})
    return pairs


def _pair(rows: pd.DataFrame, success: pd.Series, distinct: str | None, used: set[Any],
          rng: np.random.Generator) -> tuple[str, str] | None:
    def free(keys: pd.Index, taken: set[Any]) -> list[str]:
        return [k for k in keys if distinct is None or rows.at[k, distinct] not in taken]

    good = free(rows.index[success[rows.index].fillna(False).to_numpy(dtype=bool)], used)
    bad = free(rows.index[(~success[rows.index].fillna(True)).to_numpy(dtype=bool)], used)
    if not good or not bad:
        return None
    return good[rng.integers(len(good))], bad[rng.integers(len(bad))]
