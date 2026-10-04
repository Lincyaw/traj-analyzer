from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
from pydantic import BaseModel, Field
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.tree import DecisionTreeRegressor

from traj_analyzer.extract import WorkerRunner, run_groups
from traj_analyzer.features.table import KEY, feature_values, read_group, status_column
from traj_analyzer.ingest import IndexRow, load_index
from traj_analyzer.operators.base import FeatureSpec
from traj_analyzer.operators.catalog import Group, load_groups
from traj_analyzer.paths import level_surprisal
from traj_analyzer.project import ConfigError, Project
from traj_analyzer.runtime import model_name
from traj_analyzer.vectorize import LABEL_TYPES, Frame, build_frame, missing_column

FOLDS = 5
MIN_ROWS = 10 * FOLDS
"""Fewest trajectories with values for both features of a pair; below it no dependence is established."""
NO_VARIANCE = 1e-12
"""Variance below which a scaled target counts as constant, so floating-point error is not taken for variation."""
FILE = "features.json"

Role = Literal["distinguishing", "invariant", "dependent", "context"]
CONTEXT_OPERATOR = "meta.fields"
"""Features of this operator are copied metadata: conditions to group and compare by, never judged themselves."""
Decision = Literal["keep", "revise", "labels"]


class Relation(BaseModel):
    source: str
    """The distinguishing feature that determines this one."""
    r: float


class Card(BaseModel):
    feature: str
    group: str
    kind: Literal["code", "llm"]
    type: str
    noise: float | None = None
    """Mean disagreement between two measurements; None when no trajectory has both."""
    compared: int = 0
    coverage: int = 0
    """Reference trajectories with a value for the feature."""
    distinct: int
    role: Role
    relation: Relation | None = None
    decision: Decision
    drift: float | None = None
    """Jensen-Shannon divergence between the reference and the monitored trajectories; None without the latter."""


class Evaluation(BaseModel):
    reference: int
    monitored: int
    evaluation_set: list[str]
    """The draw from the whole reference set, on which mining measures its candidates."""
    measured: dict[str, list[str]] = Field(default_factory=dict)
    """Per LLM group, the reference trajectories measured a second time: the same draw among those it has rows for."""
    features: list[Card]

    def held_out(self) -> set[str]:
        """Every trajectory a measurement of noise rests on."""
        return set(self.evaluation_set).union(*self.measured.values())

    def roles(self) -> dict[str, Role]:
        return {card.feature: card.role for card in self.features}

    def context(self) -> frozenset[str]:
        return frozenset(card.feature for card in self.features if card.role == "context")

    def relations(self) -> dict[str, str]:
        return {card.feature: card.relation.source for card in self.features if card.relation is not None}


def load_evaluation(project: Project) -> Evaluation:
    path = project.evaluation_dir / FILE
    if not path.is_file():
        raise ConfigError("No feature evaluation found; run `traj evaluate`")
    return Evaluation.model_validate_json(path.read_bytes())


def reference_keys(project: Project, rows: list[IndexRow]) -> list[str]:
    """Keys of the reference set: the trajectories of `reference.datasets` without `reference.exclude`."""
    config = project.config.reference
    datasets = set(project.config.datasets) if config.datasets == "all" else set(config.datasets)
    unknown = sorted(datasets - set(project.config.datasets))
    if unknown:
        raise ConfigError(f"reference.datasets names datasets that are not configured: {unknown}")
    keys = {row.key for row in rows}
    missing = sorted(set(config.exclude) - keys)
    if missing:
        raise ConfigError(f"reference.exclude names unknown trajectory keys: {missing}")
    excluded = set(config.exclude)
    return [row.key for row in rows if row.dataset in datasets and row.key not in excluded]


def evaluation_set(project: Project, reference: list[str]) -> list[str]:
    """The reference trajectories measured a second time, drawn with `evaluation.seed`."""
    config = project.config.evaluation
    rng = np.random.default_rng(config.seed)
    ordered = sorted(reference)
    return sorted(rng.choice(ordered, size=min(config.size, len(ordered)), replace=False).tolist())


def disagreement(feature: FeatureSpec, first: Any, second: Any) -> float:
    """How far two measurements of one feature on one trajectory are apart, from 0 to 1."""
    match feature.type:
        case "boolean" | "category":
            return float(first != second)
        case "set":
            union = set(first) | set(second)
            return 1 - len(set(first) & set(second)) / len(union) if union else 0.0
        case "scalar":
            return _numeric([first], [second], feature.range)
        case "vector":
            return 1.0 if len(first) != len(second) else _numeric(first, second, feature.range)
        case "map":
            keys = sorted(set(first) | set(second))
            return _numeric([first.get(k, 0.0) for k in keys], [second.get(k, 0.0) for k in keys], feature.range)
        case "paths":
            a, b = {tuple(path) for path in first}, {tuple(path) for path in second}
            return 1 - len(a & b) / len(a | b) if a | b else 0.0
    raise AssertionError(feature.type)


def _numeric(first: list[float], second: list[float], bounds: tuple[float, float] | None) -> float:
    """Mean absolute difference, relative to the declared range or to the largest magnitude."""
    if not first:
        return 0.0
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    width = bounds[1] - bounds[0] if bounds else float(max(np.abs(a).max(), np.abs(b).max()))
    return 0.0 if width == 0 else float(min(1.0, np.abs(a - b).mean() / width))


def noise(group: Group, first: pa.Table, second: pa.Table, keys: list[str]) -> dict[str, tuple[float | None, int]]:
    """Per feature, the mean disagreement over the keys both tables hold with status ok, and how many those are."""
    wanted = set(keys)
    result: dict[str, tuple[float | None, int]] = {}
    for feature in group.features:
        measured = [ok_values(table, feature, wanted) for table in (first, second)]
        common = sorted(set(measured[0]) & set(measured[1]))
        scores = [disagreement(feature, measured[0][key], measured[1][key]) for key in common]
        result[feature.name] = (float(np.mean(scores)) if scores else None, len(common))
    return result


def ok_values(table: pa.Table, feature: FeatureSpec, wanted: set[str] | None = None) -> dict[str, Any]:
    keys = table[KEY].to_pylist()
    statuses = table[status_column(feature.name)].to_pylist()
    return {key: value for key, status, value in zip(keys, statuses, feature_values(table, feature), strict=True)
            if status == "ok" and value is not None and (wanted is None or key in wanted)}


def canonical(feature: FeatureSpec, value: Any) -> str:
    """One string per distinct value; a set is the same set in any order."""
    return json.dumps(sorted(value) if feature.type == "set" else value, sort_keys=True, ensure_ascii=False)


def labels_of(feature: FeatureSpec, values: list[Any]) -> set[str]:
    """Distinct labels of a category or set feature among these values."""
    if feature.type == "category":
        return set(values)
    return {label for value in values for label in value}


def scaled(frame: Frame, feature: str, reference: pd.Index) -> tuple[np.ndarray, np.ndarray]:
    """Which trajectories have a value for the feature, and its distance columns scaled by the reference set.

    Label columns keep their 0 to 1 values and other columns are standardized, as in `Frame.matrix`, so common
    labels weigh more than rare ones. Rows without a value hold zeros.
    """
    values = frame.distance[frame.columns[feature]].astype(float)
    has = values.notna().any(axis=1).to_numpy()
    if frame.types[feature] not in LABEL_TYPES:
        inside = values.loc[reference]
        values = (values - inside.mean()) / inside.std(ddof=0).replace(0, 1.0).fillna(1.0)
    return has, values.fillna(0.0).to_numpy()


def _tree() -> DecisionTreeRegressor:
    return DecisionTreeRegressor(random_state=0)


def _folds() -> KFold:
    return KFold(FOLDS, shuffle=True, random_state=0)


def dependence(x: tuple[np.ndarray, np.ndarray], y: tuple[np.ndarray, np.ndarray], inside: np.ndarray) -> float:
    """Share of the variance of y that a decision tree on x explains out of fold, over the reference rows with both."""
    rows = x[0] & y[0] & inside
    if rows.sum() < MIN_ROWS:
        return 0.0
    features, target = x[1][rows], y[1][rows]
    if target.var(axis=0).sum() < NO_VARIANCE:
        return 0.0
    predicted = cross_val_predict(_tree(), features, target, cv=_folds())
    return float(np.clip(r2_score(target, predicted, multioutput="variance_weighted"), 0.0, 1.0))


def assign_roles(
    order: list[str], distinct: dict[str, int], r: Callable[[str, str], float], limit: float
) -> dict[str, tuple[Role, Relation | None]]:
    """Give every feature its role, in `order`: the first feature that can stand for a group of features does."""
    roles: dict[str, tuple[Role, Relation | None]] = {}
    for feature in order:
        if distinct[feature] <= 1:
            roles[feature] = ("invariant", None)
            continue
        scores = {other: r(other, feature) for other, (role, _) in roles.items() if role == "distinguishing"}
        source = max(scores, key=lambda other: scores[other], default=None)
        if source is not None and scores[source] >= limit:
            roles[feature] = ("dependent", Relation(source=source, r=round(scores[source], 4)))
        else:
            roles[feature] = ("distinguishing", None)
    return roles


@dataclass
class Surprisal:
    """Per trajectory and term: the surprisal, the value, and how many reference trajectories share the value.

    A term is one column of the wide table, a category feature, a `__missing` column, or a relation `e->f`;
    `owner` maps each term to its feature.
    """

    terms: pd.DataFrame
    values: pd.DataFrame
    shared: pd.DataFrame
    owner: dict[str, str]

    def total(self) -> pd.Series:
        return self.terms.sum(axis=1)


def surprisal(
    frame: Frame, reference: list[str], relations: dict[str, str], skip: frozenset[str] = frozenset()
) -> Surprisal:
    """Score every trajectory of the frame against the reference set; `relations` maps a dependent feature to the
    feature that determines it, and the features in `skip` are left out."""
    index = frame.distance.index
    inside = index.isin(reference)
    terms: dict[str, np.ndarray] = {}
    values: dict[str, np.ndarray] = {}
    shared: dict[str, np.ndarray] = {}
    owner: dict[str, str] = {}

    def add(name: str, feature: str, term: np.ndarray, value: np.ndarray, count: np.ndarray, has: np.ndarray) -> None:
        terms[name] = np.where(has, term, 0.0)
        values[name] = np.where(has, value, np.nan)
        shared[name] = np.where(has, count, np.nan)
        owner[name] = feature

    everyone = np.ones(len(index), dtype=bool)
    for feature, columns in frame.columns.items():
        if feature in skip:
            continue
        raw = frame.distance[columns].astype(float)
        has = raw.notna().any(axis=1).to_numpy()
        if feature in frame.paths:
            has = frame.paths[feature].map(lambda value: isinstance(value, list)).to_numpy()
        ref = has & inside
        n = int(ref.sum())
        if not has.all():
            absent = (~has).astype(float)
            term, count = _binary(absent, absent[inside], int(inside.sum()))
            add(missing_column(feature), feature, term, absent, count, everyone)
        if feature in relations:
            source = relations[feature]
            if source not in frame.columns:
                raise ConfigError(f"The relation {source}->{feature} names a feature without extracted values; "
                                  "run `traj evaluate`")
            term, residual, count, both = _relation(scaled(frame, source, index[inside]),
                                                    scaled(frame, feature, index[inside]), inside)
            add(f"{source}->{feature}", feature, term, residual, count, both)
        elif frame.types[feature] == "paths":
            # One term per level: how unusual the trajectory's mean event surprisal at that level is.
            by_level = level_surprisal(frame.levels[feature], frame.paths[feature], inside)
            for level in by_level.columns:
                value = by_level[level].fillna(0.0).to_numpy()
                scored = has & by_level[level].notna().to_numpy()
                term, count = _upper(value, value[scored & inside])
                add(f"{feature}@{level}", feature, term, value, count, scored)
        elif frame.types[feature] == "category":
            labels = frame.query[feature].astype(object).to_numpy()
            counts = pd.Series(labels[ref]).value_counts()
            count = pd.Series(labels).map(counts).fillna(0).to_numpy(dtype=float)
            term = -np.log((count + 1) / (n + len(counts) + 1))
            add(feature, feature, term, np.full(len(index), np.nan), count, has)
        else:
            binary = frame.types[feature] in ("boolean", "set")
            for column in columns:
                value = raw[column].fillna(0.0).to_numpy()
                term, count = _binary(value, value[ref], n) if binary else _tail(value, value[ref])
                add(column, feature, term, value, count, has)
    return Surprisal(terms=pd.DataFrame(terms, index=index), values=pd.DataFrame(values, index=index),
                     shared=pd.DataFrame(shared, index=index), owner=owner)


def _binary(value: np.ndarray, reference: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    ones = float((reference == 1).sum())
    count = np.where(value == 1, ones, n - ones)
    return -np.log((count + 1) / (n + 2)), count


def _tail(value: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Surprisal of lying at least this far from the reference median, by counting reference trajectories."""
    if not len(reference):
        return np.zeros(len(value)), np.zeros(len(value))
    median = np.median(reference)
    distances = np.sort(np.abs(reference - median))
    count = len(distances) - np.searchsorted(distances, np.abs(value - median), side="left")
    return -np.log((count + 1) / (len(distances) + 1)), count.astype(float)


def _upper(value: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Surprisal of being at least this large, by counting reference trajectories."""
    known = np.sort(np.round(reference, 9))
    count = len(known) - np.searchsorted(known, np.round(value, 9), side="left")
    return -np.log((count + 1) / (len(known) + 1)), count.astype(float)


def _relation(
    x: tuple[np.ndarray, np.ndarray], y: tuple[np.ndarray, np.ndarray], inside: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Surprisal of the residual of y given x: reference rows use their out-of-fold residual."""
    both = x[0] & y[0]
    ref = both & inside
    size = len(inside)
    if ref.sum() < MIN_ROWS:
        return np.zeros(size), np.zeros(size), np.zeros(size), np.zeros(size, dtype=bool)
    residual = np.zeros(size)
    out_of_fold = cross_val_predict(_tree(), x[1][ref], y[1][ref], cv=_folds()).reshape(int(ref.sum()), -1)
    residual[ref] = np.linalg.norm(y[1][ref] - out_of_fold, axis=1)
    outside = both & ~inside
    if outside.any():
        fitted = _tree().fit(x[1][ref], y[1][ref]).predict(x[1][outside]).reshape(int(outside.sum()), -1)
        residual[outside] = np.linalg.norm(y[1][outside] - fitted, axis=1)
    term, count = _upper(residual, residual[ref])
    return term, residual, count, both


def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits, from 0 to 1."""
    middle = (p + q) / 2

    def kl(a: np.ndarray) -> float:
        positive = a > 0
        return float((a[positive] * np.log2(a[positive] / middle[positive])).sum())

    return (kl(p) + kl(q)) / 2


def _shares(codes: np.ndarray, size: int) -> np.ndarray:
    return np.bincount(codes, minlength=size) / len(codes)


def drift(frame: Frame, feature: str, inside: np.ndarray) -> float | None:
    """How far the monitored trajectories moved from the reference set, as the mean divergence of the columns."""
    columns = frame.columns[feature]
    raw = frame.distance[columns].astype(float)
    has = raw.notna().any(axis=1).to_numpy()
    ref, out = has & inside, has & ~inside
    if not ref.any() or not out.any():
        return None
    if frame.types[feature] == "category":
        codes, labels = pd.factorize(frame.query[feature].astype(object).to_numpy()[has])
        position = np.flatnonzero(has)
        within = np.isin(position, np.flatnonzero(ref))
        return _jsd(_shares(codes[within], len(labels)), _shares(codes[~within], len(labels)))
    scores = []
    for column in columns:
        value = raw[column].fillna(0.0).to_numpy()
        edges = np.unique(np.quantile(value[ref], np.linspace(0.1, 0.9, 9)))
        # A value equal to an edge gets a bin of its own, so a constant reference column still shows a move.
        codes = np.searchsorted(edges, value, side="left") + np.searchsorted(edges, value, side="right")
        size = 2 * len(edges) + 1
        scores.append(_jsd(_shares(codes[ref], size), _shares(codes[out], size)))
    return float(np.mean(scores))


def assess(
    project: Project, groups: list[Group], frame: Frame, reference: list[str],
    noises: dict[str, tuple[float | None, int]],
) -> list[Card]:
    """Compute variation, dependence, roles, decisions and drift of every feature of these groups."""
    config = project.config.evaluation
    index = frame.distance.index
    inside = index.isin(reference)
    wanted = set(reference)
    info: dict[str, tuple[Group, FeatureSpec]] = {f.name: (g, f) for g in groups for f in g.features}
    context = {f.name for g in groups for i in g.instances if i.ref.name == CONTEXT_OPERATOR for f in i.features}
    distinct: dict[str, int] = {}
    covered: dict[str, int] = {}
    overflow: set[str] = set()
    for group in groups:
        table = read_group(project, group)
        for feature in group.features:
            values = list(ok_values(table, feature, wanted).values())
            covered[feature.name] = len(values)
            distinct[feature.name] = len({canonical(feature, value) for value in values})
            if (feature.type in ("category", "set") and not feature.labels and feature.name not in context
                    and len(labels_of(feature, values)) > config.label_limit):
                overflow.add(feature.name)
    judged = [name for name in info if name not in context]
    # A paths feature has a column per node of its tree, so it takes no part in pairwise dependence.
    blocks = {name: scaled(frame, name, index[inside]) for name in judged
              if name in frame.columns and info[name][1].type != "paths"}
    cache: dict[tuple[str, str], float] = {}

    def r(x: str, y: str) -> float:
        if (x, y) not in cache:
            cache[x, y] = dependence(blocks[x], blocks[y], inside) if x in blocks and y in blocks else 0.0
        return cache[x, y]

    order = sorted(judged, key=lambda name: (not info[name][0].is_code, noises.get(name, (0.0, 0))[0] or 0.0, name))
    roles = assign_roles(order, distinct, r, config.dependence_min)
    roles.update(dict.fromkeys(context, ("context", None)))
    cards = []
    for name, (group, feature) in info.items():
        measured, compared = noises.get(name, (0.0 if group.is_code else None, 0))
        decision: Decision = "keep"
        if measured is not None and measured > config.noise_max:
            decision = "revise"
        elif name in overflow:
            decision = "labels"
        role, relation = roles[name]
        moved = drift(frame, name, inside) if name in frame.columns else None
        cards.append(Card(
            feature=name, group=group.name, kind=group.kind, type=feature.type,
            noise=None if measured is None else round(measured, 4), compared=compared, coverage=covered[name],
            distinct=distinct[name],
            role=role, relation=relation, decision=decision, drift=None if moved is None else round(moved, 4),
        ))
    return cards


def evaluate(project: Project, *, workers: int | None = None, runner: WorkerRunner | None = None) -> Evaluation:
    """Measure LLM groups a second time, judge every enabled feature, and write the result.

    A group extracted for a part of the reference set is judged on that part.
    """
    groups = load_groups(project)
    rows = load_index(project)
    reference = reference_keys(project, rows)
    if not reference:
        raise ConfigError("The reference set is empty")
    extracted = {group.name: _extracted(project, group, reference) for group in groups}
    llm = [group for group in groups if not group.is_code]
    for group in llm:
        if model_name(project, group, second=True) == model_name(project, group):
            raise ConfigError(f"evaluation.model equals the model of group {group.name}; a second measurement "
                              "by the same model shows no noise")
    by_key = {row.key: row for row in rows}
    measured: dict[str, list[str]] = {}
    noises: dict[str, tuple[float | None, int]] = {}
    for group in llm:
        measured[group.name] = evaluation_set(project, extracted[group.name])
        run_groups(project, [group], [by_key[key] for key in measured[group.name]], workers=workers, runner=runner,
                   second=True, directory=project.evaluation_dir)
        noises.update(noise(group, read_group(project, group), read_group(project, group, project.evaluation_dir),
                            measured[group.name]))
    frame = build_frame(project, groups, rows, project.config.sampling.vector_length)
    cards = assess(project, groups, frame, reference, noises)
    result = Evaluation(reference=len(reference), monitored=len(rows) - len(reference),
                        evaluation_set=evaluation_set(project, reference), measured=measured, features=cards)
    project.evaluation_dir.mkdir(parents=True, exist_ok=True)
    (project.evaluation_dir / FILE).write_text(result.model_dump_json(indent=1), encoding="utf-8")
    return result


def _extracted(project: Project, group: Group, reference: list[str]) -> list[str]:
    """Reference trajectories the group has rows for; a group without any stops the evaluation."""
    table = read_group(project, group)
    columns = [table[status_column(f.name)].to_pylist() for f in group.features]
    present = {key for key, row in zip(table[KEY].to_pylist(), zip(*columns, strict=True), strict=True)
               if any(status is not None for status in row)}
    keys = [key for key in reference if key in present]
    if not keys:
        raise ConfigError(f"Group {group.name} has no features for any reference trajectory; "
                          f"run `traj extract --group {group.name}`")
    return keys
