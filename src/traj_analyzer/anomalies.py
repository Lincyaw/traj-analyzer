from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel
from scipy.stats import ks_2samp
from sklearn.neighbors import NearestNeighbors

from traj_analyzer.batches import assign
from traj_analyzer.evaluation import CONTEXT_OPERATOR, canonical
from traj_analyzer.events.library import CONSEQUENCE_RULES
from traj_analyzer.ingest import IndexRow, load_index
from traj_analyzer.layers.tables import KEY, column, table_path, with_key
from traj_analyzer.operators.base import FeatureSpec
from traj_analyzer.operators.catalog import Group, load_groups
from traj_analyzer.project import Config, ConfigError, MiningConfig, Project
from traj_analyzer.reviews import precision_by_feature, read_verdicts
from traj_analyzer.vectorize import LABEL_TYPES, Frame, build_frame

TOP_VALUES = 8
DATASET = "dataset"
OUTLIER_NEIGHBOURS = 10
"""An outlier's score is its mean distance to this many nearest neighbours of its stratum."""
EXPLAINED_COLUMNS = 5
"""Columns named in each anomaly's reasons: those where it stands furthest from its neighbours."""
SHIFTS = pa.schema([
    column("batch", pa.string(), "The batch compared."),
    column("stratum", pa.string(), "The stratum inside which it was compared."),
    column("feature", pa.string(), "The feature of the column."),
    column("column", pa.string(), "The column of the wide table: the feature, or one of its labels or nodes."),
    column("effect", pa.float64(), "Standardized mean difference between the batch and the baseline."),
    column("ks", pa.float64(), "Kolmogorov-Smirnov statistic, for scalar and vector columns."),
    column("n", pa.int64(), "Trajectories of the batch in the stratum with a value."),
    column("n_baseline", pa.int64(), "Trajectories of the baseline in the stratum with a value."),
    column("direction", pa.string(), "up when the batch lies above the baseline, down otherwise."),
    column("rank", pa.int64(), "Position by the size of the effect, 1 first."),
])
ANOMALIES = pa.schema([
    column("batch", pa.string(), "The batch of the trajectory."),
    column("stratum", pa.string(), "The stratum of the trajectory."),
    column("outlier", pa.float64(), "Mean distance to the nearest neighbours of the stratum, divided by the median "
                                    "of that distance in the stratum."),
    column("events_score", pa.float64(), "Logarithm of one plus the sum over the unrecovered events of their "
                                         "rule's lift above 1, or 1 for a rule without a lift."),
    column("anomaly_score", pa.float64(), "outlier plus events_score."),
    column("anomaly_rank", pa.int64(), "Position by score, 1 first."),
    column("reasons", pa.string(), "JSON: the columns that set the trajectory apart, and its unrecovered events."),
])
LIFTS = pa.schema([
    column("rule", pa.string(), "The event rule."),
    column("hits", pa.int64(), "Events of the rule that fall in a turn."),
    column("followed", pa.float64(), "Share of the events whose turn is followed by a consequence: the next "
                                     "message the user typed corrects, rejects or abandons, or the user "
                                     "interrupted the turn."),
    column("baseline", pa.float64(), "Share of all turns with a tool call of the same strata that are followed by "
                                     "a consequence."),
    column("lift", pa.float64(), "followed divided by baseline."),
])


class Confinement(BaseModel):
    """One value of a context feature that holds nearly every trajectory where a feature varies."""

    context: str
    value: str
    share: float
    """Share of the varying trajectories with this context value."""
    base: float
    """Share of all trajectories with a value of the feature that have this context value."""


class Spread(BaseModel):
    """How the values of one feature spread over the trajectories it has a value for."""

    covered: int
    distinct: int
    varying: int
    """Trajectories whose value differs from the most common value."""
    varying_share: float
    top: dict[str, int]
    """The most common values with their counts, each value as JSON."""
    confined: Confinement | None = None
    batch_explained: float | None = None
    """Share of the variance the batches explain inside the strata; None without declared batches."""

    def objection(self, config: MiningConfig) -> str | None:
        """Why a feature with this spread tells too little; None when it passes."""
        if self.distinct < 2:
            return "it takes one value on every trajectory it was extracted from"
        if self.varying_share < config.varying_min:
            return (f"{self.varying} of {self.covered} trajectories differ from its most common value, a share "
                    f"below mining.varying_min")
        if self.confined is not None:
            c = self.confined
            return (f"{round(c.share * 100)}% of the trajectories where it varies have {c.context} = {c.value}, "
                    f"which {round(c.base * 100)}% of the trajectories have")
        if self.batch_explained is not None and self.batch_explained < config.batch_explained_min:
            return (f"the batches explain {round(self.batch_explained * 100, 1)}% of its variance inside the strata, "
                    f"below mining.batch_explained_min")
        return None


def context_names(groups: list[Group]) -> frozenset[str]:
    """The context features: those the `meta.fields` operator copies from metadata."""
    return frozenset(f.name for g in groups for i in g.instances if i.ref.name == CONTEXT_OPERATOR for f in i.features)


def context_frame(project: Project, groups: list[Group], rows: list[IndexRow], context: frozenset[str]) -> pd.DataFrame:
    """What a feature must not be confined to, one column each, indexed by trajectory key: the dataset of a
    trajectory, and the category features among the context features; with the batch and stratum of every
    trajectory as the columns `batch` and `stratum`."""
    frame = build_frame(project, groups, rows, project.config.sampling.vector_length)
    names = [name for name in frame.columns if name in context and frame.types[name] == "category"]
    datasets = pd.Series({row.key: row.dataset for row in rows}, name=DATASET)
    found = frame.query[names].join(datasets) if DATASET not in names else frame.query[names]
    found["batch"] = assign(project, frame, rows, project.config.batches)
    found["stratum"] = assign(project, frame, rows, project.config.strata)
    return found


def spread(feature: FeatureSpec, values: dict[str, Any], context: pd.DataFrame, config: MiningConfig) -> Spread:
    """Variation of `values`, trajectory key to value, and whether it stays inside one value of a context column.

    The varying trajectories are confined when at least `mining.context_share_max` of them share one value of a
    context column that fewer than half of the trajectories have. With batches declared, `batch_explained` is the
    share of the variance of a numeric or boolean feature that the batches explain inside the strata.
    """
    names = {key: canonical(feature, value) for key, value in values.items()}
    counts = Counter(names.values())
    if not counts:
        return Spread(covered=0, distinct=0, varying=0, varying_share=0.0, top={})
    common = min(counts, key=lambda value: (-counts[value], value))
    varying = [key for key, value in names.items() if value != common]
    found = None
    known = context.reindex(list(names))
    for name in known.columns:
        if name in ("batch", "stratum"):
            continue
        inside = known.loc[varying, name].dropna()
        everywhere = known[name].dropna()
        if inside.empty:
            continue
        sizes = inside.value_counts()
        value = min(sizes.index, key=lambda label: (-sizes[label], str(label)))
        share, base = float(sizes[value]) / len(inside), float((everywhere == value).mean())
        if share >= config.context_share_max and base < 0.5 and (found is None or share > found.share):
            found = Confinement(context=name, value=str(value), share=round(share, 4), base=round(base, 4))
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:TOP_VALUES]
    explained = None
    if "batch" in known.columns and known["batch"].nunique() > 1 and feature.type in ("scalar", "boolean", "event"):
        numeric = pd.Series({key: _number(feature, value) for key, value in values.items()}, dtype=float)
        explained = batch_explained(numeric, known["batch"], known["stratum"])
    return Spread(covered=len(names), distinct=len(counts), varying=len(varying),
                  varying_share=round(len(varying) / len(names), 4), top=dict(ordered), confined=found,
                  batch_explained=None if explained is None else round(explained, 4))


def _number(feature: FeatureSpec, value: Any) -> float:
    if feature.type == "event":
        return float(sum(not event["recovered"] for event in value))
    return float(value)


def batch_explained(values: pd.Series, batch: pd.Series, stratum: pd.Series) -> float | None:
    """Share of the variance of `values` that the means of the batches explain after the means of the strata are
    removed; None when the values do not vary."""
    known = values.dropna()
    if len(known) < 2:
        return None
    total = float(known.var(ddof=0))
    if total == 0:
        return None
    centred = known - known.groupby(stratum.reindex(known.index)).transform("mean")
    between = centred.groupby(batch.reindex(known.index)).transform("mean")
    return float(between.var(ddof=0) / total)


@dataclass
class Population:
    """The trajectories under analysis: their features, and the batch and stratum of each."""

    frame: Frame
    batch: pd.Series
    stratum: pd.Series


def population(project: Project, groups: list[Group], rows: list[IndexRow]) -> Population:
    frame = build_frame(project, groups, rows, project.config.sampling.vector_length)
    return Population(frame=frame, batch=assign(project, frame, rows, project.config.batches),
                      stratum=assign(project, frame, rows, project.config.strata))


def shifts(population: Population, config: Config, skip: frozenset[str] = frozenset()) -> pd.DataFrame:
    """Every batch against the baseline inside every stratum, one row per column of the wide table.

    The baseline is every other batch of the stratum, or the batch `baseline` names. A comparison needs
    `alerts.min_batch` trajectories with a value on each side. Rows are ranked by the size of the effect.
    The features in `skip`, the context features, are left out: they are what batches and strata are made of.
    """
    frame, batch, stratum = population.frame, population.batch, population.stratum
    owner = {name: feature for feature, names in frame.columns.items() for name in names if feature not in skip}
    distance = frame.distance[list(owner)].astype(float)
    found: list[dict[str, Any]] = []
    for layer in sorted(stratum.unique()):
        inside = stratum == layer
        for name in sorted(batch[inside].unique()):
            if name == config.baseline:
                continue
            own = inside & (batch == name)
            base = inside & (batch != name) if config.baseline == "all" else inside & (batch == config.baseline)
            if own.sum() < config.alerts.min_batch or base.sum() < config.alerts.min_batch:
                continue
            for column_name in distance.columns:
                x = distance.loc[own, column_name].dropna().to_numpy()
                y = distance.loc[base, column_name].dropna().to_numpy()
                if len(x) < config.alerts.min_batch or len(y) < config.alerts.min_batch:
                    continue
                pooled = np.sqrt((x.var(ddof=0) + y.var(ddof=0)) / 2)
                if pooled == 0:
                    continue
                effect = float((x.mean() - y.mean()) / pooled)
                feature = owner[column_name]
                ks = float(ks_2samp(x, y).statistic) if frame.types[feature] not in (*LABEL_TYPES, "boolean", "event") \
                    else None
                found.append({"batch": name, "stratum": layer, "feature": feature, "column": column_name,
                              "effect": round(effect, 4), "ks": None if ks is None else round(ks, 4),
                              "n": int(len(x)), "n_baseline": int(len(y)),
                              "direction": "up" if effect > 0 else "down"})
    table = pd.DataFrame(found, columns=[f.name for f in SHIFTS if f.name != "rank"])
    table = table.reindex(table["effect"].abs().sort_values(ascending=False, kind="stable").index)
    table["rank"] = np.arange(1, len(table) + 1)
    return table.reset_index(drop=True)


@dataclass
class Lift:
    rule: str
    hits: int
    followed: float | None
    baseline: float | None

    @property
    def lift(self) -> float | None:
        if self.followed is None or not self.baseline:
            return None
        return self.followed / self.baseline


INTERRUPT = "interrupt"
"""The consequence that needs no judgment: a turn the user interrupted, read off `turns.ended_by`."""


def turn_consequences(
    turns: pd.DataFrame, events: pd.DataFrame, config: Config, covered: set[str] | None = None
) -> pd.DataFrame:
    """The turns with one column per consequence: `judged`, whether the message that opens the next turn starts
    with a consequence event, `interrupted`, whether the user interrupted the turn, and `followed`, either one
    that `alerts.consequences` counts.

    `judged` is None for a turn no judged consequence could be found in: the last turn of a trajectory, and every
    turn of a trajectory outside `covered` when the judged consequences were extracted from a sample.
    """
    judged_rules = [rule for rule in config.alerts.consequences if rule != INTERRUPT]
    interrupts = INTERRUPT in config.alerts.consequences
    starts: set[tuple[str, int]] = set()
    if judged_rules and len(events):
        marked = events[events["rule"].isin(judged_rules)]
        starts = set(zip(marked[KEY], marked["step"], strict=True))
    # A judged consequence was looked for in the covered trajectories, or everywhere when a code rule gives it.
    looked = bool(judged_rules) and (covered is not None or bool(starts))
    ordered = turns.sort_values([KEY, "index"]).reset_index(drop=True)
    judged = np.full(len(ordered), np.nan)
    for key, group in ordered.groupby(KEY, sort=False):
        if not looked or (covered is not None and key not in covered):
            continue
        next_starts = group["start"].to_numpy()[1:]
        for row, next_start in zip(group.index[:-1], next_starts, strict=True):
            judged[row] = float((key, int(next_start)) in starts)
    ordered["judged"] = pd.Series(judged, index=ordered.index).where(~np.isnan(judged), None)
    ordered["interrupted"] = (ordered["ended_by"] == INTERRUPT) if interrupts else False
    ordered["followed"] = ordered["interrupted"] | ordered["judged"].fillna(0).astype(bool)
    return ordered


def consequence_lifts(
    events: pd.DataFrame, turns: pd.DataFrame, stratum: pd.Series, config: Config, covered: set[str] | None = None,
    skip: frozenset[str] = frozenset(),
) -> dict[str, Lift]:
    """Per event rule, how much more often a consequence follows the turns its events fall in than it follows any
    turn of the same strata in which the agent called a tool.

    `events` has key, rule and step; `turns` is the turns table; a consequence is an event of a rule in
    `alerts.consequences` at the message that opens the next turn, or the interruption of the turn when
    `interrupt` is among the consequences. With `covered`, only those trajectories count for the judged
    consequences: the ones they were extracted from, since a trajectory without them can show no such consequence.
    The rules in `skip`, those that fire on a consequence itself, get no lift.
    """
    marked = turn_consequences(turns, events, config, covered)
    judged_rules = [rule for rule in config.alerts.consequences if rule != INTERRUPT]
    if covered is not None and judged_rules:
        marked = marked[marked[KEY].isin(covered)]
        events = events[events[KEY].isin(covered)]
    active = marked[marked["calls"] > 0]
    baseline: dict[str, float] = {}
    for layer, keys in stratum.groupby(stratum).groups.items():
        inside = active[active[KEY].isin(set(keys))]
        baseline[str(layer)] = float(inside["followed"].mean()) if len(inside) else 0.0
    by_key = {key: group for key, group in marked.groupby(KEY, sort=False)}
    lifts: dict[str, Lift] = {}
    for rule, own_events in events[~events["rule"].isin([*config.alerts.consequences, *skip])].groupby("rule"):
        count = expected = seen = 0.0
        for key, group in own_events.groupby(KEY):
            own_turns = by_key.get(key)
            if own_turns is None or own_turns.empty:
                continue
            turn_starts = own_turns["start"].to_numpy()
            turn_ends = own_turns["end"].to_numpy()
            followed = own_turns["followed"].to_numpy()
            steps = group["step"].to_numpy()
            positions = np.searchsorted(turn_starts, steps, side="right") - 1
            inside = (positions >= 0) & (steps <= turn_ends[np.maximum(positions, 0)])
            count += int(inside.sum())
            seen += int(followed[positions[inside]].sum())
            expected += int(inside.sum()) * baseline.get(str(stratum.get(key, "")), 0.0)
        lifts[str(rule)] = Lift(rule=str(rule), hits=int(count), followed=seen / count if count else None,
                                baseline=expected / count if count else None)
    return lifts


OVERSIGHT = pa.schema([
    column("batch", pa.string(), "The batch."),
    column("turns", pa.int64(), "Turns, one per message the user typed."),
    column("ask_share", pa.float64(), "Share of the turns in which the agent asked the user something."),
    column("interrupt_share", pa.float64(), "Share of the turns the user interrupted."),
    column("pushback_share", pa.float64(), "Share of the judged turns whose next message corrects, rejects or "
                                           "abandons; empty without judged consequences."),
    column("duration_p50", pa.float64(), "Median seconds of a turn, over the turns with a duration."),
    column("duration_p99", pa.float64(), "99th percentile of the seconds of a turn."),
    column("calls_per_turn", pa.float64(), "Mean tool calls per turn."),
])


def oversight(turns: pd.DataFrame, events: pd.DataFrame, batch: pd.Series, config: Config,
              covered: set[str] | None = None) -> pd.DataFrame:
    """Per batch, how the user and the agent oversee each other: how often the agent asks, how often the user
    interrupts or pushes back, how long a turn runs and how many calls it holds."""
    marked = turn_consequences(turns, events, config, covered)
    marked["batch"] = marked[KEY].map(batch)
    rows = []
    for name, group in marked.groupby("batch", sort=True):
        durations = pd.to_numeric(group["duration_s"], errors="coerce").dropna()
        judged = group["judged"].dropna()
        rows.append({
            "batch": str(name), "turns": int(len(group)),
            "ask_share": float((group["asks"] > 0).mean()),
            "interrupt_share": float((group["ended_by"] == INTERRUPT).mean()),
            "pushback_share": float(judged.astype(float).mean()) if len(judged) else None,
            "duration_p50": float(durations.quantile(0.5)) if len(durations) else None,
            "duration_p99": float(durations.quantile(0.99)) if len(durations) else None,
            "calls_per_turn": float(group["calls"].mean()),
        })
    return pd.DataFrame(rows, columns=[field_.name for field_ in OVERSIGHT])


def event_weight(lift: Lift | None) -> float:
    """What one unrecovered event adds before the logarithm: how much more often than any call a consequence
    follows the rule's events, 1 for a rule whose lift is unknown, 0 for a rule that is followed no more often."""
    if lift is None or lift.lift is None:
        return 1.0
    return max(lift.lift - 1.0, 0.0)


def anomalies(
    population: Population, events: pd.DataFrame, lifts: dict[str, Lift], skip: frozenset[str] = frozenset()
) -> pd.DataFrame:
    """Every trajectory scored inside its stratum: its distance to its neighbours over the features outside `skip`,
    and the logarithm of one plus the weighted sum of its unrecovered events."""
    frame, stratum = population.frame, population.stratum
    keys = list(frame.query.index)
    weights = {name: 1.0 for name in frame.columns if name not in skip}
    matrix = frame.matrix(weights) if weights else None
    outlier = np.full(len(keys), np.nan)
    reasons: list[dict[str, Any]] = [{"columns": [], "events": []} for _ in keys]
    position = {key: i for i, key in enumerate(keys)}
    if matrix is not None:
        for _, members in stratum.groupby(stratum).groups.items():
            rows = np.array([position[key] for key in members])
            if len(rows) < 3:
                continue
            dist, idx = NearestNeighbors(n_neighbors=min(OUTLIER_NEIGHBOURS, len(rows) - 1)).fit(
                matrix.x[rows]).kneighbors()
            own = dist.mean(axis=1)
            typical = float(np.median(own)) or float(own.mean()) or 1.0
            outlier[rows] = own / typical
            for i, row in enumerate(rows):
                deviation = matrix.x[row] - matrix.x[rows[idx[i]]].mean(axis=0)
                top = np.argsort(-np.abs(deviation))[:EXPLAINED_COLUMNS]
                reasons[row]["columns"] = [
                    {"column": matrix.columns[j],
                     "value": None if np.isnan(matrix.raw[row, j]) else round(float(matrix.raw[row, j]), 4),
                     "deviation": round(float(deviation[j]), 3)} for j in top]
    weighted = np.zeros(len(keys))
    open_events = events[~events["recovered"].fillna(False).astype(bool)] if len(events) else events
    for key, group in open_events.groupby("key"):
        if key not in position:
            continue
        row = position[key]
        for _, event in group.sort_values("step").iterrows():
            lift = lifts.get(event["rule"])
            weighted[row] += event_weight(lift)
            reasons[row]["events"].append({"rule": event["rule"], "step": int(event["step"]),
                                           "lift": None if lift is None or lift.lift is None else round(lift.lift, 3)})
    events_score = np.log1p(weighted)
    score = np.nan_to_num(outlier, nan=0.0) + events_score
    order = np.argsort(-score, kind="stable")
    rank = np.empty(len(keys), dtype=int)
    rank[order] = np.arange(1, len(keys) + 1)
    return pd.DataFrame({
        KEY: keys, "batch": population.batch.to_numpy(), "stratum": stratum.to_numpy(),
        "outlier": np.round(outlier, 4), "events_score": np.round(events_score, 4),
        "anomaly_score": np.round(score, 4), "anomaly_rank": rank,
        "reasons": [json.dumps(reason, ensure_ascii=False) for reason in reasons],
    })


def read_events(project: Project) -> pd.DataFrame:
    path = table_path(project, "events")
    if not path.is_file():
        return pd.DataFrame(columns=[KEY, "rule", "step", "end", "edge", "objects", "recovered", "recovered_at",
                                     "note"])
    return pq.read_table(path).to_pandas()


def consequence_coverage(project: Project) -> set[str] | None:
    """The trajectories the consequence rules were extracted from: those with a judged unit of such a rule, when
    the rules are judged outputs; None when they are code rules, which every trajectory has."""
    path = table_path(project, "judgments")
    if not path.is_file():
        return None
    judgments = pq.read_table(path, columns=[KEY, "feature"]).to_pandas()
    judged = judgments[judgments["feature"].isin(project.config.alerts.consequences)]
    if judged.empty:
        return None
    return set(judged[KEY].unique())


def read_turns(project: Project) -> pd.DataFrame:
    """The turns table; the `turns` layer writes it, and the consequences and the oversight rest on it."""
    path = table_path(project, "turns")
    if not path.is_file():
        raise ConfigError("No turns table; enable the `turns` layer in traj.yaml and run `traj extract`")
    return pq.read_table(path).to_pandas()


def human_precision(project: Project) -> dict[str, tuple[float, int]]:
    """Per feature, the share of the trajectories a person found worth a look and how many they judged."""
    return precision_by_feature(read_verdicts(project))


def run_anomalies(project: Project) -> dict[str, Any]:
    """Compute the shifts, the lifts, the oversight and the anomalies of the project and write them as tables."""
    groups = load_groups(project)
    rows = load_index(project)
    found = population(project, groups, rows)
    context = context_names(groups)
    shifted = shifts(found, project.config, context)
    events = read_events(project)
    turns = read_turns(project)
    covered = consequence_coverage(project)
    lifts = consequence_lifts(events, turns, found.stratum, project.config, covered, CONSEQUENCE_RULES)
    watched = oversight(turns, events, found.batch, project.config, covered)
    scored = anomalies(found, events, lifts, context)
    project.tables_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(shifted, schema=SHIFTS, preserve_index=False), table_path(project, "shifts"))
    pq.write_table(pa.Table.from_pandas(scored, schema=with_key(ANOMALIES), preserve_index=False),
                   table_path(project, "anomalies"))
    pq.write_table(pa.Table.from_pandas(watched, schema=OVERSIGHT, preserve_index=False),
                   table_path(project, "oversight"))
    lift_rows = [{"rule": lift.rule, "hits": lift.hits, "followed": lift.followed, "baseline": lift.baseline,
                  "lift": lift.lift} for lift in lifts.values()]
    pq.write_table(pa.Table.from_pylist(lift_rows, schema=LIFTS), table_path(project, "lifts"))
    reviews = human_precision(project)
    return {
        "trajectories": len(rows),
        "batches": sorted(found.batch.unique().tolist()),
        "strata": sorted(found.stratum.unique().tolist()),
        "shifts": shifted.head(20).to_dict("records"),
        "lifts": lift_rows,
        "oversight": watched.to_dict("records"),
        "anomalies": scored.sort_values("anomaly_rank").head(20).to_dict("records"),
        "reviews": {feature: {"precision": round(share, 4), "verdicts": count}
                    for feature, (share, count) in sorted(reviews.items())},
    }
