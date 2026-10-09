from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pydantic import BaseModel, Field
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from traj_analyzer.anomalies import read_events
from traj_analyzer.batches import assign
from traj_analyzer.ingest import IndexRow, load_trajectory
from traj_analyzer.layers.tables import KEY, table_path
from traj_analyzer.operators.catalog import Group
from traj_analyzer.project import ConfigError, Project
from traj_analyzer.vectorize import Frame

MIN_ROWS = 20
TOP_SHARE = 0.1
"""Share of the trajectories, by anomaly rank, whose failure rate is compared with everyone's."""
REACTION = "reaction"
"""The judged output checked against the turn labels."""
PUSHBACK_LABELS = ("correction", "rejection", "failure_report", "non_pushback")
"""The labels a turn label takes; a judged reaction outside the first three maps to non_pushback."""


class Agreement(BaseModel):
    feature: str
    rho: float | None = None
    """Spearman correlation with the outcome inside the strata, over trajectories with both."""
    n: int = 0


class LabelsReport(BaseModel):
    """How the self-mined signals and the anomalies agree with the labels, where a dataset has them."""

    outcome: str | None = None
    failed: int = 0
    """Trajectories the outcome label counts as failed."""
    agreement: list[Agreement] = Field(default_factory=list)
    """Every signal, event count and alerting feature against the outcome."""
    anomaly_lift: float | None = None
    """Failure rate of the top anomalies divided by the failure rate of everyone."""
    anomaly_top: int = 0
    correction: str | None = None
    corrected: int = 0
    """Trajectories the correction label counts as pushed back on."""
    event_coverage: float | None = None
    """Share of the corrected trajectories that hold at least one event."""
    turn_correction: str | None = None
    turns_compared: int = 0
    """Messages the user typed with both a judged reaction and a turn label."""
    turn_agreement: float | None = None
    """Share of the compared messages where the judged reaction, mapped to the four labels, equals the label."""
    turn_kappa: float | None = None
    """Cohen's kappa over the four labels."""
    turn_agreement_binary: float | None = None
    """Agreement with correction, rejection and failure_report collapsed into pushback."""
    turn_kappa_binary: float | None = None
    turn_confusion: dict[str, dict[str, int]] = Field(default_factory=dict)
    """Judged reaction, mapped, to the turn label to the number of messages."""


def failure(frame: Frame, project: Project) -> pd.Series | None:
    """Per trajectory, whether the outcome label counts it as failed; None without an outcome label."""
    name = project.config.labels.outcome
    if name is None:
        return None
    if name not in frame.query.columns:
        raise ConfigError(f"labels.outcome names {name}, which is not an extracted feature")
    values = frame.query[name]
    numeric = pd.to_numeric(values, errors="coerce")
    if frame.types.get(name) == "boolean":
        return (numeric == 0).where(numeric.notna())
    threshold = project.config.labels.failure_below
    if threshold is None:
        threshold = float(numeric.median())
    return (numeric < threshold).where(numeric.notna())


def corrected(frame: Frame, project: Project) -> pd.Series | None:
    name = project.config.labels.correction
    if name is None:
        return None
    if name not in frame.query.columns:
        raise ConfigError(f"labels.correction names {name}, which is not an extracted feature")
    numeric = pd.to_numeric(frame.query[name], errors="coerce")
    return (numeric > 0).where(numeric.notna())


def stratified_rho(x: pd.Series, y: pd.Series, stratum: pd.Series) -> float | None:
    """Spearman's correlation of `x` and `y` inside the strata: one correlation per stratum with at least
    `MIN_ROWS` trajectories and both variables varying, averaged with the strata weighted by their size; None when
    no stratum qualifies."""
    groups = stratum.reindex(x.index)
    total, weighted = 0, 0.0
    for _, members in x.groupby(groups).groups.items():
        a, b = x.loc[members], y.loc[members]
        if len(members) < MIN_ROWS or a.nunique() < 2 or b.nunique() < 2:
            continue
        rho = float(spearmanr(a, b).statistic)
        if np.isnan(rho):
            continue
        total += len(members)
        weighted += rho * len(members)
    return weighted / total if total else None


def map_reaction(reaction: str) -> str:
    """The turn label a judged reaction stands for: its own name for the three pushback kinds, non_pushback for
    the rest, among them `repeat`, which the turn labels do not tell apart."""
    return reaction if reaction in PUSHBACK_LABELS[:3] else "non_pushback"


def judged_reactions(project: Project) -> pd.DataFrame:
    """The judged reaction of every message the user typed: key, step and the label it maps to."""
    path = table_path(project, "judgments")
    if not path.is_file():
        return pd.DataFrame(columns=[KEY, "step", "judged"])
    judgments = pq.read_table(path, columns=[KEY, "start", "feature", "value"]).to_pandas()
    judgments = judgments[judgments["feature"] == REACTION]
    return pd.DataFrame({KEY: judgments[KEY].to_numpy(), "step": judgments["start"].to_numpy(),
                         "judged": [map_reaction(json.loads(value)) for value in judgments["value"]]})


def turn_labels(project: Project, keys: list[str]) -> pd.DataFrame:
    """The turn labels of these trajectories from their metadata: key, step and label."""
    name = project.config.labels.turn_correction
    assert name is not None
    found = []
    for key in keys:
        trajectory = load_trajectory(project, key)
        for item in trajectory.metadata.get(name) or []:
            label = item.get("label", item.get("pushback"))
            if label is None:
                continue
            if label not in PUSHBACK_LABELS:
                raise ConfigError(f"{key}: the turn label {label!r} is not one of {PUSHBACK_LABELS}")
            found.append({KEY: key, "step": int(item["step"]), "label": label})
    return pd.DataFrame(found, columns=[KEY, "step", "label"])


def _kappa(a: pd.Series, b: pd.Series) -> float | None:
    if len(a) < 2 or (a.nunique() < 2 and b.nunique() < 2):
        return None
    return round(float(cohen_kappa_score(a, b)), 4)


def turn_report(project: Project, report: LabelsReport) -> None:
    """Fill in how the judged reactions agree with the turn labels, over the messages that have both."""
    judged = judged_reactions(project)
    if judged.empty:
        return
    labels = turn_labels(project, sorted(judged[KEY].unique()))
    both = judged.merge(labels, on=[KEY, "step"])
    report.turns_compared = len(both)
    if both.empty:
        return
    report.turn_agreement = round(float((both["judged"] == both["label"]).mean()), 4)
    report.turn_kappa = _kappa(both["judged"], both["label"])
    binary = both.replace({"judged": {label: "pushback" for label in PUSHBACK_LABELS[:3]},
                           "label": {label: "pushback" for label in PUSHBACK_LABELS[:3]}})
    report.turn_agreement_binary = round(float((binary["judged"] == binary["label"]).mean()), 4)
    report.turn_kappa_binary = _kappa(binary["judged"], binary["label"])
    confusion: dict[str, dict[str, int]] = {}
    for (ours, theirs), count in sorted(Counter(zip(both["judged"], both["label"], strict=True)).items()):
        confusion.setdefault(ours, {})[theirs] = int(count)
    report.turn_confusion = confusion


def labels_report(project: Project, frame: Frame, groups: list[Group], rows: list[IndexRow]) -> LabelsReport:
    labels = project.config.labels
    report = LabelsReport(outcome=labels.outcome, correction=labels.correction, turn_correction=labels.turn_correction)
    stratum = assign(project, frame, rows, project.config.strata)
    failed = failure(frame, project)
    if failed is not None:
        report.failed = int(failed.fillna(False).sum())
        target = failed.astype(float)
        candidates = [name for name in frame.columns
                      if frame.types[name] in ("scalar", "boolean", "event") and name != project.config.labels.outcome]
        alerting = {f.name for g in groups for i in g.instances for f in i.features
                    if i.use is not None and i.use.alert}
        signals = {f.name for g in groups for i in g.instances for f in i.features if i.ref.name.startswith("signals.")}
        for name in candidates:
            column = name if frame.types[name] != "event" else f"{name}__unrecovered"
            if column not in frame.distance.columns:
                continue
            both = pd.concat([pd.to_numeric(frame.distance[column], errors="coerce"), target], axis=1).dropna()
            if len(both) < MIN_ROWS or both.iloc[:, 1].nunique() < 2:
                continue
            # A column that differs from its most common value on fewer than MIN_ROWS trajectories tells nothing.
            if (both.iloc[:, 0] != both.iloc[:, 0].mode().iloc[0]).sum() < MIN_ROWS:
                continue
            rho = stratified_rho(both.iloc[:, 0], both.iloc[:, 1], stratum)
            if name in signals or frame.types[name] == "event" or name in alerting:
                report.agreement.append(Agreement(feature=name, rho=None if rho is None else round(rho, 4),
                                                  n=len(both)))
        path = table_path(project, "anomalies")
        if path.is_file():
            ranks = pq.read_table(path, columns=[KEY, "anomaly_rank"]).to_pandas().set_index(KEY)["anomaly_rank"]
            ranks = ranks.reindex(frame.query.index)
            known = failed.notna() & ranks.notna()
            top = max(1, int(round(known.sum() * TOP_SHARE)))
            chosen = known & (ranks <= top)
            everyone = float(failed[known].astype(float).mean()) if known.any() else 0.0
            report.anomaly_top = int(chosen.sum())
            if chosen.any() and everyone > 0:
                report.anomaly_lift = round(float(failed[chosen].astype(float).mean()) / everyone, 4)
    pushed = corrected(frame, project)
    if pushed is not None:
        report.corrected = int(pushed.fillna(False).sum())
        events = read_events(project)
        with_events = set(events[KEY].unique()) if len(events) else set()
        keys = [key for key, value in pushed.items() if value is True]
        if keys:
            report.event_coverage = round(sum(key in with_events for key in keys) / len(keys), 4)
    if labels.turn_correction is not None:
        turn_report(project, report)
    return report
