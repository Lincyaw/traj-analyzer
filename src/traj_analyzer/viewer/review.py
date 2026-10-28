from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field

from traj_analyzer.anomalies import consequence_coverage, context_names, oversight, read_events, read_turns
from traj_analyzer.batches import META, assign
from traj_analyzer.evaluation import FILE as EVALUATION_FILE
from traj_analyzer.evaluation import Card, load_evaluation
from traj_analyzer.features.table import KEY
from traj_analyzer.hunting import VERDICTS_TABLE, load_hunt
from traj_analyzer.hunting.spec import hunt_names
from traj_analyzer.hunting.summary import finished_runs, summarize_run
from traj_analyzer.ingest import load_index
from traj_analyzer.labels import TOP_SHARE
from traj_analyzer.operators.base import FeatureSpec
from traj_analyzer.operators.catalog import Group, Instance
from traj_analyzer.project import ConfigError, Project
from traj_analyzer.reviews import Verdict, append_verdict, precision_by_feature, read_verdicts
from traj_analyzer.vectorize import Frame, build_frame
from traj_analyzer.viewer.query import FEATURES, quote

if TYPE_CHECKING:
    from traj_analyzer.viewer.server import Database

REASONS = "review.reasons"
"""The columns of the anomaly reasons, one row per trajectory and column, with the feature the column belongs to."""
SHIFTS_SHOWN = 6
"""Features shown per batch in the overview: those that moved furthest from the baseline."""
EVENTS_SHOWN = 6
"""Unrecovered events listed per trajectory in the list."""
STEPS_PAGE = 400
"""Most steps one request for step contents returns."""
NUMERIC = {"DOUBLE", "FLOAT", "BIGINT", "INTEGER", "HUGEINT", "SMALLINT", "TINYINT"}

Sort = Literal["anomaly", "score", "duration", "turns", "steps"]
ORDERS: dict[str, str] = {
    "anomaly": "base.anomaly_rank ASC NULLS LAST",
    "score": "base.anomaly_score DESC NULLS LAST",
    "duration": "t.duration_s DESC NULLS LAST",
    "turns": "t.turns DESC NULLS LAST",
    "steps": "tr.steps DESC NULLS LAST",
}


class Judged(BaseModel):
    """A label of a per-unit output a trajectory must carry on some unit."""

    model_config = ConfigDict(extra="forbid")

    feature: str
    label: str


class Grouping(BaseModel):
    """One group of the overview: the columns the trajectories were grouped by, and the group's value."""

    model_config = ConfigDict(extra="forbid")

    by: list[str] = Field(min_length=1)
    value: str


class TrajectoriesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group: Grouping | None = None
    batch: str | None = None
    stratum: str | None = None
    rule: str | None = None
    """An event rule with an unrecovered event in the trajectory."""
    judged: Judged | None = None
    ended_by: str | None = None
    """A way some turn of the trajectory ended."""
    feature: str | None = None
    """A feature among the trajectory's anomaly reasons, or the rule of one of its unrecovered events."""
    reviewed: bool | None = None
    search: str = ""
    sort: Sort = "anomaly"
    page: int = Field(default=1, ge=1)
    size: int = Field(default=50, ge=1, le=500)


class VerdictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    feature: str
    step: int | None = None
    worth: bool
    note: str = ""


@dataclass(frozen=True)
class Described:
    """A feature of a group, or a per-unit output of a judgment group, with where it comes from."""

    spec: FeatureSpec
    group: Group
    instance: Instance
    unit: str | None
    """None for a feature of the trajectory, the unit for a per-unit output."""

    def info(self) -> dict[str, Any]:
        spec = self.spec
        return {"name": spec.name, "description": spec.description.strip(), "type": spec.type, "labels": spec.labels,
                "range": spec.range, "group": self.group.name, "kind": self.group.kind, "operator": self.instance.id,
                "unit": self.unit, "event": spec.event.edge if spec.event is not None else None}


class Review:
    """What the review pages show: the batches, the trajectories worth a look, one trajectory's course, and how the
    features hold up, all read from the tables and the feature specs of the project."""

    def __init__(self, project: Project, database: Database) -> None:
        self.project = project
        self.database = database
        self.tables = set(database.stored)
        self.features = {f.name: Described(f, g, i, None)
                         for g in database.groups for i in g.instances for f in i.features}
        self.outputs = {o.name: Described(o, g, i, g.unit)
                        for g in database.groups if g.kind == "judgment" for i in g.instances for o in i.outputs}
        self.rows = load_index(project)
        self.frame: Frame = build_frame(project, database.groups, self.rows, project.config.sampling.vector_length)
        self.context = context_names(database.groups)
        self.stratum = assign(project, self.frame, self.rows, project.config.strata)
        self.batch = assign(project, self.frame, self.rows, project.config.batches)
        self.turns = read_turns(project) if "turns" in self.tables else None
        self.events = read_events(project)
        self.covered = consequence_coverage(project)
        self._overviews: dict[tuple[str, ...], dict[str, Any]] = {}
        self._load_reasons()
        self._load_hunts()

    # Loading

    def _load_reasons(self) -> None:
        rows: dict[str, list[Any]] = {KEY: [], "column": [], "feature": [], "value": [], "deviation": []}
        if "anomalies" in self.tables:
            for key, text in self._fetch("SELECT key, reasons FROM tables.anomalies"):
                for reason in json.loads(text)["columns"]:
                    rows[KEY].append(key)
                    rows["column"].append(reason["column"])
                    rows["feature"].append(self._feature_of(reason["column"]))
                    rows["value"].append(reason["value"])
                    rows["deviation"].append(reason["deviation"])
        schema = pa.schema([(KEY, pa.string()), ("column", pa.string()), ("feature", pa.string()),
                            ("value", pa.float64()), ("deviation", pa.float64())])
        self.database.connection.execute("CREATE SCHEMA review")
        self.database.load(REASONS, pa.table(rows, schema=schema))

    def _feature_of(self, column: str) -> str:
        """The feature a column of the wide table belongs to: itself, or the name before `__`."""
        if column in self.features:
            return column
        name, _, _ = column.rpartition("__")
        return name if name in self.features else column

    def _label_info(self, feature: str, label: Any) -> tuple[str, bool]:
        """What a label of a per-unit output means, and whether it counts as an event."""
        described = self.outputs.get(feature)
        if described is None or not isinstance(label, str):
            return "", False
        spec = described.spec
        return (spec.labels or {}).get(label, ""), spec.event is not None and label in spec.event.when

    # Queries

    def _fetch(self, sql: str, params: list[Any] | None = None) -> list[tuple[Any, ...]]:
        with self.database.connection.cursor() as cursor:
            try:
                return cursor.execute(sql, params or []).fetchall()
            except duckdb.Error as error:
                raise HTTPException(400, str(error)) from error

    def _rows(self, sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
        with self.database.connection.cursor() as cursor:
            try:
                result = cursor.execute(sql, params or [])
                names = [column[0] for column in result.description or []]
                return [dict(zip(names, row, strict=True)) for row in result.fetchall()]
            except duckdb.Error as error:
                raise HTTPException(400, str(error)) from error

    def _table(self, name: str) -> str:
        return f"tables.{quote(name)}"

    def _has(self, *names: str) -> bool:
        return all(name in self.tables for name in names)

    def _columns(self, table: str) -> dict[str, str]:
        """The columns of a table with their descriptions, for the labels and tooltips of the pages."""
        source = self.database.sources[f"tables.{table}"]
        return {name: self.database.notes.get(name, "") for name in source.columns}

    # Overview

    def groupings(self) -> list[dict[str, Any]]:
        """The columns the overview can group trajectories by: a metadata key, or a feature whose values form
        groups, numbers cut into quartiles."""
        out: list[dict[str, Any]] = []
        counts: dict[str, int] = {}
        for row in self.rows:
            for key, value in row.metadata.items():
                if value is not None and not isinstance(value, (list, dict)):
                    counts[key] = counts.get(key, 0) + 1
        for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            out.append({"name": f"{META}{key}", "label": key, "kind": "metadata",
                        "description": f"The metadata key {key}, set on {count} trajectories."})
        for name, described in self.features.items():
            spec = described.spec
            if name in self.context or spec.type not in ("scalar", "boolean", "category"):
                continue
            cut = " Numbers are cut into quartiles." if spec.type == "scalar" else ""
            out.append({"name": name, "label": name, "kind": "feature", "description": spec.description.strip() + cut})
        return out

    def overview(self, by: list[str] | None) -> dict[str, Any]:
        """The trajectories in groups: how the user and the agent oversaw each other in each, how many of each
        rank among the most anomalous, and the features where each differs most from the other groups."""
        config = self.project.config
        names = tuple(by or config.batches)
        if names in self._overviews:
            return self._overviews[names]
        try:
            group = assign(self.project, self.frame, self.rows, list(names))
        except ConfigError as error:
            raise HTTPException(400, str(error)) from error
        out: dict[str, Any] = {"by": list(names), "configured": list(config.batches), "strata": config.strata,
                               "min_batch": config.alerts.min_batch, "metrics": [], "rows": []}
        metrics: dict[str, str] = {}
        watched: dict[str, dict[str, Any]] = {}
        if self.turns is not None:
            table = oversight(self.turns, self.events, group, config, self.covered)
            metrics = {name: note for name, note in self._columns("oversight").items() if name != "batch"} \
                if self._has("oversight") else dict.fromkeys(c for c in table.columns if c != "batch")
            watched = {row["batch"]: row for row in table.to_dict("records")}
        out["metrics"] = [{"name": name, "description": note or ""} for name, note in metrics.items()]
        unusual = self._unusual()
        differences = self._differences(group)
        for name, count in group.value_counts().items():
            keys = group.index[group == name]
            row = watched.get(name, {})
            out["rows"].append({
                "group": name, "trajectories": int(count),
                "unusual": int(unusual.reindex(keys).fillna(False).sum()) if unusual is not None else None,
                "metrics": {metric: _number(row.get(metric)) for metric in metrics},
                "differences": differences.get(name, []),
            })
        self._overviews[names] = out
        return out

    def _unusual(self) -> pd.Series | None:
        """Per trajectory, whether it ranks in the top tenth of the anomaly scores; None before `traj anomalies`."""
        if not self._has("anomalies"):
            return None
        ranks = pd.Series(dict(self._fetch("SELECT key, anomaly_rank FROM tables.anomalies")))
        return ranks <= max(1, int(round(len(ranks) * TOP_SHARE)))

    def _differences(self, group: pd.Series) -> dict[str, list[dict[str, Any]]]:
        """Per group, the features where it differs most from the other groups: the standardized difference
        inside the strata, averaged over the strata with `alerts.min_batch` trajectories on each side, with the
        plain values of the group and of the others; only columns a person can read as a feature or a label."""
        frame, min_batch = self.frame, self.project.config.alerts.min_batch
        columns = [(column, self.features[feature]) for feature, names in frame.columns.items()
                   if feature in self.features and feature not in self.context
                   for column in names if _readable(column, self.features[feature].spec)]
        scaled = frame.distance[[column for column, _ in columns]].astype(float)
        plain = pd.DataFrame({column: _plain_values(frame, column, described.spec) for column, described in columns})
        out: dict[str, list[dict[str, Any]]] = {}
        for name in group.unique():
            own_all = group == name
            if own_all.sum() < min_batch:
                continue
            found = []
            for column, described in columns:
                weighted, weight = 0.0, 0
                for layer in self.stratum.unique():
                    inside = self.stratum == layer
                    x = scaled.loc[inside & own_all, column].dropna()
                    y = scaled.loc[inside & ~own_all, column].dropna()
                    if len(x) < min_batch or len(y) < min_batch:
                        continue
                    pooled = float(np.sqrt((x.var(ddof=0) + y.var(ddof=0)) / 2))
                    if pooled == 0:
                        continue
                    weighted += float((x.mean() - y.mean()) / pooled) * len(x)
                    weight += len(x)
                if not weight:
                    continue
                own = plain.loc[own_all, column].dropna()
                rest = plain.loc[~own_all, column].dropna()
                found.append({"feature": described.spec.name, "column": column, "effect": round(weighted / weight, 3),
                              "own": _number(float(own.mean())), "others": _number(float(rest.mean())),
                              "n": int(len(own)), "n_others": int(len(rest)),
                              "unit": _unit_of(column, described.spec),
                              "description": described.spec.description.strip()})
            found.sort(key=lambda item: -abs(item["effect"]))
            # One column per feature: an event's count and its unrecovered count tell the same thing.
            first: dict[str, dict[str, Any]] = {}
            for item in found:
                first.setdefault(item["feature"], item)
            out[name] = list(first.values())[:SHIFTS_SHOWN]
        return out

    # Filters

    def filters(self) -> dict[str, Any]:
        """The values the trajectory list can be narrowed by, with what each means."""
        out: dict[str, Any] = {"batches": [], "strata": [], "rules": [], "judged": [], "ended_by": [],
                               "features": [d.info() for d in self.features.values()]}
        if self._has("anomalies"):
            out["batches"] = [b for (b,) in self._fetch("SELECT DISTINCT batch FROM tables.anomalies ORDER BY 1")]
            out["strata"] = [s for (s,) in self._fetch("SELECT DISTINCT stratum FROM tables.anomalies ORDER BY 1")]
        if self._has("events"):
            for rule, count in self._fetch(
                    "SELECT rule, count(*) FROM tables.events WHERE NOT recovered GROUP BY rule ORDER BY 2 DESC"):
                described = self.features.get(rule) or self.outputs.get(rule)
                out["rules"].append({"rule": rule, "unrecovered": count,
                                     "description": described.spec.description.strip() if described else ""})
        if self._has("judgments"):
            for feature, value, count in self._fetch(
                    "SELECT feature, value, count(*) FROM tables.judgments GROUP BY ALL ORDER BY 1, 3 DESC"):
                label = json.loads(value)
                if feature not in self.outputs or not isinstance(label, str):
                    continue
                description, event = self._label_info(feature, label)
                out["judged"].append({"feature": feature, "label": label, "count": count,
                                      "description": description, "event": event})
        if self._has("turns"):
            out["ended_by"] = [{"value": value, "count": count} for value, count in self._fetch(
                "SELECT ended_by, count(*) FROM tables.turns GROUP BY ended_by ORDER BY 2 DESC")]
            out["ended_by_description"] = self._columns("turns")["ended_by"]
        return out

    # Trajectories

    def trajectories(self, request: TrajectoriesRequest) -> dict[str, Any]:
        conditions, params = self._conditions(request)
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        base = "tables.anomalies" if self._has("anomalies") else (
            "(SELECT key, NULL AS batch, NULL AS stratum, NULL AS outlier, NULL AS events_score, "
            "NULL AS anomaly_score, NULL AS anomaly_rank FROM tables.trajectories)")
        turn_stats = ("(SELECT key, count(*) AS turns, sum(duration_s) AS duration_s, sum(asks) AS asks, "
                      "sum(failed_calls) AS failed_calls FROM tables.turns GROUP BY key)" if self._has("turns")
                      else "(SELECT NULL AS key, NULL AS turns, NULL AS duration_s, NULL AS asks, "
                           "NULL AS failed_calls)")
        sql = (f"FROM {base} AS base LEFT JOIN {turn_stats} AS t USING (key) "
               f"LEFT JOIN tables.trajectories AS tr USING (key){where}")
        (total,) = self._fetch(f"SELECT count(*) {sql}", params)[0]
        rows = self._rows(
            f"SELECT base.key, base.batch, base.stratum, base.outlier, base.events_score, base.anomaly_score, "
            f"base.anomaly_rank, t.turns, t.duration_s, t.asks, t.failed_calls, tr.steps {sql} "
            f"ORDER BY {ORDERS[request.sort]}, base.key LIMIT ? OFFSET ?",
            [*params, request.size, (request.page - 1) * request.size])
        keys = [row[KEY] for row in rows]
        reasons = self._reasons_of(keys)
        events = self._events_of(keys)
        judged = self._last_judgments(keys)
        ended = self._ended_by(keys)
        verdicts = self._verdicts_of(keys)
        for row in rows:
            key = row[KEY]
            for name in ("outlier", "events_score", "anomaly_score", "duration_s"):
                row[name] = _number(row[name])
            row.update(reasons=reasons.get(key, []), events=events.get(key, {"shown": [], "counts": {}}),
                       judged=judged.get(key, []), ended_by=ended.get(key, {}), verdicts=verdicts.get(key, []))
        return {"total": total, "page": request.page, "size": request.size,
                "pages": max(1, math.ceil(total / request.size)), "rows": rows}

    def _conditions(self, request: TrajectoriesRequest) -> tuple[list[str], list[Any]]:
        conditions: list[str] = []
        params: list[Any] = []
        if request.group is not None:
            try:
                assigned = assign(self.project, self.frame, self.rows, request.group.by)
            except ConfigError as error:
                raise HTTPException(400, str(error)) from error
            conditions.append("list_contains(?::VARCHAR[], base.key)")
            params.append(sorted(assigned.index[assigned == request.group.value]))
        if request.batch is not None:
            conditions.append("base.batch = ?")
            params.append(request.batch)
        if request.stratum is not None:
            conditions.append("base.stratum = ?")
            params.append(request.stratum)
        if request.rule is not None:
            self._require("events")
            conditions.append("EXISTS (SELECT 1 FROM tables.events e WHERE e.key = base.key AND e.rule = ? "
                              "AND NOT e.recovered)")
            params.append(request.rule)
        if request.judged is not None:
            self._require("judgments")
            conditions.append("EXISTS (SELECT 1 FROM tables.judgments j WHERE j.key = base.key AND j.feature = ? "
                              "AND j.value = ?)")
            params += [request.judged.feature, json.dumps(request.judged.label, ensure_ascii=False)]
        if request.ended_by is not None:
            self._require("turns")
            conditions.append("EXISTS (SELECT 1 FROM tables.turns u WHERE u.key = base.key AND u.ended_by = ?)")
            params.append(request.ended_by)
        if request.feature is not None:
            parts = [f"EXISTS (SELECT 1 FROM {REASONS} r WHERE r.key = base.key AND r.feature = ?)"]
            params.append(request.feature)
            if self._has("events"):
                parts.append("EXISTS (SELECT 1 FROM tables.events e WHERE e.key = base.key AND e.rule = ? "
                             "AND NOT e.recovered)")
                params.append(request.feature)
            conditions.append(f"({' OR '.join(parts)})")
        if request.reviewed is not None:
            reviewed = sorted({v.key for v in read_verdicts(self.project)})
            conditions.append(f"{'' if request.reviewed else 'NOT '}list_contains(?::VARCHAR[], base.key)")
            params.append(reviewed)
        if request.search.strip():
            conditions.append("contains(lower(base.key), lower(?))")
            params.append(request.search.strip())
        return conditions, params

    def _require(self, table: str) -> None:
        if table not in self.tables:
            raise HTTPException(409, f"The {table} table is not computed yet")

    def _reasons_of(self, keys: list[str]) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {}
        for row in self._rows(f"SELECT * FROM {REASONS} WHERE list_contains(?::VARCHAR[], key) "
                              "ORDER BY key, abs(deviation) DESC", [keys]):
            described = self.features.get(row["feature"])
            out.setdefault(row.pop(KEY), []).append({
                **row, "value": _number(row["value"]), "deviation": _number(row["deviation"]),
                "description": described.spec.description.strip() if described else ""})
        return out

    def _events_of(self, keys: list[str]) -> dict[str, dict[str, Any]]:
        """Per trajectory, the first unrecovered events and the counts of events per rule."""
        if not self._has("events"):
            return {}
        out: dict[str, dict[str, Any]] = {}
        for key, rule, total, unrecovered in self._fetch(
                "SELECT key, rule, count(*), count(*) FILTER (WHERE NOT recovered) FROM tables.events "
                "WHERE list_contains(?::VARCHAR[], key) GROUP BY ALL ORDER BY key, 4 DESC", [keys]):
            out.setdefault(key, {"shown": [], "counts": {}})["counts"][rule] = {"total": total,
                                                                                "unrecovered": unrecovered}
        for row in self._rows(
                "SELECT key, rule, step, \"end\", edge, note FROM (SELECT *, row_number() OVER (PARTITION BY key "
                "ORDER BY step) AS own FROM tables.events WHERE list_contains(?::VARCHAR[], key) AND NOT recovered) "
                "WHERE own <= ? ORDER BY key, step", [keys, EVENTS_SHOWN]):
            out.setdefault(row[KEY], {"shown": [], "counts": {}})["shown"].append(
                {k: v for k, v in row.items() if k != KEY})
        return out

    def _last_judgments(self, keys: list[str]) -> dict[str, list[dict[str, Any]]]:
        """Per trajectory, the label of the last unit of every per-unit output."""
        if not self._has("judgments"):
            return {}
        out: dict[str, list[dict[str, Any]]] = {}
        for key, feature, value, start in self._fetch(
                "SELECT key, feature, value, start FROM (SELECT *, row_number() OVER (PARTITION BY key, feature "
                "ORDER BY start DESC) AS own FROM tables.judgments WHERE list_contains(?::VARCHAR[], key)) "
                "WHERE own = 1 ORDER BY key, feature", [keys]):
            label = json.loads(value)
            description, event = self._label_info(feature, label)
            out.setdefault(key, []).append({"feature": feature, "label": label, "step": start, "event": event,
                                            "description": description})
        return out

    def _ended_by(self, keys: list[str]) -> dict[str, dict[str, int]]:
        if not self._has("turns"):
            return {}
        out: dict[str, dict[str, int]] = {}
        for key, value, count in self._fetch(
                "SELECT key, ended_by, count(*) FROM tables.turns WHERE list_contains(?::VARCHAR[], key) "
                "GROUP BY ALL", [keys]):
            out.setdefault(key, {})[value] = count
        return out

    def _verdicts_of(self, keys: list[str]) -> dict[str, list[dict[str, Any]]]:
        wanted = set(keys)
        out: dict[str, list[dict[str, Any]]] = {}
        for verdict in read_verdicts(self.project):
            if verdict.key in wanted:
                out.setdefault(verdict.key, []).append(verdict.model_dump(mode="json"))
        return out

    # One trajectory

    def trajectory(self, key: str) -> dict[str, Any]:
        rows = self._rows("SELECT * FROM tables.trajectories WHERE key = ?", [key])
        if not rows:
            raise HTTPException(404, f"Unknown trajectory {key}")
        header = rows[0]
        header["metadata"] = json.loads(header["metadata"])
        anomaly = self._rows("SELECT * FROM tables.anomalies WHERE key = ?", [key]) if self._has("anomalies") else []
        if anomaly:
            anomaly[0]["reasons"] = json.loads(anomaly[0]["reasons"])
        return {
            "trajectory": header,
            "anomaly": anomaly[0] if anomaly else None,
            "turns": self._course(key, header["steps"]),
            "features": self._feature_values(key, anomaly[0]["batch"] if anomaly else None),
            "verdicts": [v.model_dump(mode="json") for v in read_verdicts(self.project) if v.key == key],
            "outputs": {name: d.info() for name, d in self.outputs.items()},
            "hunts": self._rows(f"SELECT hunt, found, facet, steps, why, round, source, probes FROM "
                                f"tables.{VERDICTS_TABLE} WHERE key = ? ORDER BY hunt", [key])
            if self._has(VERDICTS_TABLE) else [],
        }

    def _course(self, key: str, steps: int) -> list[dict[str, Any]]:
        """The turns of the trajectory, each with the message that opened it, the actions, events, notices and
        judgments that fall in its steps; what precedes the first message forms a turn of its own."""
        turns = self._rows("SELECT * FROM tables.turns WHERE key = ? ORDER BY index", [key]) if self._has("turns") \
            else []
        blocks: list[dict[str, Any]] = []
        if not turns or turns[0]["start"] > 0:
            end = turns[0]["start"] - 1 if turns else steps - 1
            blocks.append({"index": None, "start": 0, "end": end})
        for turn in turns:
            blocks.append({k: _number(v) if isinstance(v, float) else v for k, v in turn.items() if k != KEY})
        for block in blocks:
            block.update(prompt=None, actions=[], events=[], notices=[], judgments=[])
        starts = [block["start"] for block in blocks if block["index"] is not None]
        for step, content in self._fetch(
                "SELECT step, content FROM tables.steps WHERE key = ? AND list_contains(?::BIGINT[], step)",
                [key, starts]):
            _block_at(blocks, step)["prompt"] = content
        if self._has("actions"):
            for row in self._rows("SELECT position, call, ordinal, tool, act, program, subcommand, effect, outcome, "
                                  "error, forced, lines, text, objects FROM tables.actions WHERE key = ? "
                                  "ORDER BY position", [key]):
                _block_at(blocks, row["call"])["actions"].append(row)
        if self._has("events"):
            for row in self._rows("SELECT rule, step, \"end\", edge, objects, recovered, recovered_at, note "
                                  "FROM tables.events WHERE key = ? ORDER BY step", [key]):
                described = self.features.get(row["rule"]) or self.outputs.get(row["rule"])
                row["description"] = described.spec.description.strip() if described else ""
                _block_at(blocks, row["step"])["events"].append(row)
        for row in self._rows("SELECT step, name, content FROM tables.steps WHERE key = ? AND kind = 'notice' "
                              "ORDER BY step", [key]):
            _block_at(blocks, row["step"])["notices"].append(row)
        if self._has("judgments"):
            for row in self._rows("SELECT \"group\", unit, index AS unit_index, start, \"end\", feature, value, "
                                  "evidence FROM tables.judgments WHERE key = ? ORDER BY start, feature", [key]):
                row["value"] = json.loads(row["value"])
                row["description"], row["event"] = self._label_info(row["feature"], row["value"])
                _block_at(blocks, row["start"])["judgments"].append(row)
        return blocks

    def _feature_values(self, key: str, batch: str | None) -> list[dict[str, Any]]:
        """The trajectory's value of every feature it has, with what is typical of its batch, the steps the value
        rests on and whether it is among the anomaly reasons."""
        rows = self._rows(f"SELECT * FROM {FEATURES} WHERE key = ?", [key])
        if not rows:
            return []
        values = {name: value for name, value in rows[0].items() if name in self.features and value is not None}
        types = self.database.sources[FEATURES].columns
        typical = self._typical(list(values), types, batch) if batch is not None else {}
        evidence: dict[str, list[dict[str, Any]]] = {}
        if self._has("evidence"):
            for feature, step, note in self._fetch(
                    "SELECT feature, step, note FROM tables.evidence WHERE key = ? ORDER BY feature, step", [key]):
                evidence.setdefault(feature, []).append({"step": step, "note": note})
        reasons = {row["feature"]: row for row in self._rows(
            f"SELECT feature, \"column\", deviation FROM {REASONS} WHERE key = ? ORDER BY abs(deviation) DESC",
            [key])}
        out = []
        for name, value in values.items():
            described = self.features[name]
            out.append({**described.info(), "value": _plain(value), "typical": typical.get(name),
                        "evidence": evidence.get(name, []), "reason": reasons.get(name)})
        out.sort(key=lambda f: (f["reason"] is None, -abs(f["reason"]["deviation"]) if f["reason"] else 0, f["name"]))
        return out

    def _typical(self, names: list[str], types: dict[str, str], batch: str) -> dict[str, Any]:
        """What the batch typically shows for each feature: the median of a number, the share of a boolean, the
        most common label of a category."""
        parts = []
        for name in names:
            column, dtype = quote(name), types.get(name, "")
            if dtype in NUMERIC:
                parts.append(f"quantile_cont({column}, 0.5) AS {column}")
            elif dtype == "BOOLEAN":
                parts.append(f"avg(CAST({column} AS DOUBLE)) AS {column}")
            elif dtype == "VARCHAR":
                parts.append(f"mode({column}) AS {column}")
        if not parts:
            return {}
        rows = self._rows(f"SELECT {', '.join(parts)} FROM {FEATURES} f JOIN tables.anomalies a USING (key) "
                          "WHERE a.batch = ?", [batch])
        return {name: _plain(value) for name, value in rows[0].items()} if rows else {}

    def steps(self, key: str, start: int, end: int) -> list[dict[str, Any]]:
        if end < start:
            raise HTTPException(400, "end lies before start")
        return self._rows("SELECT step, role, kind, name, is_error, turn, timestamp, chars, content FROM tables.steps "
                          "WHERE key = ? AND step BETWEEN ? AND ? ORDER BY step LIMIT ?",
                          [key, start, end, STEPS_PAGE])

    # Verdicts

    def verdict(self, request: VerdictRequest) -> dict[str, Any]:
        if request.feature not in self.features and request.feature not in self.outputs:
            raise HTTPException(400, f"Unknown feature {request.feature}")
        if not self._fetch("SELECT 1 FROM tables.trajectories WHERE key = ?", [request.key]):
            raise HTTPException(404, f"Unknown trajectory {request.key}")
        verdict = Verdict(**request.model_dump())
        append_verdict(self.project, verdict)
        precision = precision_by_feature(read_verdicts(self.project))[request.feature]
        return {"verdict": verdict.model_dump(mode="json"), "precision": round(precision[0], 4),
                "verdicts": precision[1]}

    # Features

    def feature_cards(self) -> list[dict[str, Any]]:
        """Every feature and per-unit output with how it holds up: its role in the evaluation, its tail precision,
        lift and alert, the verdicts people gave, how often it explains an anomaly, its events and its shifts."""
        cards = self._evaluation_cards()
        verdicts = precision_by_feature(read_verdicts(self.project))
        reasons = dict(self._fetch(f"SELECT feature, count(DISTINCT key) FROM {REASONS} GROUP BY feature"))
        events = {rule: (total, unrecovered) for rule, total, unrecovered in self._fetch(
            "SELECT rule, count(*), count(*) FILTER (WHERE NOT recovered) FROM tables.events GROUP BY rule")} \
            if self._has("events") else {}
        lifts = {row["rule"]: row for row in self._rows("SELECT * FROM tables.lifts")} if self._has("lifts") else {}
        shifts = {row["feature"]: row for row in self._rows(
            "SELECT feature, batch, stratum, effect, direction FROM (SELECT *, row_number() OVER (PARTITION BY "
            "feature ORDER BY abs(effect) DESC) AS own FROM tables.shifts) WHERE own = 1")} \
            if self._has("shifts") else {}
        out = []
        for name, described in [*self.features.items(), *self.outputs.items()]:
            use = described.instance.use
            card = cards.get(name)
            found = verdicts.get(name)
            lift = lifts.get(name)
            shift = shifts.get(name)
            out.append({
                **described.info(),
                "role": card.role if card else None,
                "noise": card.noise if card else None,
                "coverage": card.coverage if card else None,
                "decision": card.decision if card else None,
                "tail_precision": use.tail_precision if use else None,
                "lift": use.lift if use else None,
                "alert": bool(use.alert) if use else False,
                "human_precision": _number(found[0]) if found else None,
                "human_verdicts": found[1] if found else 0,
                "explains": reasons.get(name, 0),
                "events": {"total": events[name][0], "unrecovered": events[name][1]} if name in events else None,
                "consequence_lift": _number(lift["lift"]) if lift else None,
                "consequence_hits": lift["hits"] if lift else None,
                "shift": {"batch": shift["batch"], "stratum": shift["stratum"], "effect": _number(shift["effect"]),
                          "direction": shift["direction"]} if shift else None,
            })
        return out

    def _evaluation_cards(self) -> dict[str, Card]:
        if not (self.project.evaluation_dir / EVALUATION_FILE).is_file():
            return {}
        return {card.feature: card for card in load_evaluation(self.project).features}

    # Hunts

    def _load_hunts(self) -> None:
        """Read what the last finished run of every hunt found, when the server starts, so the summary of a hunt and
        the rows under it come from the one snapshot the pages show."""
        self._hunts: dict[str, dict[str, Any]] = {}
        for name in hunt_names(self.project):
            need = load_hunt(self.project, name).need.strip()
            found = summarize_run(self.project, name) if finished_runs(self.project, name) else {"run": None}
            self._hunts[name] = {**found, "hunt": name, "need": need}

    def hunts(self) -> list[dict[str, Any]]:
        """Every hunt of the project with what its last finished run found; a hunt without one has no run."""
        out = []
        for name, found in self._hunts.items():
            if found["run"] is None:
                out.append({"hunt": name, "need": found["need"], "run": None})
                continue
            out.append({key: found[key] for key in ("hunt", "need", "run", "rounds", "converged", "found",
                                                    "coverage")}
                       | {"facets": len(found["facets"]), "probes": len(found["probes"])})
        return out

    def hunt(self, name: str) -> dict[str, Any]:
        """What the last finished run of a hunt found, and every trajectory it read or its marks point to: its
        verdict, where the marks of the probes it kept are, and its batch."""
        found = self._hunts.get(name)
        if found is None:
            raise HTTPException(404, f"Unknown hunt {name}")
        if found["run"] is None:
            return {"hunt": name, "need": found["need"], "run": None, "trajectories": []}
        facets = found["marks"]["features"]
        rows: dict[str, dict[str, Any]] = {}
        if self._has(VERDICTS_TABLE):
            for row in self._rows(f"SELECT key, found, facet, steps, why, round, source FROM tables.{VERDICTS_TABLE} "
                                  "WHERE hunt = ?", [name]):
                rows[row[KEY]] = {**row, "status": "found" if row["found"] else "rejected", "marks": []}
        if self._has("events"):
            for row in self._rows("SELECT key, rule, step, note FROM tables.events WHERE list_contains(?::VARCHAR[], "
                                  "rule) ORDER BY key, step", [list(facets)]):
                entry = rows.setdefault(row[KEY], {"key": row[KEY], "found": None, "facet": None, "steps": [],
                                                   "why": "", "round": None, "source": None, "status": "unread",
                                                   "marks": []})
                entry["marks"].append({"feature": row["rule"], "facet": facets[row["rule"]], "step": row["step"],
                                       "note": row["note"]})
        order = {"found": 0, "unread": 1, "rejected": 2}
        trajectories = sorted(rows.values(), key=lambda row: (order[row["status"]], row[KEY]))
        for row in trajectories:
            row["batch"] = self.batch.get(row[KEY])
            row["stratum"] = self.stratum.get(row[KEY])
        return {**found, "trajectories": trajectories}


def _plain_values(frame: Frame, column: str, spec: FeatureSpec) -> pd.Series:
    """A column in the units a person reads: the feature's own values for a number or a boolean, the 0 or 1 of a
    label and the count of an event's unrecovered events as the wide table holds them."""
    if column == spec.name and spec.type in ("scalar", "boolean"):
        return pd.to_numeric(frame.query[column], errors="coerce")
    return pd.to_numeric(frame.distance[column], errors="coerce")


def _unit_of(column: str, spec: FeatureSpec) -> str:
    """How a column's mean reads: `share` for a boolean, a label or a feature ranging from 0 to 1, else `number`."""
    if column != spec.name or spec.type == "boolean" or spec.range == (0, 1):
        return "share" if column != f"{spec.name}__unrecovered" else "number"
    return "number"


def _readable(column: str, spec: FeatureSpec) -> bool:
    """Whether a column of the wide table reads as the feature or one of its labels: the feature's own column, a
    label of a category or set, or the unrecovered count of an event; the nodes of paths and the positions of
    vectors do not."""
    if column == spec.name:
        return True
    suffix = column[len(spec.name) + 2:] if column.startswith(f"{spec.name}__") else None
    if suffix is None:
        return False
    if spec.type in ("category", "set"):
        return suffix in (spec.labels or {})
    return spec.type == "event" and suffix == "unrecovered"


def _block_at(blocks: list[dict[str, Any]], step: int) -> dict[str, Any]:
    """The turn whose steps hold this step; a step past the last turn's end belongs to the last turn."""
    for block in blocks:
        if block["start"] <= step <= block["end"]:
            return block
    return blocks[-1]


def _number(value: Any) -> Any:
    if isinstance(value, float):
        return None if math.isnan(value) else round(value, 4)
    return value


def _plain(value: Any) -> Any:
    """A value of the wide table as JSON allows it."""
    if isinstance(value, float):
        return _number(value)
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value
