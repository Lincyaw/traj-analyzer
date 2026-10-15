from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
from fastapi.testclient import TestClient
from test_anomalies import ensure_turns
from test_pipeline import edit_config
from test_tables import project  # noqa: F401, F811

from traj_analyzer.anomalies import run_anomalies
from traj_analyzer.extract import extract
from traj_analyzer.ingest import load_index
from traj_analyzer.layers import read_table, table_path
from traj_analyzer.project import Project
from traj_analyzer.viewer.server import create_app


def plant_events(target: Project) -> tuple[str, str]:
    """Write one unrecovered and one recovered blind_retry event, since the conversations of the fixture break no
    expectation; return the keys of the two trajectories."""
    first, second = [row.key for row in load_index(target)[:2]]
    table = read_table(target, "events")
    rows = [
        {"key": first, "rule": "blind_retry", "step": 2, "end": 3, "edge": "tool", "objects": ["x.py"],
         "recovered": False, "recovered_at": None, "note": "repeated unchanged after a failure: read x.py"},
        {"key": second, "rule": "blind_retry", "step": 1, "end": 1, "edge": "tool", "objects": [],
         "recovered": True, "recovered_at": 3, "note": "repeated unchanged after a failure: ls"},
    ]
    pq.write_table(pa.concat_tables([table, pa.Table.from_pylist(rows, schema=table.schema)]),
                   table_path(target, "events"))
    return first, second


def test_review_pages_read_the_tables_and_record_verdicts(project: Project) -> None:  # noqa: F811
    # A few trajectories per batch are enough for the groups of the fixture to be compared.
    edit_config(project.root, lambda config: config.setdefault("alerts", {}).update({"min_batch": 5}))
    project = Project.load(project.root)
    extract(project)
    ensure_turns(project)
    broken, recovered = plant_events(project)
    # Before `traj anomalies`, the overview counts no unusual trajectories and the list holds every trajectory.
    early = TestClient(create_app(project))
    assert all(row["unusual"] is None for row in early.get("/api/review/overview").json()["rows"])
    plain = early.post("/api/review/trajectories", json={"size": 5}).json()
    assert plain["total"] == 70 and plain["rows"][0]["anomaly_rank"] is None

    run_anomalies(project)
    client = TestClient(create_app(project))
    assert client.get("/").status_code == 200 and client.get("/data.html").status_code == 200
    overview = client.get("/api/review/overview").json()
    assert len(overview["rows"]) > 1 and overview["by"] == overview["configured"] == ["meta.subset_name"]
    assert {m["name"] for m in overview["metrics"]} >= {"turns", "ask_share", "interrupt_share", "calls_per_turn"}
    assert all(row["trajectories"] > 0 and set(row["metrics"]) == {m["name"] for m in overview["metrics"]}
               for row in overview["rows"])
    assert sum(row["unusual"] for row in overview["rows"]) == 7
    largest = max(overview["rows"], key=lambda row: row["trajectories"])
    assert largest["differences"] and all({"column", "own", "others", "effect", "unit"} <= set(d)
                                          for d in largest["differences"])
    groupings = {g["name"]: g for g in client.get("/api/review/groupings").json()}
    assert groupings["meta.subset_name"]["kind"] == "metadata" and groupings["n_steps"]["kind"] == "feature"
    by_steps = client.get("/api/review/overview", params={"by": ["n_steps"]}).json()
    assert by_steps["by"] == ["n_steps"] and {row["group"] for row in by_steps["rows"]} <= {
        "n_steps:q1", "n_steps:q2", "n_steps:q3", "n_steps:q4"}
    assert sum(row["trajectories"] for row in by_steps["rows"]) == 70
    assert client.get("/api/review/overview", params={"by": ["meta.nope", "x"]}).status_code == 400
    grouped = client.post("/api/review/trajectories", json={"size": 100, "group": {"by": ["n_steps"], "value":
                                                                                    by_steps["rows"][0]["group"]}}).json()
    assert grouped["total"] == by_steps["rows"][0]["trajectories"]
    filters = client.get("/api/review/filters").json()
    assert set(filters["batches"]) == {row["group"] for row in overview["rows"]}
    assert filters["ended_by"] and filters["ended_by_description"].startswith("How the turn ended")
    assert {f["name"] for f in filters["features"]} >= {"n_steps", "blind_retry"}

    listing = client.post("/api/review/trajectories", json={"size": 10}).json()
    assert (listing["total"], listing["pages"], len(listing["rows"])) == (70, 7, 10)
    first = listing["rows"][0]
    assert first["anomaly_rank"] == 1 and {"reasons", "events", "judged", "ended_by", "verdicts"} <= set(first)
    assert first["reasons"][0]["description"] and first["reasons"][0]["feature"]
    narrowed = client.post("/api/review/trajectories", json={"size": 10, "batch": first["batch"]}).json()
    assert 0 < narrowed["total"] < listing["total"] and all(r["batch"] == first["batch"] for r in narrowed["rows"])
    assert [(r["rule"], r["unrecovered"]) for r in filters["rules"]] == [("blind_retry", 1)]
    assert filters["rules"][0]["description"].startswith("After a failed action")
    with_rule = client.post("/api/review/trajectories", json={"size": 50, "rule": "blind_retry"}).json()
    assert with_rule["total"] == 1 and with_rule["rows"][0]["key"] == broken
    assert with_rule["rows"][0]["events"] == {
        "shown": [{"rule": "blind_retry", "step": 2, "end": 3, "edge": "tool",
                   "note": "repeated unchanged after a failure: read x.py"}],
        "counts": {"blind_retry": {"total": 1, "unrecovered": 1}}}
    both = client.post("/api/review/trajectories", json={"size": 50, "feature": "blind_retry"}).json()
    assert {r["key"] for r in both["rows"]} >= {broken}
    course = client.get("/api/review/trajectory", params={"key": recovered}).json()["turns"]
    assert [(e["rule"], e["recovered"], e["recovered_at"]) for t in course for e in t["events"]] == [
        ("blind_retry", True, 3)]
    reason = first["reasons"][0]["feature"]
    explained = client.post("/api/review/trajectories", json={"size": 50, "feature": reason}).json()
    assert first["key"] in {r["key"] for r in explained["rows"]}
    assert client.post("/api/review/trajectories", json={"size": 5, "search": first["key"][-6:]}).json()["total"] >= 1
    assert client.post("/api/review/trajectories", json={"size": 5, "sort": "sideways"}).status_code == 422

    detail = client.get("/api/review/trajectory", params={"key": first["key"]}).json()
    assert detail["trajectory"]["key"] == first["key"] and detail["anomaly"]["anomaly_rank"] == 1
    assert detail["turns"]
    assert all({"actions", "events", "judgments", "notices", "prompt"} <= set(t) for t in detail["turns"])
    assert detail["turns"][-1]["end"] == detail["trajectory"]["steps"] - 1
    assert [f["name"] for f in detail["features"] if f["reason"] is not None][0] == reason
    assert all("typical" in f and "evidence" in f for f in detail["features"])
    steps = client.get("/api/review/steps", params={"key": first["key"], "start": 0, "end": 3}).json()
    assert [s["step"] for s in steps] == list(range(len(steps))) and steps[0]["content"]
    assert client.get("/api/review/steps", params={"key": first["key"], "start": 3, "end": 0}).status_code == 400
    assert client.get("/api/review/trajectory", params={"key": "toucan/none"}).status_code == 404

    answer = client.post("/api/review/verdict", json={"key": first["key"], "feature": reason, "step": 2, "worth": True,
                                                     "note": "seen"}).json()
    assert (answer["precision"], answer["verdicts"], answer["verdict"]["note"]) == (1.0, 1, "seen")
    assert project.reviews_path.is_file()
    reviewed = client.post("/api/review/trajectories", json={"size": 10, "reviewed": True}).json()
    assert reviewed["total"] == 1 and reviewed["rows"][0]["verdicts"][0]["feature"] == reason
    unreviewed = client.post("/api/review/trajectories", json={"size": 10, "reviewed": False}).json()
    assert unreviewed["total"] == 69
    unknown = client.post("/api/review/verdict", json={"key": first["key"], "feature": "nope", "worth": False})
    assert unknown.status_code == 400
    missing = client.post("/api/review/verdict", json={"key": "toucan/none", "feature": reason, "worth": False})
    assert missing.status_code == 404

    cards = {c["name"]: c for c in client.get("/api/review/features").json()}
    assert (cards[reason]["human_verdicts"], cards[reason]["human_precision"]) == (1, 1.0)
    assert cards[reason]["explains"] >= 1
    assert cards["blind_retry"]["events"] == {"total": 2, "unrecovered": 1} and cards["blind_retry"]["kind"] == "code"
    columns = {"role", "noise", "tail_precision", "lift", "alert", "consequence_lift", "shift"}
    assert all(columns <= set(c) for c in cards.values())
