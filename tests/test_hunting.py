from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest
from aifn import Worker
from aifn.engines import ReplayEngine
from pydantic import ValidationError
from ruamel.yaml import YAML
from test_mining import call_key
from test_pipeline import DATA, TOUCAN, ULTRACHAT, ULTRACHAT_DESCRIPTION, edit_config

from traj_analyzer.adapters.messages import MessagesAdapter
from traj_analyzer.calls import WorkspaceCalls
from traj_analyzer.cli import main
from traj_analyzer.extract import SubprocessWorkers, extract
from traj_analyzer.hunting import FINDINGS_TABLE, hunt, load_hunt
from traj_analyzer.hunting.draw import apportion, clusters_of, draw
from traj_analyzer.hunting.driver import apply, keep_probes, start, write_findings
from traj_analyzer.hunting.functions import (
    NEW,
    CaseVerdict,
    Facet,
    Probe,
    Proposal,
    Verification,
    functions,
    hunt_functions,
    proposal_model,
)
from traj_analyzer.hunting.ledger import (
    HuntState,
    ProbeState,
    Reading,
    coverage,
    new_findings,
    probe_ledger,
    unproductive,
)
from traj_analyzer.hunting.probes import run_probes
from traj_analyzer.hunting.workspace import prepare, write_case
from traj_analyzer.ingest import ingest, load_index
from traj_analyzer.operators.catalog import load_groups
from traj_analyzer.project import ConfigError, Project
from traj_analyzer.runtime import mailbox, policy
from traj_analyzer.vectorize import Matrix, build_frame
from traj_analyzer.worker import registry

REPLAY = DATA / "hunting" / "replay"
NEED = "Find the trajectories where a tool call fails and the assistant answers as if it had worked."
WEATHER_SQL = """\
SELECT key, step, name AS note FROM 'tables/contents.parquet'
WHERE kind = 'tool_call' AND regexp_matches(name, 'weather')"""
TOUCAN_LAST_SQL = """\
SELECT key, max(step) AS step, 'last step' AS note FROM 'tables/steps.parquet'
WHERE key LIKE 'toucan/%' GROUP BY key"""
REAL_RUN = {"reads_per_round": 12, "cases_per_call": 6, "max_rounds": 4, "patience": 2, "clusters": 6,
            "probes_max": 6, "min_reads_per_probe": 2, "final_reads": 6}
"""The hunt of the recorded real run, small enough for a test."""
REAL_MODEL = "deepseek-flash"
"""The model that answered the recorded real run."""


def hunting_project(root: Path, **hunt_options: Any) -> Project:
    """A project over the 30 Toucan and 40 UltraChat conversations with code features and one hunt, `misuse`."""
    assert main(["init", str(root)]) == 0

    def configure(config: dict[str, Any]) -> None:
        config["datasets"] = {"toucan": TOUCAN, "ultrachat": {**ULTRACHAT, "description": ULTRACHAT_DESCRIPTION}}
        config["operators"] = [{"use": "stats.basic"}, {"use": "stats.tool_usage"}]
        config["calls"] = {}

    edit_config(root, configure)
    project = Project.load(root)
    assert ingest(project) == {"toucan": 30, "ultrachat": 40}
    extract(project)
    project.hunts_dir.mkdir()
    YAML(typ="safe").dump({"need": NEED, **hunt_options}, project.hunts_dir / "misuse.yaml")
    return project


def real_run_project(root: Path) -> Project:
    """The project of the recorded real run: the hunt of `REAL_RUN`, answered by `REAL_MODEL`."""
    hunting_project(root, **REAL_RUN)
    edit_config(root, lambda config: config["engine"].update(model=REAL_MODEL))
    return Project.load(root)


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Project:
    project = hunting_project(tmp_path / "proj")
    monkeypatch.chdir(project.root)
    return project


def workspace_calls(project: Project, workspace: Path) -> WorkspaceCalls:
    spec = load_hunt(project, "misuse")
    box = mailbox(project)
    return WorkspaceCalls(project=project, runner=SubprocessWorkers(), workers=None, box=box, known={},
                          functions=hunt_functions(project, "misuse", spec), workspace=workspace)


def frame_of(project: Project) -> tuple[list[str], Matrix]:
    rows = sorted(load_index(project), key=lambda row: row.key)
    frame = build_frame(project, load_groups(project), rows, project.config.sampling.vector_length)
    return [row.key for row in rows], frame.matrix(dict.fromkeys(frame.columns, 1.0))


def weather_positions() -> set[tuple[str, int]]:
    """The tool calls of the weather tools in the Toucan data, read with the adapter itself."""
    return {(f"toucan/{trajectory.id}", step.index)
            for trajectory in MessagesAdapter(**TOUCAN["options"]).read(DATA / "toucan.jsonl", "toucan")
            for step in trajectory.steps if step.kind == "tool_call" and "weather" in (step.name or "")}


def probe(name: str, kind: str = "sql", text: str = WEATHER_SQL, **fields: Any) -> ProbeState:
    return ProbeState(**{"name": name, "facet": "false_success", "kind": kind, "text": text, "why": "test",
                         "since": 1, **fields})


def reading(key: str, found: bool, source: str = "hit", facet: str = "false_success", round_: int = 1) -> Reading:
    return Reading(key=key, round=round_, source=source, found=found, facet=facet if found else None,
                   steps=[3] if found else [], why="test")


def test_hunt_specs_are_checked(project: Project) -> None:
    spec = load_hunt(project, "misuse")
    assert (spec.patience, spec.max_rounds, spec.reads_per_round, spec.datasets) == (3, 12, 60, "all")
    with pytest.raises(ConfigError, match="must match"):
        load_hunt(project, "Misuse")
    with pytest.raises(ConfigError, match="No hunt"):
        load_hunt(project, "other")
    YAML(typ="safe").dump({"need": NEED, "datasets": ["sharegpt"]}, project.hunts_dir / "other.yaml")
    with pytest.raises(ConfigError, match="does not configure"):
        load_hunt(project, "other")
    YAML(typ="safe").dump({"need": NEED, "patience": 5, "max_rounds": 4}, project.hunts_dir / "other.yaml")
    with pytest.raises(ValidationError, match="patience is larger than max_rounds"):
        load_hunt(project, "other")
    YAML(typ="safe").dump({"need": NEED, "rounds": 5}, project.hunts_dir / "other.yaml")
    with pytest.raises(ValidationError, match="rounds"):
        load_hunt(project, "other")


def test_every_hunt_declares_its_functions_for_the_workers(project: Project) -> None:
    names = sorted(functions(project))
    assert names == ["hunt-propose-misuse", "hunt-verify-misuse"]
    assert set(names) <= set(registry(project).functions)
    assert functions(project)["hunt-propose-misuse"].model.name == project.config.engine.model


def test_workspace_holds_the_tables_and_the_whole_text(project: Project, tmp_path: Path) -> None:
    calls = workspace_calls(project, tmp_path / "ws")
    keys = [row.key for row in load_index(project)]
    prepare(calls, load_hunt(project, "misuse"), keys)
    workspace = calls.workspace
    assert (workspace / "need.md").read_text(encoding="utf-8").strip() == NEED
    assert len(list((workspace / "trajectories").glob("*/*.md"))) == 70
    for name in ("README.md", "query.py", "operators/rules.py", "operators/example_operator.py"):
        assert (workspace / name).is_file()
    assert "tables/contents.parquet" in (workspace / "README.md").read_text(encoding="utf-8")
    # The steps table and the trajectory files cut a tool result of 26177 characters; contents and cases hold it.
    key = "toucan/cde1f5c1-0033-5e1c-9473-ab863857b47c"
    steps = pq.read_table(workspace / "tables" / "steps.parquet").to_pandas()
    contents = pq.read_table(workspace / "tables" / "contents.parquet").to_pandas()
    cut = steps[(steps["key"] == key) & (steps["step"] == 9)].iloc[0]
    whole = contents[(contents["key"] == key) & (contents["step"] == 9)].iloc[0]
    assert (cut["chars"], len(cut["content"]), len(whole["content"])) == (26177, 4000, 26177)
    case = write_case(calls, "rounds/1/cases", key)
    assert case == f"rounds/1/cases/{key}.md"
    text = (workspace / case).read_text(encoding="utf-8")
    assert whole["content"] in text and "truncated" not in text
    assert "truncated" in project.trajectory_path(key, ".md").read_text(encoding="utf-8")
    # Toucan calls hold their arguments as parsed by the adapter.
    calls_rows = contents[(contents["kind"] == "tool_call") & contents["key"].str.startswith("toucan/")]
    assert calls_rows["arguments"].map(lambda text: isinstance(json.loads(text), dict)).all()


def test_sql_and_code_probes_find_the_same_positions(project: Project, tmp_path: Path) -> None:
    calls = workspace_calls(project, tmp_path / "ws")
    keys = [row.key for row in load_index(project)]
    prepare(calls, load_hunt(project, "misuse"), keys)
    code = (DATA / "hunting" / "weather_calls.py").read_text(encoding="utf-8")
    probes = [
        probe("weather_sql"),
        probe("weather_calls", kind="code", text=code),
        probe("columns_wrong", text="SELECT key, step FROM 'tables/steps.parquet'"),
        probe("two_statements", text="SELECT 1 AS key, 2 AS step, 'x' AS note; SELECT 2"),
        probe("writes", text="COPY (SELECT 1) TO 'out.csv'"),
        probe("unknown_keys", text="SELECT 'toucan/none' AS key, 1 AS step, '' AS note"),
        probe("named_otherwise", kind="code", text=code.replace('"weather_calls"', '"rain_calls"')),
    ]
    for state in probes:
        calls.hand_off(f"probes/{state.name}.{'sql' if state.kind == 'sql' else 'py'}", state.text)
    hits, failed = run_probes(project, calls.workspace, probes, set(keys))
    assert sorted(failed) == ["columns_wrong", "named_otherwise", "two_statements", "unknown_keys", "writes"]
    assert "returns the columns" in failed["columns_wrong"] and "one SELECT" in failed["two_statements"]
    assert "one SELECT" in failed["writes"] and "toucan/none" in failed["unknown_keys"]
    assert "named named_otherwise; this one has event rain_calls" in failed["named_otherwise"]
    expected = weather_positions()
    for name in ("weather_sql", "weather_calls"):
        own = hits[hits["probe"] == name]
        assert set(zip(own["key"], own["step"], strict=True)) == expected
    assert len(expected) == 19 and len({key for key, _ in expected}) == 6


def test_draws_read_every_probe_and_explore_what_no_probe_hits(project: Project, tmp_path: Path) -> None:
    calls = workspace_calls(project, tmp_path / "ws")
    keys, matrix = frame_of(project)
    prepare(calls, load_hunt(project, "misuse"), keys)
    toucan = [key for key in keys if key.startswith("toucan/")]
    calls.hand_off("probes/weather.sql", WEATHER_SQL)
    calls.hand_off("probes/toucan_last.sql", TOUCAN_LAST_SQL)
    probes = [probe("weather"), probe("toucan_last", text=TOUCAN_LAST_SQL)]
    hits, failed = run_probes(project, calls.workspace, probes, set(keys))
    assert not failed
    weather = sorted({key for key, _ in weather_positions()})
    state = HuntState(probes=probes, hits={"weather": weather, "toucan_last": toucan},
                      readings={weather[0]: reading(weather[0], True), weather[1]: reading(weather[1], False)})
    spec = load_hunt(project, "misuse").model_copy(update={"reads_per_round": 20, "hit_share": 0.5,
                                                           "clusters": 8})
    clusters = clusters_of(keys, matrix, spec.clusters, spec.seed)
    drawn = draw(spec, state, hits, keys, matrix, clusters, 1)
    assert drawn == draw(spec, state, hits, keys, matrix, clusters, 1)
    chosen = [d.key for d in drawn]
    assert len(chosen) == len(set(chosen)) == 20 and not set(chosen) & set(state.readings)
    by_source = {source: [d for d in drawn if d.source == source] for source in ("hit", "coverage", "random")}
    assert [len(by_source[source]) for source in ("hit", "coverage", "random")] == [10, 5, 5]
    # Every probe with unread hits is read at least min_reads_per_probe times.
    assert sum(any(h.probe == "weather" for h in d.hits) for d in by_source["hit"]) >= 3
    assert all(d.hits and d.key.startswith("toucan/") for d in by_source["hit"])
    assert all(not d.hits and d.key.startswith("ultrachat/") for d in [*by_source["coverage"], *by_source["random"]])
    # The coverage reads go to different clusters, those read least, as far as the unhit trajectories reach.
    reachable = {clusters[key] for key in keys if key.startswith("ultrachat/")}
    assert len({clusters[d.key] for d in by_source["coverage"]}) == min(5, len(reachable))
    # With every hit read, the hit reads go to exploration.
    done = state.model_copy(update={"readings": {key: reading(key, False) for key in toucan}})
    assert {d.source for d in draw(spec, done, hits, keys, matrix, clusters, 2)} == {"coverage", "random"}


def test_apportion_follows_the_weights() -> None:
    assert apportion(7, {"a": 10, "b": 5, "c": 0}) == {"a": 5, "b": 2, "c": 0}
    assert apportion(10, {"a": 2, "b": 3}) == {"a": 2, "b": 3}
    assert apportion(0, {"a": 2}) == {}
    assert sum(apportion(5, {"a": 1, "b": 1, "c": 1}).values()) == 3


def test_ledger_counts_precision_misses_and_new_findings(project: Project) -> None:
    spec = load_hunt(project, "misuse")
    keys, matrix = frame_of(project)
    clusters = clusters_of(keys, matrix, spec.clusters, spec.seed)
    toucan = [key for key in keys if key.startswith("toucan/")]
    ultrachat = [key for key in keys if key.startswith("ultrachat/")]
    readings = {key: reading(key, index < 3) for index, key in enumerate(toucan[:10])}
    readings.update({key: reading(key, False, source="random") for key in ultrachat[:36]})
    state = HuntState(probes=[probe("broad"), probe("narrow"), probe("idle", status="retired")],
                      hits={"broad": toucan, "narrow": toucan[:2]}, readings=readings)
    rows = {row["name"]: row for row in probe_ledger(state, spec)}
    assert (rows["broad"]["read"], rows["broad"]["found"], rows["broad"]["precision"]) == (10, 3, 0.3)
    assert rows["broad"]["interval"] == [0.1078, 0.6032] and rows["broad"]["revise"]
    assert (rows["broad"]["unread"], rows["broad"]["expected_unverified"], rows["broad"]["only_hits"]) == (20, 6.0, 28)
    assert (rows["narrow"]["precision"], rows["narrow"]["revise"]) == (1.0, False)
    assert rows["idle"]["hits"] == 0 and rows["idle"]["precision"] is None
    found = coverage(state, keys, clusters)
    # 36 random reads of trajectories no probe hits, none of them real: the share missed is below 8% at 95%.
    assert (found["random_reads_unhit"], found["random_misses"], found["miss_share_upper"]) == (36, 0, 0.0798)
    assert (found["read"], found["unread_unhit"], found["expected_missed"]) == (46, 4, 0.32)
    assert not unproductive(state, spec)
    cold = state.model_copy(update={"readings": {key: reading(key, False) for key in toucan[:6]}})
    assert unproductive(cold, spec) == {"broad": 6}

    before = HuntState(readings={toucan[0]: reading(toucan[0], True)})
    later = [reading(toucan[1], True, round_=2), reading(toucan[2], True, facet="fake_test", round_=2),
             reading(ultrachat[0], True, source="coverage", round_=2), reading(ultrachat[1], False, source="random")]
    new = new_findings(before, later)
    assert len(new) == 2 and "facet fake_test first confirmed" in new[0] and "no probe hits it" in new[1]
    assert new_findings(before, [reading(toucan[1], True, round_=2)]) == []


def test_proposals_keep_the_counts_of_unchanged_probes(project: Project) -> None:
    facets = [Facet(name="false_success", definition="The assistant reports success after a failed call.")]
    state = HuntState(facets=[*facets, Facet(name="fake_test", definition="A test is changed to pass.")],
                      probes=[probe("kept"), probe("changed"), probe("dropped"), probe("cold", status="unproductive"),
                              probe("gone", status="retired", reason="earlier")],
                      readings={"toucan/a": reading("toucan/a", True, facet="fake_test")})
    proposal = Proposal(facets=facets, probes=[
        Probe(name="kept", facet="false_success", kind="sql", text=WEATHER_SQL, why="still good"),
        Probe(name="changed", facet="false_success", kind="sql", text=WEATHER_SQL + " LIMIT 5", why="tighter"),
        Probe(name="cold", facet="false_success", kind="sql", text=WEATHER_SQL, why="again"),
        Probe(name="fresh", facet="false_success", kind="sql", text=WEATHER_SQL, why="new"),
    ], retired=[{"probe": "dropped", "reason": "too broad"}])
    apply(state, proposal, 3)
    probes = {p.name: p for p in state.probes}
    assert (probes["kept"].since, probes["kept"].status, probes["kept"].why) == (1, "active", "still good")
    assert (probes["changed"].since, probes["fresh"].since) == (3, 3)
    assert (probes["dropped"].status, probes["dropped"].reason) == ("retired", "too broad")
    assert probes["cold"].status == "unproductive" and probes["gone"].reason == "earlier"
    # A facet with a confirmed trajectory stays though the proposal leaves it out.
    assert [facet.name for facet in state.facets] == ["false_success", "fake_test"]


def test_proposals_and_verdicts_are_checked() -> None:
    facet = {"name": "false_success", "definition": "The assistant reports success after a failed call."}
    sql = {"name": "weather", "facet": "false_success", "kind": "sql", "text": WEATHER_SQL, "why": "w"}
    limited = proposal_model(1)
    limited(facets=[facet], probes=[sql])
    with pytest.raises(ValidationError, match="at most 1"):
        limited(facets=[facet], probes=[sql, {**sql, "name": "other"}])
    with pytest.raises(ValidationError, match="share a name"):
        Proposal(facets=[facet], probes=[sql, sql])
    with pytest.raises(ValidationError, match="facets that `facets` lacks"):
        Proposal(facets=[facet], probes=[{**sql, "facet": "other"}])
    with pytest.raises(ValidationError, match="reserved"):
        Proposal(facets=[{**facet, "name": NEW}], probes=[])
    with pytest.raises(ValidationError, match="both returned and retired"):
        Proposal(facets=[facet], probes=[sql], retired=[{"probe": "weather", "reason": "r"}])
    CaseVerdict(file="a.md", found=True, facet="false_success", steps=[4], why="w")
    CaseVerdict(file="a.md", found=True, facet=NEW, new_facet=facet, steps=[4], why="w")
    CaseVerdict(file="a.md", found=False, why="w")
    with pytest.raises(ValidationError, match="names its facet and its steps"):
        CaseVerdict(file="a.md", found=True, facet="false_success", why="w")
    with pytest.raises(ValidationError, match="exactly when"):
        CaseVerdict(file="a.md", found=True, facet=NEW, steps=[4], why="w")
    with pytest.raises(ValidationError, match="has no facet"):
        CaseVerdict(file="a.md", found=False, facet="false_success", why="w")
    with pytest.raises(ValidationError, match="same file"):
        Verification(verdicts=[{"file": "a.md", "found": False, "why": "w"}] * 2)


def test_findings_and_kept_probes_carry_over_to_the_next_run(project: Project) -> None:
    state = HuntState(
        facets=[Facet(name="false_success", definition="The assistant reports success after a failed call.")],
        probes=[probe("weather"), probe("idle", status="retired")], hits={"weather": ["toucan/a", "toucan/b"]},
        readings={"toucan/a": reading("toucan/a", True), "toucan/b": reading("toucan/b", False)})
    write_findings(project, "misuse", list(state.readings.values()))
    write_findings(project, "other", [reading("toucan/c", True)])
    write_findings(project, "misuse", list(state.readings.values()))
    table = pq.read_table(project.tables_dir / f"{FINDINGS_TABLE}.parquet").to_pandas()
    assert sorted(zip(table["hunt"], table["key"], strict=True)) == [("misuse", "toucan/a"), ("other", "toucan/c")]
    assert table.set_index("key").loc["toucan/a", "step"] == 3
    keep_probes(project, "misuse", state, probe_ledger(state, load_hunt(project, "misuse")))
    kept = project.hunts_dir / "misuse"
    assert sorted(path.name for path in kept.iterdir()) == ["facets.yaml", "probes.yaml", "weather.sql"]
    again = start(project, "misuse")
    assert [(p.name, p.text, p.since, p.status) for p in again.probes] == [("weather", WEATHER_SQL, 0, "active")]
    assert again.facets == state.facets and not again.readings


def test_hunt_needs_the_engine_credentials_and_the_tables(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="DEEPSEEK_API_KEY"):
        hunt(project, "misuse")
    # The check reads only whether the variable is set; no model is called before the tables are found missing.
    monkeypatch.setenv("DEEPSEEK_API_KEY", "unused")
    shutil.rmtree(project.tables_dir)
    with pytest.raises(ConfigError, match="No tables yet"):
        hunt(project, "misuse")


class HuntReplay:
    """Answer every pending call with the submission a real model run gave to the same call."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def check(self, project: Project) -> None:
        pass

    def run(self, project: Project, count: int, functions: frozenset[str] = frozenset()) -> None:
        box = mailbox(project)
        self.directory.mkdir(parents=True, exist_ok=True)
        for call in box.pending():
            shutil.copytree(REPLAY / call_key(call), self.directory / call.id)
        Worker(mailbox=box, registry=registry(project), engine=ReplayEngine(self.directory),
               policy=policy(project), id="replay").work()


class NoCalls:
    """A runner for a run whose every call is already answered in the mailbox."""

    def check(self, project: Project) -> None:
        pass

    def run(self, project: Project, count: int, functions: frozenset[str] = frozenset()) -> None:
        raise AssertionError("a call was pending, so the run did not continue from its state")


def test_hunt_replays_a_real_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = real_run_project(tmp_path / "proj")
    monkeypatch.chdir(project.root)
    report = hunt(project, "misuse", runner=HuntReplay(tmp_path / "replay"))
    # Round 1 confirmed three facets in four trajectories; rounds 2 and 3 found nothing new, which is patience.
    assert (report["rounds"], report["converged"], report["quiet"], report["found"]) == (3, True, 2, 4)
    assert [(record["round"], record["reads"], record["found"], len(record["new"])) for record in
            report["per_round"]] == [(1, {"hit": 4, "coverage": 4, "random": 4}, 4, 3),
                                     (2, {"coverage": 6, "random": 6}, 0, 0), (3, {"coverage": 6, "random": 6}, 0, 0)]
    assert [(facet["name"], facet["found"]) for facet in report["facets"]] == [
        ("substitute_answered_as_result", 1), ("answer_claims_done_over_failure", 1),
        ("outline_result_answered_as_findings", 2)]
    probes = {row["name"]: row for row in report["probes"]}
    active = sorted(name for name, row in probes.items() if row["status"] == "active")
    assert active == ["answer_fills_outline_result", "closing_answer_over_failed_call", "outline_result_filled_in",
                      "retried_with_other_arguments", "substitute_call_after_failure"]
    assert all((row["hits"], row["read"], row["precision"]) == (2, 2, 1.0) for row in probes.values()
               if row["status"] == "active")
    # The probes of round 2 hit nothing, and the proposal of round 3 retired them with its reasons.
    retired = {name: row for name, row in probes.items() if row["status"] == "retired"}
    assert sorted(retired) == ["empty_looking_result_filled_in", "failure_passed_over_in_answer",
                               "retry_with_unlisted_failure"]
    assert all(row["since"] == 2 and row["reason"] for row in retired.values())
    assert (probes["substitute_call_after_failure"]["since"], probes["outline_result_filled_in"]["since"]) == (3, 1)
    found = report["coverage"]
    assert (found["read"], found["clusters_read"], found["hit"], found["random_reads_unhit"]) == (36, 6, 4, 16)
    assert (found["random_misses"], found["miss_share_upper"], found["expected_missed"]) == (0, 0.1707, 5.81)
    # Every hit was read in the rounds, so the closing check had nothing left.
    assert report["final"] == {"read": 0, "found": 0, "unanswered": []}
    assert all(not keys for keys in report["unverified"].values())

    run = project.hunting_dir / "misuse" / "1"
    assert json.loads((run / "report.json").read_text(encoding="utf-8")) == report
    assert sorted(path.name for path in (run / "state").iterdir()) == ["1.json", "2.json", "3.json"]
    workspace = run / "workspace"
    for number in (1, 2, 3):
        verdicts = json.loads((workspace / "rounds" / str(number) / "verdicts.json").read_text(encoding="utf-8"))
        assert len(verdicts) == 12 and (workspace / "rounds" / str(number) / "hits.parquet").is_file()
        assert len(list((workspace / "rounds" / str(number) / "cases").glob("*/*.md"))) == 12
    findings = pq.read_table(project.tables_dir / f"{FINDINGS_TABLE}.parquet").to_pandas()
    assert len(findings) == 4 and set(findings["hunt"]) == {"misuse"} and set(findings["source"]) == {"hit"}
    assert findings.set_index("key").loc["toucan/70d18959-d21d-5aa8-87e2-1fd120f8e31d", "facet"] == \
        "outline_result_answered_as_findings"
    kept = project.hunts_dir / "misuse"
    assert sorted(path.name for path in kept.iterdir()) == ["answer_fills_outline_result.sql",
                                                            "closing_answer_over_failed_call.sql", "facets.yaml",
                                                            "outline_result_filled_in.sql", "probes.yaml",
                                                            "retried_with_other_arguments.sql",
                                                            "substitute_call_after_failure.sql"]
    assert sorted(probe.name for probe in start(project, "misuse").probes) == active

    # A run stopped after round 2 continues from its state, and every call it makes is answered from the mailbox.
    (run / "report.json").unlink()
    (run / "state" / "3.json").unlink()
    assert hunt(Project.load(project.root), "misuse", runner=NoCalls()) == report
