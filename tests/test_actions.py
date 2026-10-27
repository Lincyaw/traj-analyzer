from __future__ import annotations

import json
import re
from collections import Counter

from test_pipeline import DATA, TOUCAN

from traj_analyzer.actions import Call, actions, calls_of, classify, project_root, relate, role_of
from traj_analyzer.adapters.messages import MessagesAdapter
from traj_analyzer.harness import EMPTY, Harness, discover_harnesses
from traj_analyzer.layers import discover_layers
from traj_analyzer.layers.tables import STEPS, Tables, steps_rows
from traj_analyzer.operators.library.stats import actions as operator
from traj_analyzer.project import RenderConfig
from traj_analyzer.schema import Step, Trajectory

ROOT = "/home/u/proj"
CLAUDE = discover_harnesses(None)["claude_code"]


def call(index: int, tool: str, arguments: dict, result: str | None, turn: int = 0) -> Call:
    return Call(index=index, tool=tool, arguments=json.dumps(arguments), result=result, turn=turn)


CALLS = [
    call(1, "Read", {"file_path": f"{ROOT}/src/a.py"}, "1\tx"),
    call(3, "Bash", {"command": f"cd {ROOT} && pytest -q tests/ && git status"},
         "Exit code 1\nFAILED tests/test_a.py"),
    call(5, "Edit", {"file_path": f"{ROOT}/src/a.py", "old_string": "x", "new_string": "y\nz"},
         "The file has been updated", 1),
    call(7, "Bash", {"command": "uv run pytest -q tests/"}, "3 passed", 1),
    call(9, "Bash", {"command": "git checkout -- src/a.py"}, "", 1),
    call(11, "Grep", {"pattern": "foo", "path": f"{ROOT}/src"}, "No matches found", 1),
    call(13, "Write", {"file_path": f"{ROOT}/tests/test_b.py", "content": "assert True\n"},
         "File created successfully at: x", 1),
    call(15, "Bash", {"command": "git push --force origin main"}, "Exit code: 0\nWall time: 1", 2),
]


def test_actions_expand_shell_calls_and_resolve_paths() -> None:
    found = actions(CALLS, CLAUDE)
    assert [(a.call, a.ordinal, a.act) for a in found[:5]] == [(1, 0, "read"), (3, 0, "shell"), (3, 1, "test"),
                                                               (3, 2, "vcs"), (5, 0, "edit")]
    read, cd, pytest_, status, edit = found[:5]
    assert read.objects[0].relative == "src/a.py" and read.objects[0].role == "source"
    assert (cd.program, pytest_.program, status.subcommand) == ("cd", "pytest", "status")
    assert (pytest_.outcome, pytest_.error, status.error) == ("error", "test_failed", "exit")
    assert (edit.effect, edit.lines, edit.turn) == ("workspace", 2, 1)
    push = found[-1]
    assert (push.program, push.subcommand, push.effect, push.forced) == ("git", "push", "destructive", True)
    assert push.outcome == "ok"
    created = found[-2]
    assert (created.act, created.objects[0].role) == ("create", "test")


def test_profiles_decide_acts_argument_keys_and_outcomes() -> None:
    plain = actions(CALLS, EMPTY)
    assert {a.act for a in plain} == {"other"} and all(a.objects == () for a in plain)
    assert [a.outcome for a in plain] == ["ok"] * 4 + ["empty"] + ["ok"] * 3
    codex = Harness(name="codex", tool_acts={"exec_command": "shell"}, command_keys=("cmd",), cwd_keys=("workdir",))
    recorded = call(15, "exec_command", {"cmd": "git push --force origin main", "workdir": ROOT},
                    "Exit code: 0\nWall time: 1", 2)
    (push,) = actions([recorded], codex)
    assert (push.program, push.subcommand, push.effect, push.forced) == ("git", "push", "destructive", True)
    assert actions([recorded], CLAUDE)[0].act == "other"
    assert classify("boom", "read", True, EMPTY) == ("error", "unknown")
    assert classify("Exit code 1\nFAILED tests/test_a.py", "test", None, EMPTY) == ("ok", None)


def test_relations_link_actions_by_their_objects() -> None:
    found = actions(CALLS, CLAUDE)
    kinds = {(r.kind, found[r.later].call, found[r.earlier].call) for r in relate(found)}
    assert ("seen", 5, 1) in kinds and ("verify", 7, 5) in kinds and ("undo", 9, 5) in kinds


def test_actions_of_toucan_calls_read_the_arguments_the_adapter_parsed() -> None:
    # The tools of the Word document server in the Toucan data name the document in `filename`.
    prefix = "office-word-document-server-"
    word = Harness(name="word", mcp_prefixes=(prefix,), path_keys=("filename",))
    found = [(action.act, action.paths)
             for trajectory in MessagesAdapter(**TOUCAN["options"]).read(DATA / "toucan.jsonl", "t")
             for action in actions(calls_of(trajectory), word) if action.tool.startswith(prefix)]
    assert Counter(found) == {("mcp", ("Q1_2024_Sales_Report.docx",)): 8, ("mcp", ("Q3_Sales_Report.docx",)): 5}


def test_classify_reads_results() -> None:
    assert classify("The user doesn't want to proceed with this tool use.", "edit", None, CLAUDE) == ("rejected", None)
    mismatch = "<tool_use_error>String to replace not found in file.</tool_use_error>"
    assert classify(mismatch, "edit", None, CLAUDE) == ("error", "mismatch")
    unread = "<tool_use_error>File has not been read yet.</tool_use_error>"
    assert classify(unread, "edit", None, CLAUDE) == ("error", "precondition")
    assert classify("Exit code 143\nCommand timed out after 10m 0s", "run", None, CLAUDE) == ("error", "timeout")
    assert classify("Exit code 127\nzsh: command not found: foo", "run", None, CLAUDE) == ("error", "tool_missing")
    assert classify("(Bash completed with no output)", "shell", None, CLAUDE) == ("empty", None)
    assert classify("No matches found", "search", None, CLAUDE) == ("empty", None)
    assert classify(None, "read", None, CLAUDE) == ("unanswered", None)
    assert classify("     1\timport os\n", "read", None, CLAUDE) == ("ok", None)
    assert classify("[Request interrupted by user for tool use]", "shell", None, CLAUDE) == ("interrupted", None)
    sibling = "<tool_use_error>Sibling tool call errored</tool_use_error>"
    assert classify(sibling, "read", True, CLAUDE) == ("error", "sibling")


def test_classify_reads_an_exit_code_the_profile_places_at_the_end() -> None:
    closing = Harness(name="closing", closing_exit_code=re.compile(r"\[Command finished with exit code (-?\d+)\]$"))
    output = "row\n" * 300
    assert classify(f"{output}[Command finished with exit code 2]", "run", None, closing) == ("error", "exit")
    assert classify(f"{output}[Command finished with exit code 124]", "run", None, closing) == ("error", "timeout")
    assert classify(f"{output}[Command finished with exit code 0]", "run", None, closing) == ("ok", None)
    # A mark inside the output is no exit code of the command.
    quoted = f"[Command finished with exit code 1]\n{output}[Command finished with exit code 0]"
    assert classify(quoted, "run", None, closing) == ("ok", None)
    assert classify(f"{output}[Command finished with exit code 2]", "run", None, CLAUDE) == ("ok", None)


def test_roles_and_root() -> None:
    assert role_of("src/app/main.py") == "source" and role_of("tests/test_main.py") == "test"
    assert role_of("pyproject.toml") == "config" and role_of("uv.lock") == "lock" and role_of("docs/guide.md") == "docs"
    assert role_of(".claude/skills/x/SKILL.md") == "harness" and role_of("node_modules/a/index.js") == "generated"
    assert project_root([f"{ROOT}/src/a.py", f"{ROOT}/tests/b.py", "/tmp/x.log"]) == ROOT
    assert project_root(["/home/u/a.py", "/home/u/b.py"]) is None


def trajectory_of(calls: list[Call]) -> Trajectory:
    steps = [Step(index=0, role="user", content="fix it")]
    for c in calls:
        steps.append(Step(index=len(steps), role="assistant", kind="tool_call", name=c.tool, content=c.arguments))
        steps.append(Step(index=len(steps), role="tool", kind="tool_result", name=c.tool, content=c.result or ""))
    return Trajectory(id="t", dataset="d", steps=steps, harness="claude_code")


def tables_of(trajectory: Trajectory) -> Tables:
    layer = discover_layers(None)["actions"].layer
    rows = {"steps": steps_rows(trajectory, RenderConfig())}
    tables = Tables.of(rows, {"steps": STEPS})
    rows.update(layer.compute(trajectory, tables, CLAUDE))
    return Tables.of(rows, {"steps": STEPS, **layer.schemas})


def test_actions_layer_writes_one_row_per_action_with_its_objects() -> None:
    tables = tables_of(trajectory_of(CALLS))
    found = tables["actions"]
    assert list(found["position"]) == list(range(len(found)))
    assert found.loc[0, "objects"][0]["role"] == "source" and found.loc[2, "act"] == "test"
    assert set(tables["relations"]["kind"]) >= {"seen", "verify", "undo"}


def test_operator_computes_factors_from_a_trajectory() -> None:
    trajectory = trajectory_of(CALLS)
    values = operator.compute(trajectory, tables_of(trajectory), operator.OPERATOR.params())
    assert values["edited_files"] == 2 and values["edit_seen_share"] == 0.5 and values["verified_at_end"] is False
    assert (values["side_effects"].value, values["forced_actions"].value, values["undo_actions"].value) == (2, 1, 1)
    # The evidence of a count names the calls it counts.
    assert values["side_effects"].steps == (9, 15)
    assert values["action_paths"][2] == ["test", "pytest", "none", "error:test_failed"]
    assert values["action_moves"][0] == ["prompt", "read", "read"]
    assert values["shell_commands_per_call"] == 6 / 4
