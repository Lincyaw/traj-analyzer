from __future__ import annotations

from collections import Counter
from typing import Any

from traj_analyzer.layers.tables import Tables
from traj_analyzer.operators.base import Evidenced, FeatureSpec, NoParams, Operator, has_tool_calls
from traj_analyzer.schema import Trajectory

RATE = (0.0, 1.0)
PATH_LEVELS = ["act", "program", "effect", "outcome"]
MOVE_LEVELS = ["after", "act", "program"]
EDITING = {"edit", "create", "notebook"}
OBSERVING = {"read", "search", "inspect"}
VERIFYING = {"test", "build"}
SIDE_EFFECTS = {"destructive", "remote"}


def outputs(params: NoParams) -> list[FeatureSpec]:
    def rate(name: str, description: str) -> FeatureSpec:
        return FeatureSpec(name=name, type="scalar", range=RATE, description=description)

    def count(name: str, description: str) -> FeatureSpec:
        return FeatureSpec(name=name, type="scalar", description=description)

    return [
        FeatureSpec(
            name="action_paths", type="paths", levels=PATH_LEVELS,
            description="One path per action in order, where a shell call holds one action per simple command of "
                        "its script. act: what the action does, as read, search, edit, create, delete, inspect, "
                        "test, build, install, run, vcs, network, process, shell, delegate, wait, plan, ask, web, "
                        "skill, notebook, mcp or other. program: the program and subcommand of a shell command, "
                        "such as `git push`, or the tool name. effect: what it does outside the agent, as none, "
                        "workspace, history, destructive, remote, environment or unknown. outcome: ok, empty, "
                        "error, rejected, interrupted or unanswered, with the error class after a colon."),
        FeatureSpec(
            name="action_moves", type="paths", levels=MOVE_LEVELS,
            description="One path per action in order, relating it to the action before it. after: the act and "
                        "outcome of the action before as <act>:<outcome>, `prompt` for the first action after a "
                        "message the user typed, and `start` for the first action. act and program: as in "
                        "action_paths."),
        count("edited_files", "Distinct files edited or created."),
        count("observed_files", "Distinct files read, searched or inspected."),
        rate("edit_seen_share", "Share of edited files that an earlier action read, searched or inspected before "
                                "their first edit. Empty without an edit."),
        rate("reedit_share", "Share of edits that change a file edited before. Empty without an edit."),
        rate("edit_verified_share", "Share of edits followed by a test or build in the same turn, before the next "
                                    "message the user typed. Empty without an edit."),
        FeatureSpec(name="verified_at_end", type="boolean",
                    description="A test or build follows the last edit. Empty without an edit."),
        FeatureSpec(name="last_verification_failed", type="boolean",
                    description="The last test or build of the trajectory failed. Empty without a test or build."),
        rate("verification_failure_share", "Share of tests and builds that failed. Empty without a test or build."),
        rate("call_failure_share", "Share of tool calls whose result reads as a failure. Empty without a call."),
        rate("retry_share", "Among the failed actions, the share repeated later with the same arguments. Empty "
                            "without a failed action."),
        rate("recovery_share", "Among the objects an action failed on, the share on which a later action of the "
                               "same act succeeded. Empty without a failure on an object."),
        count("longest_failure_run", "Most consecutive failed tool calls."),
        count("side_effects", "Actions whose effect is destructive or remote."),
        count("forced_actions", "Actions that discard or overwrite without asking, such as `rm -f`, "
                                "`git push --force` or `git commit --no-verify`."),
        count("undo_actions", "Version-control actions that discard changes to a file edited earlier."),
        rate("explore_share", "Share of actions before the first edit that read, search or inspect. Empty without "
                              "an edit."),
        rate("test_edit_share", "Share of edited files that are tests. Empty without an edit."),
        rate("config_edit_share", "Share of edited files that are configuration or lock files. Empty without an "
                                  "edit."),
        rate("unknown_program_share", "Share of shell commands running a program the shell component does not "
                                      "know. Empty without a shell command."),
        count("shell_commands_per_call", "Simple shell commands divided by shell calls. Empty without a shell "
                                         "call."),
    ]


def _share(hits: int, total: int) -> float | None:
    return hits / total if total else None


def _program(action: dict[str, Any]) -> str:
    if action["program"]:
        return f"{action['program']} {action['subcommand']}" if action["subcommand"] else action["program"]
    return action["tool"].lower()


def _paths(action: dict[str, Any]) -> list[str]:
    return [o["name"] for o in action["objects"] if o["kind"] in ("file", "directory")]


def compute(trajectory: Trajectory, tables: Tables, params: NoParams) -> dict[str, Any]:
    found = tables.records("actions")
    by_kind: dict[str, list[tuple[int, int]]] = {}
    for relation in tables.records("relations"):
        by_kind.setdefault(relation["kind"], []).append((relation["later"], relation["earlier"]))

    paths = [[a["act"], _program(a), a["effect"], f"{a['outcome']}:{a['error']}" if a["error"] else a["outcome"]]
             for a in found]
    moves = []
    before = "start"
    turn = 0
    for action in found:
        if action["turn"] != turn:
            before, turn = "prompt", action["turn"]
        moves.append([before, action["act"], _program(action)])
        before = f"{action['act']}:{action['outcome']}"

    edits = [i for i, a in enumerate(found) if a["act"] in EDITING]
    edited: dict[str, int] = {}
    for i in edits:
        for path in _paths(found[i]):
            edited.setdefault(path, i)
    seen_edits = {later for later, _ in by_kind.get("seen", [])}
    first_edits = set(edited.values())
    verified = {earlier for _, earlier in by_kind.get("verify", [])}
    verifications = [i for i, a in enumerate(found) if a["act"] in VERIFYING]
    observed = {p for a in found if a["act"] in OBSERVING for p in _paths(a)}
    first_edit = edits[0] if edits else None
    calls_failed = [(a["outcome"] == "error", a["call"]) for a in found if a["ordinal"] == 0]
    longest = run = 0
    run_end: list[int] = []
    for failed, call in calls_failed:
        run = run + 1 if failed else 0
        if run > longest:
            longest, run_end = run, [call]

    retries = {earlier for _, earlier in by_kind.get("retry", [])}
    failed_objects: dict[str, str] = {}
    recovered: set[str] = set()
    for action in found:
        for path in _paths(action):
            if action["outcome"] == "error":
                failed_objects.setdefault(path, action["act"])
            elif path in failed_objects and failed_objects[path] == action["act"] and action["outcome"] == "ok":
                recovered.add(path)
    failed_actions = sum(a["outcome"] == "error" for a in found)

    side_effects = [a for a in found if a["effect"] in SIDE_EFFECTS]
    forced = [a for a in found if a["forced"]]
    undone = {later for later, _ in by_kind.get("undo", [])}
    shell_actions = [a for a in found if a["text"] is not None]
    shell_calls = {a["call"] for a in shell_actions}
    roles = Counter(o["role"] for i in edits for o in found[i]["objects"] if o["kind"] == "file")
    return {
        "action_paths": paths,
        "action_moves": moves,
        "edited_files": float(len(edited)),
        "observed_files": float(len(observed)),
        "edit_seen_share": _share(len(first_edits & seen_edits), len(first_edits)),
        "reedit_share": _share(len({later for later, _ in by_kind.get("reedit", [])}), len(edits)),
        "edit_verified_share": Evidenced(
            _share(len(verified), len(edits)),
            tuple(sorted({found[i]["call"] for i in edits if i not in verified})), "edits without a verification"),
        "verified_at_end": None if not edits else any(v > edits[-1] for v in verifications),
        "last_verification_failed": found[verifications[-1]]["outcome"] == "error" if verifications else None,
        "verification_failure_share": _share(sum(found[v]["outcome"] == "error" for v in verifications),
                                             len(verifications)),
        "call_failure_share": _share(sum(failed for failed, _ in calls_failed), len(calls_failed)),
        "retry_share": _share(len(retries), failed_actions),
        "recovery_share": _share(len(recovered), len(failed_objects)),
        "longest_failure_run": Evidenced(float(longest), tuple(run_end), "the last call of the longest run"),
        "side_effects": Evidenced(float(len(side_effects)), tuple(sorted({a["call"] for a in side_effects}))),
        "forced_actions": Evidenced(float(len(forced)), tuple(sorted({a["call"] for a in forced}))),
        "undo_actions": Evidenced(float(len(undone)), tuple(sorted({found[i]["call"] for i in undone}))),
        "explore_share": None if first_edit is None
        else _share(sum(a["act"] in OBSERVING for a in found[:first_edit]), first_edit),
        "test_edit_share": _share(roles["test"], sum(roles.values())),
        "config_edit_share": _share(roles["config"] + roles["lock"], sum(roles.values())),
        "unknown_program_share": _share(sum(a["act"] == "other" for a in shell_actions), len(shell_actions)),
        "shell_commands_per_call": len(shell_actions) / len(shell_calls) if shell_calls else None,
    }


OPERATOR = Operator(
    kind="code",
    description="Every tool call parsed into actions, shell calls into their commands: what each does, to what, "
                "with what effect and outcome, and how the actions relate: reading before editing, verifying after, "
                "retrying, undoing.",
    tags=("agent", "coding"),
    outputs=outputs,
    compute=compute,
    requires=has_tool_calls,
    tables=("actions", "relations"),
)
