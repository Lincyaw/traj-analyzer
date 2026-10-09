from __future__ import annotations

from typing import Any

import pyarrow as pa

from traj_analyzer.actions import actions, calls_of, relate
from traj_analyzer.harness import Harness
from traj_analyzer.layers.base import Layer
from traj_analyzer.layers.tables import Tables, column
from traj_analyzer.schema import Trajectory

OBJECT = pa.struct([("kind", pa.string()), ("name", pa.string()), ("role", pa.string())])
ACTIONS = pa.schema([
    column("position", pa.int64(), "The position of the action among the actions of the trajectory."),
    column("call", pa.int64(), "The step of the tool call the action belongs to."),
    column("ordinal", pa.int64(), "The position inside the call: a shell call holds one action per simple command."),
    column("turn", pa.int64(), "How many messages the user had typed before the call."),
    column("tool", pa.string(), "The tool of the call."),
    column("act", pa.string(), "What the action does: read, search, edit, create, delete, inspect, test, build, "
                               "install, run, vcs, network, process, shell, delegate, wait, plan, ask, web, skill, "
                               "notebook, mcp or other."),
    column("program", pa.string(), "The program of a shell command, such as git."),
    column("subcommand", pa.string(), "The subcommand of a shell command, such as push."),
    column("effect", pa.string(), "What it does outside the agent: none, workspace, history, destructive, remote, "
                                  "environment or unknown."),
    column("forced", pa.bool_(), "Whether it discards or overwrites without asking, as `rm -f` or `--force`."),
    column("outcome", pa.string(), "ok, empty, error, rejected, interrupted or unanswered."),
    column("error", pa.string(), "The class of a failure: not_found, mismatch, precondition, permission, timeout, "
                                 "test_failed, build_failed, exit, exception, argument, tool_missing, denied, "
                                 "sibling or unknown."),
    column("shape", pa.list_(pa.string()), "Qualifiers: range, replace_all, pipe, background, heredoc, expanded, "
                                           "cached, unparsed, and the nesting of a shell command."),
    column("lines", pa.int64(), "Lines of new text an edit or create writes."),
    column("objects", pa.list_(OBJECT), "What the action acts on: kind (file, directory, glob, pattern, url, "
                                        "agent, skill), name (the resolved path or text) and role of a file "
                                        "(source, test, config, lock, docs, script, data, asset, notebook, "
                                        "generated, harness, other)."),
    column("text", pa.string(), "The simple command as written, for a shell action."),
])
RELATIONS = pa.schema([
    column("kind", pa.string(), "repeat, retry, seen, reedit, verify, undo or narrow."),
    column("later", pa.int64(), "Position of the later action in the actions table."),
    column("earlier", pa.int64(), "Position of the earlier action."),
])


def compute(trajectory: Trajectory, tables: Tables, harness: Harness) -> dict[str, list[dict[str, Any]]]:
    found = actions(calls_of(trajectory), harness)
    rows = []
    for position, action in enumerate(found):
        rows.append({
            "position": position, "call": action.call, "ordinal": action.ordinal, "turn": action.turn,
            "tool": action.tool, "act": action.act, "program": action.program, "subcommand": action.subcommand,
            "effect": action.effect, "forced": action.forced, "outcome": action.outcome, "error": action.error,
            "shape": list(action.shape), "lines": action.lines,
            "objects": [{"kind": o.kind, "name": o.name, "role": o.role} for o in action.objects],
            "text": " ".join(action.command.argv) if action.command is not None else None,
        })
    relations = [{"kind": r.kind, "later": r.later, "earlier": r.earlier} for r in relate(found)]
    return {"actions": rows, "relations": relations}


LAYER = Layer(
    name="actions",
    description="Every tool call parsed into actions, shell calls into their simple commands: what each does, to "
                "what, with what effect and outcome, and how the actions relate.",
    schemas={"actions": ACTIONS, "relations": RELATIONS},
    compute=compute,
)
