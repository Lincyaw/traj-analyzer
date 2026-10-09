from __future__ import annotations

from datetime import datetime
from typing import Any

import pyarrow as pa

from traj_analyzer.harness import Harness
from traj_analyzer.ingest import turn_units
from traj_analyzer.layers.base import Layer
from traj_analyzer.layers.tables import Tables, column
from traj_analyzer.schema import INTERRUPTION, Step, Trajectory

EDITING = frozenset({"edit", "create", "notebook"})
VERIFYING = frozenset({"test", "build"})
SIDE_EFFECTS = frozenset({"destructive", "remote"})
TURNS = pa.schema([
    column("index", pa.int64(), "The turn number, from 0."),
    column("start", pa.int64(), "The step of the message the user typed that opens the turn."),
    column("end", pa.int64(), "The last step of the turn, before the next message the user typed."),
    column("prompt_chars", pa.int64(), "Characters of the message that opens the turn."),
    column("notices", pa.int64(), "Notice steps of the turn: text the harness wrote under the user's role, such "
                                  "as the notice that the user interrupted the turn."),
    column("steps", pa.int64(), "Steps of the turn, the message included."),
    column("messages", pa.int64(), "Assistant messages of the turn."),
    column("calls", pa.int64(), "Tool calls of the turn."),
    column("actions", pa.int64(), "Actions of the turn: a shell call holds one per simple command."),
    column("edits", pa.int64(), "Actions that edit, create or write a notebook."),
    column("verifications", pa.int64(), "Actions that test or build."),
    column("asks", pa.int64(), "Actions that ask the user."),
    column("plans", pa.int64(), "Actions that write or leave a plan."),
    column("side_effects", pa.int64(), "Actions whose effect is destructive or remote."),
    column("failed_calls", pa.int64(), "Tool calls whose result reads as a failure."),
    column("delegations", pa.int64(), "Actions that delegate to another agent."),
    column("duration_s", pa.float64(), "Seconds from the first to the last timestamped step of the turn; empty "
                                       "without two timestamps."),
    column("ended_by", pa.string(), "How the turn ended: interrupt, when the user interrupted it; ask, when the "
                                    "last action asked the user; response, when the assistant answered; tool, "
                                    "when it ends on a call, a result, thinking or a system step."),
    column("first_act", pa.string(), "The act of the first action of the turn."),
    column("last_act", pa.string(), "The act of the last action of the turn."),
    column("last_program", pa.string(), "The program and subcommand of the last action, or its tool name."),
])


def _program(action: dict[str, Any]) -> str:
    if action["program"]:
        return f"{action['program']} {action['subcommand']}" if action["subcommand"] else action["program"]
    return action["tool"].lower()


def _duration(steps: list[Step]) -> float | None:
    stamps = [datetime.fromisoformat(step.timestamp) for step in steps if step.timestamp]
    return (stamps[-1] - stamps[0]).total_seconds() if len(stamps) > 1 else None


def _is_interruption(step: Step) -> bool:
    return step.kind == "notice" and step.name == INTERRUPTION


def compute(trajectory: Trajectory, tables: Tables, harness: Harness) -> dict[str, list[dict[str, Any]]]:
    units = turn_units(trajectory)
    by_index = {step.index: step for step in trajectory.steps}
    actions = sorted(tables.records("actions"), key=lambda a: (a["call"], a["ordinal"]))
    rows: list[dict[str, Any]] = []
    for number, (start, end) in enumerate(units):
        steps = [by_index[index] for index in range(start, end + 1) if index in by_index]
        own = [a for a in actions if start <= a["call"] <= end]
        last_step = steps[-1]
        last = own[-1] if own else None
        after = last["call"] if last is not None else start
        interrupted = any(_is_interruption(step) for step in steps if step.index > after) or (
            last is not None and last["outcome"] == "interrupted")
        if interrupted:
            ended_by = "interrupt"
        elif last is not None and last["act"] == "ask":
            ended_by = "ask"
        elif last_step.role == "assistant" and last_step.kind == "message":
            ended_by = "response"
        else:
            ended_by = "tool"
        rows.append({
            "index": number, "start": start, "end": end, "prompt_chars": len(by_index[start].content),
            "notices": sum(step.kind == "notice" for step in steps),
            "steps": len(steps),
            "messages": sum(step.role == "assistant" and step.kind == "message" for step in steps),
            "calls": sum(step.kind == "tool_call" for step in steps),
            "actions": len(own),
            "edits": sum(a["act"] in EDITING for a in own),
            "verifications": sum(a["act"] in VERIFYING for a in own),
            "asks": sum(a["act"] == "ask" for a in own),
            "plans": sum(a["act"] == "plan" for a in own),
            "side_effects": sum(a["effect"] in SIDE_EFFECTS for a in own),
            "failed_calls": sum(a["ordinal"] == 0 and a["outcome"] == "error" for a in own),
            "delegations": sum(a["act"] == "delegate" for a in own),
            "duration_s": _duration(steps),
            "ended_by": ended_by,
            "first_act": own[0]["act"] if own else None,
            "last_act": last["act"] if last is not None else None,
            "last_program": _program(last) if last is not None else None,
        })
    return {"turns": rows}


LAYER = Layer(
    name="turns",
    description="Every message the user typed with what followed it until the next one: how much the agent did, "
                "whether it edited, verified or asked, and how the turn ended.",
    schemas={"turns": TURNS},
    compute=compute,
    requires=("actions",),
)
