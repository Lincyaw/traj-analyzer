from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from traj_analyzer.schema import Trajectory

BuiltinFn = Callable[[Trajectory], Any]
REGISTRY: dict[str, BuiltinFn] = {}


def builtin(fn: BuiltinFn) -> BuiltinFn:
    REGISTRY[fn.__name__] = fn
    return fn


@builtin
def n_steps(t: Trajectory) -> float:
    return float(len(t.steps))


@builtin
def n_user_turns(t: Trajectory) -> float:
    return float(sum(1 for s in t.steps if s.role == "user" and s.kind == "message"))


@builtin
def n_tool_calls(t: Trajectory) -> float:
    return float(sum(1 for s in t.steps if s.kind == "tool_call"))


@builtin
def tool_error_rate(t: Trajectory) -> float | None:
    results = [s for s in t.steps if s.kind == "tool_result"]
    if not results:
        return None
    return sum(s.is_error for s in results) / len(results)


@builtin
def total_chars(t: Trajectory) -> float:
    return float(sum(len(s.content) for s in t.steps))


@builtin
def duration_minutes(t: Trajectory) -> float | None:
    stamps = [datetime.fromisoformat(s.timestamp) for s in t.steps if s.timestamp]
    if len(stamps) < 2:
        return None
    return (stamps[-1] - stamps[0]).total_seconds() / 60


@builtin
def tools_used(t: Trajectory) -> list[str]:
    return sorted({s.name for s in t.steps if s.kind == "tool_call" and s.name})
