from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from traj_analyzer.actions.kinds import Act, ErrorClass, Outcome
from traj_analyzer.actions.objects import Object, object_of, project_root
from traj_analyzer.actions.outcomes import classify
from traj_analyzer.harness.base import Harness
from traj_analyzer.schema import Trajectory
from traj_analyzer.shell import Command, Description, Effect, ParseError, describe, parse

EFFECTS: dict[str, Effect] = {"edit": "workspace", "create": "workspace", "delete": "destructive",
                              "notebook": "workspace", "delegate": "unknown", "mcp": "unknown", "other": "unknown"}
"""The effect of a file or harness tool by its act; every other act has none, and the shell component gives the
effect of a command."""


@dataclass(frozen=True)
class Call:
    """One tool call of a trajectory with its result, as the actions component takes it."""

    index: int
    tool: str
    arguments: str
    result: str | None = None
    turn: int = 0
    """How many messages the user had typed before the call."""
    is_error: bool | None = None
    """The adapter's error flag for the result, when it has one."""


@dataclass(frozen=True)
class Action:
    """One thing the agent did: a file tool call, or one simple command of a shell call."""

    call: int
    """The step index of the call."""
    ordinal: int
    """The action's position inside its call: a shell call holds one action per simple command."""
    tool: str
    act: Act
    effect: Effect
    objects: tuple[Object, ...]
    outcome: Outcome
    error: ErrorClass | None
    turn: int
    program: str | None = None
    subcommand: str | None = None
    forced: bool = False
    shape: tuple[str, ...] = ()
    """Qualifiers of how it was done: `range` for a partial read, `replace_all`, `pipe`, `background`, the
    nesting of a shell command, `cached` for a read the harness answered from memory."""
    lines: int | None = None
    """Lines of new text an edit or create writes, when the arguments say."""
    command: Command | None = field(default=None, compare=False)
    description: Description | None = field(default=None, compare=False)

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(o.name for o in self.objects if o.kind in ("file", "directory"))

    @property
    def failed(self) -> bool:
        return self.outcome == "error"


def calls_of(trajectory: Trajectory) -> list[Call]:
    """The tool calls of a trajectory paired with their results.

    A result answers the earliest call without one, of the same tool when the result names its tool.
    """
    pending: list[tuple[Call, int]] = []
    found: dict[int, Call] = {}
    turn = 0
    for step in trajectory.steps:
        if step.is_user_message:
            turn += 1
        elif step.kind == "tool_call":
            arguments = step.arguments if step.arguments is not None else step.content
            call = Call(index=step.index, tool=step.name or "unnamed", arguments=arguments, turn=turn)
            found[step.index] = call
            pending.append((call, step.index))
        elif step.kind == "tool_result":
            for position, (call, index) in enumerate(pending):
                if step.name in (None, call.tool):
                    found[index] = Call(index=call.index, tool=call.tool, arguments=call.arguments,
                                        result=step.content, turn=call.turn, is_error=step.is_error or None)
                    del pending[position]
                    break
    return [found[index] for index in sorted(found)]


def actions(calls: list[Call], harness: Harness) -> list[Action]:
    """The actions of these calls in order, read with the harness's profile; a shell call that does not parse
    yields one `shell` action."""
    root = project_root([path for call in calls for path in _argument_paths(call, harness)])
    out: list[Action] = []
    for call in calls:
        out.extend(_actions_of(call, root, harness))
    return out


def _arguments(call: Call) -> dict[str, Any]:
    text = call.arguments.lstrip()
    if not text.startswith("{"):
        return {}
    return json.loads(call.arguments)


def _string(arguments: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    """The value of the first of these argument keys that holds a string."""
    for key in keys:
        if isinstance(arguments.get(key), str):
            return arguments[key]
    return None


def _argument_paths(call: Call, harness: Harness) -> list[str]:
    arguments = _arguments(call)
    found = [arguments[key] for key in harness.path_keys if isinstance(arguments.get(key), str)]
    found += [arguments[key] + "/x" for key in harness.cwd_keys if isinstance(arguments.get(key), str)]
    return found


def _actions_of(call: Call, root: str | None, harness: Harness) -> list[Action]:
    arguments = _arguments(call)
    name = call.tool.lower()
    act: Act = harness.tool_acts.get(name, "mcp" if name.startswith(harness.mcp_prefixes) else "other")
    cwd = _string(arguments, harness.cwd_keys)
    if act == "shell":
        command_text = _command_text(arguments, harness)
        if command_text is not None:
            return _shell_actions(call, command_text, cwd, root, harness)
    outcome, error = classify(call.result, act, call.is_error, harness)
    objects: list[Object] = []
    shape: list[str] = []
    lines: int | None = None
    patch = _patch_text(call, arguments, harness) if act == "edit" else None
    path = _string(arguments, harness.path_keys)
    if patch is not None:
        assert harness.patch_file is not None
        changes = harness.patch_file.findall(patch)
        objects.extend(object_of(changed, cwd, root) for _, changed in changes)
        if any(verb == "Delete" for verb, _ in changes):
            act = "delete"
        elif changes and all(verb == "Add" for verb, _ in changes):
            act = "create"
    elif path:
        objects.append(object_of(path, cwd, root))
    if act in ("read", "search") and any(key in arguments for key in harness.range_keys):
        shape.append("range")
    pattern = _string(arguments, harness.pattern_keys)
    if act == "search" and pattern is not None:
        objects.append(Object(kind="pattern", name=pattern))
    if act == "web":
        url = _string(arguments, harness.url_keys)
        if url is not None:
            objects.append(Object(kind="url", name=url))
        if pattern is not None:
            objects.append(Object(kind="pattern", name=pattern))
    agent = _string(arguments, harness.agent_keys)
    if act == "delegate" and agent is not None:
        objects.append(Object(kind="agent", name=agent))
    skill = _string(arguments, harness.skill_keys)
    if act == "skill" and skill is not None:
        objects.append(Object(kind="skill", name=skill))
    if act in ("edit", "create"):
        lines = _new_lines(arguments, harness)
        if any(arguments.get(key) in (True, "true") for key in harness.replace_all_keys):
            shape.append("replace_all")
    if call.result is not None and call.result.startswith(harness.cached):
        shape.append("cached")
    return [Action(call=call.index, ordinal=0, tool=call.tool, act=act, effect=EFFECTS.get(act, "none"),
                   objects=tuple(objects), outcome=outcome, error=error, turn=call.turn, shape=tuple(shape),
                   lines=lines)]


def _patch_text(call: Call, arguments: dict[str, Any], harness: Harness) -> str | None:
    """The patch an edit tool of the harness takes: under one of `patch_keys`, or the whole arguments text of a
    call that records no JSON."""
    if harness.patch_file is None:
        return None
    if not arguments:
        return call.arguments
    return _string(arguments, harness.patch_keys)


def _new_lines(arguments: dict[str, Any], harness: Harness) -> int | None:
    """Lines of the new text of an edit or create: the text under `new_text_keys`, or the sum over the objects of
    a list argument, as a tool that takes several edits records them."""
    text = _string(arguments, harness.new_text_keys)
    if text is not None:
        return text.count("\n") + 1
    batches = [value for value in arguments.values()
               if isinstance(value, list) and value and all(isinstance(item, dict) for item in value)]
    texts = [_string(item, harness.new_text_keys) for batch in batches for item in batch]
    found = [text for text in texts if text is not None]
    return sum(text.count("\n") + 1 for text in found) if found else None


def _command_text(arguments: dict[str, Any], harness: Harness) -> str | None:
    for key in harness.command_keys:
        value = arguments.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, list) and all(isinstance(part, str) for part in value):
            return " ".join(value)
    return None


def _shell_actions(call: Call, text: str, cwd: str | None, root: str | None, harness: Harness) -> list[Action]:
    if harness.shell_wrapper is not None:
        wrapped = harness.shell_wrapper.match(text.strip())
        if wrapped is not None:
            inner = wrapped.group(1).strip()
            if len(inner) >= 2 and inner[0] == inner[-1] and inner[0] in "'\"":
                text = inner[1:-1]
    try:
        script = parse(text)
    except ParseError:
        outcome, error = classify(call.result, "shell", call.is_error, harness)
        return [Action(call=call.index, ordinal=0, tool=call.tool, act="shell", effect="unknown", objects=(),
                       outcome=outcome, error=error, turn=call.turn, shape=("unparsed",))]
    out: list[Action] = []
    for ordinal, command in enumerate(script.commands):
        described = describe(command)
        act: Act = described.act  # type: ignore[assignment]
        outcome, error = classify(call.result, act, call.is_error, harness)
        base = command.cwd if command.cwd and command.cwd.startswith("/") else cwd
        if command.cwd and not command.cwd.startswith("/") and base:
            base = f"{base}/{command.cwd}"
        elif command.cwd and not command.cwd.startswith("/") and root:
            base = f"{root}/{command.cwd}"
        objects = [object_of(path, base, root) for path in described.paths]
        shape = [*command.nesting]
        if command.join == "pipe":
            shape.append("pipe")
        if command.background:
            shape.append("background")
        if command.expanded:
            shape.append("expanded")
        if any(r.kind == "heredoc" for r in command.redirects):
            shape.append("heredoc")
        out.append(Action(call=call.index, ordinal=ordinal, tool=call.tool, act=act, effect=described.effect,
                          objects=tuple(objects), outcome=outcome, error=error, turn=call.turn,
                          program=described.program or None, subcommand=described.subcommand, forced=described.forced,
                          shape=tuple(shape), command=command, description=described))
    if not out:
        outcome, error = classify(call.result, "shell", call.is_error, harness)
        out.append(Action(call=call.index, ordinal=0, tool=call.tool, act="shell", effect="none", objects=(),
                          outcome=outcome, error=error, turn=call.turn))
    return out
