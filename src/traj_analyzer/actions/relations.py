from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from traj_analyzer.actions.calls import Action

Kind = Literal["repeat", "retry", "seen", "reedit", "verify", "undo", "narrow"]
"""How a later action relates to an earlier one on the same object:
repeat, the same action with the same arguments; retry, the same act on the same object after the earlier one
failed; seen, an edit of a file an earlier action read, searched or inspected; reedit, an edit of a file edited
before; verify, a test or build after an edit, in the same turn; undo, a destructive version-control action on a file
edited earlier; narrow, a read of a part of a file read whole before."""

OBSERVING = frozenset({"read", "search", "inspect"})
EDITING = frozenset({"edit", "create", "notebook"})
VERIFYING = frozenset({"test", "build"})
UNDOING_SUBCOMMANDS = frozenset({"checkout", "restore", "stash", "reset", "clean"})


@dataclass(frozen=True)
class Relation:
    kind: Kind
    later: int
    """Position of the later action in the list."""
    earlier: int
    """Position of the earlier action."""


def _key(action: Action) -> tuple:
    argv = action.command.argv if action.command is not None else ()
    return (action.act, action.program, action.subcommand, action.paths, argv, action.tool)


def relate(actions: list[Action]) -> list[Relation]:
    """Every relation between the actions, found by the objects they share."""
    out: list[Relation] = []
    last_by_key: dict[tuple, int] = {}
    observed: dict[str, int] = {}
    edited: dict[str, int] = {}
    whole_reads: dict[str, int] = {}
    last_edit_turn: dict[str, tuple[int, int]] = {}
    for position, action in enumerate(actions):
        key = _key(action)
        if key in last_by_key and action.act not in ("plan", "wait", "ask"):
            earlier = last_by_key[key]
            out.append(Relation("retry" if actions[earlier].failed else "repeat", position, earlier))
        elif action.act not in ("plan", "wait", "ask"):
            for path in action.paths:
                for earlier in range(position - 1, -1, -1):
                    before = actions[earlier]
                    if before.act == action.act and path in before.paths:
                        if before.failed:
                            out.append(Relation("retry", position, earlier))
                        break
        last_by_key[key] = position
        if action.act in EDITING:
            for path in action.paths:
                if path in observed:
                    out.append(Relation("seen", position, observed[path]))
                if path in edited:
                    out.append(Relation("reedit", position, edited[path]))
                edited[path] = position
                last_edit_turn[path] = (position, action.turn)
        if action.act in OBSERVING:
            for path in action.paths:
                observed[path] = position
                if action.act == "read" and "range" in action.shape and path in whole_reads:
                    out.append(Relation("narrow", position, whole_reads[path]))
                if action.act == "read" and "range" not in action.shape:
                    whole_reads[path] = position
        if action.act in VERIFYING:
            for path, (edit_position, turn) in list(last_edit_turn.items()):
                if turn == action.turn:
                    out.append(Relation("verify", position, edit_position))
                    del last_edit_turn[path]
        if action.act == "vcs" and action.effect == "destructive" and action.subcommand in UNDOING_SUBCOMMANDS:
            targets = set(action.paths)
            for path, edit_position in list(edited.items()):
                if not targets or path in targets or any(path.startswith(t.rstrip("/") + "/") for t in targets):
                    out.append(Relation("undo", position, edit_position))
    return out
