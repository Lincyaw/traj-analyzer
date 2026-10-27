from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from traj_analyzer.actions.kinds import Act, ErrorClass


@dataclass(frozen=True)
class Harness:
    """What one harness writes: the names of its tools, the keys of their arguments, and the fixed texts its tool
    results open with. The actions component reads every table from here and knows no harness itself.

    Every field is empty by default; under an empty profile every tool is `other`, and a result is a failure only
    when the adapter flagged it.
    """

    name: str
    tool_acts: Mapping[str, Act] = field(default_factory=dict)
    """Lower-cased tool name to what a call of it does; a tool outside the map is `mcp` when its name opens with one
    of `mcp_prefixes` and `other` otherwise."""
    mcp_prefixes: tuple[str, ...] = ()
    """Openings of the lower-cased names of the tools the harness mounts from MCP servers."""
    command_keys: tuple[str, ...] = ()
    """Argument keys of a shell tool that hold the command text."""
    path_keys: tuple[str, ...] = ()
    """Argument keys that hold the path a file tool acts on."""
    pattern_keys: tuple[str, ...] = ()
    """Argument keys that hold the pattern of a search, or the query of a web search."""
    cwd_keys: tuple[str, ...] = ()
    """Argument keys that hold the directory a shell command runs in."""
    new_text_keys: tuple[str, ...] = ()
    """Argument keys that hold the new text of an edit or the content of a write, at the top level or inside each
    object of a list of edits."""
    replace_all_keys: tuple[str, ...] = ()
    """Argument keys that make an edit replace every occurrence."""
    range_keys: tuple[str, ...] = ()
    """Argument keys that make a read or search partial."""
    agent_keys: tuple[str, ...] = ()
    """Argument keys that name the agent a delegation goes to."""
    skill_keys: tuple[str, ...] = ()
    """Argument keys that name the skill a call invokes."""
    url_keys: tuple[str, ...] = ()
    """Argument keys that hold the URL a web call fetches."""
    patch_keys: tuple[str, ...] = ()
    """Argument keys that hold the patch text of an edit tool that takes a patch; a call of such a tool that
    records no JSON holds the patch as its whole arguments text."""
    shell_wrapper: re.Pattern[str] | None = None
    """Matches a command recorded wrapped in the shell it ran with, with the inner command as group 1."""
    patch_file: re.Pattern[str] | None = None
    """Matches the file headers of a patch text, with the verb (Update, Add, Delete) as group 1 and the path as
    group 2; the files of an edit tool that takes a patch come from it."""
    exit_code: re.Pattern[str] | None = None
    """Matches the exit code a shell result reports, with the number as group 1."""
    closing_exit_code: re.Pattern[str] | None = None
    """Matches the exit code a shell result reports at its end, with the number as group 1; it is searched in the
    closing characters of the result, as `exit_code` is in the opening."""
    rejected: tuple[str, ...] = ()
    """Marks in the opening of a result that says the user or a policy refused the call."""
    interrupted: tuple[str, ...] = ()
    """Marks in the opening of a result that says the user interrupted the call."""
    cached: tuple[str, ...] = ()
    """Openings of a result the harness answered from memory, without running the tool."""
    empty: tuple[str, ...] = ()
    """Openings of a result that holds nothing; a result without text is empty under every profile."""
    error_marks: Mapping[ErrorClass, tuple[str, ...]] = field(default_factory=dict)
    """Per error class, the marks in the opening of a failed result that give it that class; `denied` and
    `sibling` make a result a failure by themselves, `mismatch` does so for an edit or create, `exception` for a
    run, test or build, and `not_found` for a short read or inspect, while the other classes are read off a result
    the exit code, `error_openings` or the adapter's flag already made a failure."""
    error_openings: tuple[str, ...] = ()
    """Marks in the opening that make a result a failure whatever the adapter's flag says."""

    def merged(self, other: Harness, name: str) -> Harness:
        """This profile with `other` added under `name`: every table is the union of both, and a pattern is
        `other`'s where it has one."""
        values: dict[str, Any] = {}
        for own in fields(self):
            if own.name == "name":
                continue
            mine, theirs = getattr(self, own.name), getattr(other, own.name)
            if isinstance(mine, tuple):
                values[own.name] = _union(mine, theirs)
            elif own.name == "error_marks":
                values[own.name] = {label: _union(mine.get(label, ()), theirs.get(label, ()))
                                    for label in [*mine, *theirs]}
            elif isinstance(mine, Mapping):
                values[own.name] = {**mine, **theirs}
            else:
                values[own.name] = theirs if theirs is not None else mine
        return Harness(name=name, **values)


def _union(first: tuple[str, ...], second: tuple[str, ...]) -> tuple[str, ...]:
    return first + tuple(item for item in second if item not in first)


EMPTY = Harness(name="none")
"""The profile of a trajectory whose adapter names no harness."""
