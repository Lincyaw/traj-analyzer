from __future__ import annotations

import json
import os
import select
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Literal

import libdash
import shasta.ast_node as ast
from shasta.json_to_ast import to_ast_node

Join = Literal["start", "then", "and", "or", "pipe"]
"""How a command follows the one before it: first of its script, `;`, `&&`, `||`, or `|`."""
Nesting = Literal["subshell", "group", "loop", "condition", "case", "function", "substitution", "shell", "negation"]
RedirectKind = Literal["in", "out", "append", "clobber", "readwrite", "heredoc", "dup"]
SHELLS = frozenset({"sh", "bash", "zsh", "dash", "ksh"})
OUT_KINDS: dict[str, RedirectKind] = {"To": "out", "From": "in", "Append": "append", "Clobber": "clobber",
                                       "ReadWrite": "readwrite", "FromTo": "readwrite"}


@dataclass(frozen=True)
class Redirect:
    kind: RedirectKind
    fd: int | None
    target: str | None = None
    """The file for a file redirect, the descriptor for a dup."""
    heredoc: str | None = None


@dataclass(frozen=True)
class Command:
    """One simple command of a script, in execution order, with how it is joined and where it sits."""

    argv: tuple[str, ...]
    """The words of the command, quotes removed; a variable reads `${NAME}` and a substitution `$(...)`."""
    assignments: tuple[tuple[str, str], ...] = ()
    redirects: tuple[Redirect, ...] = ()
    join: Join = "start"
    background: bool = False
    nesting: tuple[Nesting, ...] = ()
    """The constructs the command sits in, outermost first."""
    cwd: str | None = None
    """The directory a `cd` earlier in the script moved to, as written."""
    expanded: bool = False
    """Whether a word holds a variable, substitution or glob, so that argv is not the literal invocation."""

    @property
    def program(self) -> str | None:
        return self.argv[0].rsplit("/", 1)[-1] if self.argv else None


@dataclass(frozen=True)
class Script:
    commands: tuple[Command, ...]
    text: str


@dataclass
class _Walk:
    commands: list[Command] = field(default_factory=list)
    cwd: str | None = None
    expanded: bool = False


_initialized = False


class ParseError(ValueError):
    """The text is not shell dash accepts, or the parser died on it."""


def parse(text: str) -> Script:
    """Parse shell text with the dash parser into its simple commands; raises `ParseError` for text dash rejects.

    The dash parser runs in a helper process: it keeps state across calls that a rejected text corrupts, and on
    some texts it aborts, so the process is started again after either.
    """
    return _parser().parse(text)


class _Parser:
    """The helper process that runs the dash parser, spoken to with one JSON line per text."""

    TIMEOUT_S = 20.0

    def __init__(self) -> None:
        self.process: subprocess.Popen[str] | None = None

    def start(self) -> subprocess.Popen[str]:
        if self.process is None or self.process.poll() is not None:
            self.process = subprocess.Popen([sys.executable, "-m", "traj_analyzer.shell.worker"], stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        return self.process

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.kill()
            self.process.wait()
        self.process = None

    def parse(self, text: str) -> Script:
        process = self.start()
        assert process.stdin is not None and process.stdout is not None
        try:
            process.stdin.write(json.dumps({"text": text}) + "\n")
            process.stdin.flush()
            ready, _, _ = select.select([process.stdout], [], [], self.TIMEOUT_S)
            line = process.stdout.readline() if ready else ""
        except (BrokenPipeError, OSError):
            line = ""
        if not line:
            # The parser aborted or hung on this text; a fresh process serves the next one.
            self.stop()
            raise ParseError("the shell parser died on the text")
        answer = json.loads(line)
        if "error" in answer:
            # A rejected text leaves the parser in a state that corrupts later parses.
            self.stop()
            raise ParseError(answer["error"])
        commands = tuple(Command(**{**item, "assignments": tuple(tuple(a) for a in item["assignments"]),
                                    "redirects": tuple(Redirect(**r) for r in item["redirects"]),
                                    "argv": tuple(item["argv"]), "nesting": tuple(item["nesting"])})
                         for item in answer["commands"])
        return Script(commands=commands, text=text)


_PARSER: _Parser | None = None


def _parser() -> _Parser:
    global _PARSER
    if _PARSER is None:
        _PARSER = _Parser()
    return _PARSER


def parse_here(text: str) -> Script:
    """Parse in this process; `traj_analyzer.shell.worker` calls this, and `parse` is what the rest of the code uses."""
    global _initialized
    fd = os.memfd_create("shell")
    try:
        os.write(fd, (text.rstrip("\n") + "\n").encode("utf-8"))
        os.lseek(fd, 0, os.SEEK_SET)
        nodes = [to_ast_node(node) for node, _, _, _ in libdash.parse(f"/proc/self/fd/{fd}", init=not _initialized)]
    finally:
        os.close(fd)
    _initialized = True
    walk = _Walk()
    join: Join = "start"
    for node in nodes:
        join = _visit(node, walk, join, (), False)
    return Script(commands=tuple(walk.commands), text=text)


def _visit(node: ast.AstNode, walk: _Walk, join: Join, nesting: tuple[Nesting, ...], background: bool) -> Join:
    """Append the simple commands below `node`; return the join the next sibling starts with."""
    match node:
        case ast.CommandNode():
            _command(node, walk, join, nesting, background)
            return "then"
        case ast.PipeNode():
            for index, item in enumerate(node.items):
                _visit(item, walk, join if index == 0 else "pipe", nesting, background or bool(node.is_background))
            return "then"
        case ast.AndNode():
            _visit(node.left_operand, walk, join, nesting, background)
            return _visit(node.right_operand, walk, "and", nesting, background)
        case ast.OrNode():
            _visit(node.left_operand, walk, join, nesting, background)
            return _visit(node.right_operand, walk, "or", nesting, background)
        case ast.SemiNode():
            _visit(node.left_operand, walk, join, nesting, background)
            return _visit(node.right_operand, walk, "then", nesting, background)
        case ast.BackgroundNode():
            return _visit(node.node, walk, join, nesting, True)
        case ast.RedirNode():
            # Redirections of a compound command apply to its whole body; they are kept on the body's commands.
            return _visit(node.node, walk, join, nesting, background)
        case ast.NotNode():
            return _visit(node.body, walk, join, (*nesting, "negation"), background)
        case ast.SubshellNode():
            return _visit(node.body, walk, join, (*nesting, "subshell"), background)
        case ast.GroupNode():
            return _visit(node.body, walk, join, (*nesting, "group"), background)
        case ast.ForNode():
            walk.expanded |= _holds_expansion(node.argument)
            return _visit(node.body, walk, join, (*nesting, "loop"), background)
        case ast.WhileNode():
            _visit(node.test, walk, join, (*nesting, "loop"), background)
            return _visit(node.body, walk, "then", (*nesting, "loop"), background)
        case ast.IfNode():
            _visit(node.cond, walk, join, (*nesting, "condition"), background)
            _visit(node.then_b, walk, "then", (*nesting, "condition"), background)
            if node.else_b is not None:
                _visit(node.else_b, walk, "then", (*nesting, "condition"), background)
            return "then"
        case ast.CaseNode():
            for item in node.cases:
                if item.get("cbody") is not None:
                    _visit(item["cbody"], walk, join, (*nesting, "case"), background)
            return "then"
        case ast.DefunNode():
            return _visit(node.body, walk, join, (*nesting, "function"), background)
        case _:
            raise ValueError(f"unhandled shell node {type(node).__name__}")


def _command(node: ast.CommandNode, walk: _Walk, join: Join, nesting: tuple[Nesting, ...], background: bool) -> None:
    before = len(walk.commands)
    assignments = tuple((assignment.var, _word(assignment.val, walk, nesting, background))
                        for assignment in node.assignments)
    argv = tuple(_word(argument, walk, nesting, background) for argument in node.arguments)
    redirects = tuple(_redirect(redirect, walk, nesting, background) for redirect in node.redir_list)
    expanded = any(_holds_expansion(part) for part in (*node.arguments, *(a.val for a in node.assignments)))
    command = Command(argv=argv, assignments=assignments, redirects=redirects, join=join, background=background,
                      nesting=nesting, cwd=walk.cwd, expanded=expanded)
    # Substitutions inside the words were walked first; the command itself comes before them in execution order.
    walk.commands.insert(before, command)
    if argv and command.program in ("cd", "pushd") and not nesting:
        walk.cwd = argv[1] if len(argv) > 1 else "~"
    # `sh -c '<script>'`, possibly behind a wrapper such as timeout, runs a script of its own.
    for index in range(len(argv) - 2):
        if argv[index].rsplit("/", 1)[-1] in SHELLS and argv[index + 1] == "-c":
            _nested_script(argv[index + 2], walk, (*nesting, "shell"), background)
            break


def _nested_script(text: str, walk: _Walk, nesting: tuple[Nesting, ...], background: bool) -> None:
    inner = parse_here(text)
    for command in inner.commands:
        walk.commands.append(Command(argv=command.argv, assignments=command.assignments, redirects=command.redirects,
                                     join=command.join, background=background or command.background,
                                     nesting=(*nesting, *command.nesting), cwd=command.cwd or walk.cwd,
                                     expanded=command.expanded))


def _redirect(node: ast.AstNode, walk: _Walk, nesting: tuple[Nesting, ...], background: bool) -> Redirect:
    fd = _fd(node.fd)
    match node:
        case ast.FileRedirNode():
            return Redirect(kind=OUT_KINDS[str(node.redir_type)], fd=fd,
                            target=_word(node.arg, walk, nesting, background))
        case ast.HeredocRedirNode():
            return Redirect(kind="heredoc", fd=fd, heredoc=_word(node.arg, walk, nesting, background))
        case ast.DupRedirNode():
            return Redirect(kind="dup", fd=fd, target=_text(node.arg))
        case _:
            return Redirect(kind="dup", fd=fd)


def _fd(value: object) -> int | None:
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], int):
        return value[1]
    return value if isinstance(value, int) else None


def _word(parts: object, walk: _Walk, nesting: tuple[Nesting, ...], background: bool) -> str:
    """The text of a word with quotes removed; commands inside substitutions are walked as they are met."""
    if isinstance(parts, str):
        return parts
    out: list[str] = []
    raw = bytearray()

    def flush() -> None:
        # The parser hands text back one byte at a time; the bytes of a word are UTF-8.
        if raw:
            out.append(raw.decode("utf-8", errors="replace"))
            raw.clear()

    for part in parts:  # type: ignore[union-attr]
        if isinstance(part, ast.CArgChar | ast.EArgChar):
            raw.append(part.char % 256)
            continue
        flush()
        match part:
            case ast.QArgChar():
                out.append(_word(part.arg, walk, nesting, background))
            case ast.VArgChar():
                out.append("${" + str(part.var) + "}")
            case ast.BArgChar():
                _visit(part.node, walk, "start", (*nesting, "substitution"), background)
                out.append("$(" + part.node.pretty() + ")")
            case ast.AArgChar():
                out.append("$((" + _text(part.arg) + "))")
            case ast.TArgChar():
                out.append(part.pretty())
            case _:
                out.append(part.pretty())
    flush()
    return "".join(out)


def _text(parts: object) -> str:
    if isinstance(parts, str | int):
        return str(parts)
    if hasattr(parts, "pretty"):
        return parts.pretty()
    return "".join(_text(part) for part in parts)  # type: ignore[union-attr]


def _holds_expansion(parts: object) -> bool:
    if isinstance(parts, str):
        return False
    for part in parts:  # type: ignore[union-attr]
        if isinstance(part, ast.VArgChar | ast.BArgChar | ast.AArgChar | ast.TArgChar):
            return True
        if isinstance(part, ast.QArgChar) and _holds_expansion(part.arg):
            return True
        if isinstance(part, ast.CArgChar) and chr(part.char) in "*?":
            return True
    return False
