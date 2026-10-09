from __future__ import annotations

import pytest

from traj_analyzer.shell import ParseError, describe, parse


def argvs(text: str) -> list[tuple[str, ...]]:
    return [command.argv for command in parse(text).commands]


def test_parse_splits_a_script_into_simple_commands_with_their_joins() -> None:
    script = parse("cd /a/b && git -C sub push --force origin main 2>&1 | tail -5")
    cd, push, tail = script.commands
    assert (cd.argv, cd.join, cd.cwd) == (("cd", "/a/b"), "start", None)
    assert (push.argv[:2], push.join, push.cwd) == (("git", "-C"), "and", "/a/b")
    assert push.redirects[0].kind == "dup" and tail.join == "pipe"


def test_parse_keeps_quotes_out_and_marks_expansions() -> None:
    (rm, echo) = parse('rm -rf "$HOME/x"/* ; echo done > out.txt').commands
    assert rm.argv == ("rm", "-rf", "${HOME}/x/*") and rm.expanded
    assert (echo.argv, echo.join, echo.expanded) == (("echo", "done"), "then", False)
    assert echo.redirects[0].kind == "out" and echo.redirects[0].target == "out.txt"


def test_parse_walks_substitutions_loops_and_nested_shells() -> None:
    assert argvs("x=$(git rev-parse HEAD); echo $x") == [(), ("git", "rev-parse", "HEAD"), ("echo", "${x}")]
    substitution = parse("x=$(git rev-parse HEAD)").commands[1]
    assert substitution.nesting == ("substitution",)
    loop = parse("for f in *.py; do wc -l $f; done").commands[0]
    assert (loop.argv, loop.nesting) == (("wc", "-l", "${f}"), ("loop",))
    outer, inner = parse('timeout 30 bash -c "pytest -q tests/"').commands
    assert outer.argv[2:] == ("bash", "-c", "pytest -q tests/")
    assert (inner.argv, inner.nesting) == (("pytest", "-q", "tests/"), ("shell",))
    heredoc = parse("python3 - <<EOF\nprint(1)\nEOF").commands[0]
    assert heredoc.redirects[0].kind == "heredoc" and heredoc.redirects[0].heredoc == "print(1)\n"
    background = parse("FOO=1 npm test -- --watch=false &").commands[0]
    assert (background.assignments, background.background) == ((("FOO", "1"),), True)


def test_parse_rejects_what_dash_rejects() -> None:
    with pytest.raises(ParseError, match="Parsing"):
        parse("if then")
    # The parser is started again after a rejected text, and the next text parses.
    assert argvs("echo after") == [("echo", "after")]
    # A heredoc inside a substitution, with text beyond ASCII, is read as UTF-8.
    (commit, cat) = parse('git commit -m "$(cat <<\'EOF\'\nfeat: 배틀 모드\nEOF\n)"').commands
    assert cat.argv == ("cat",) and "배틀" in commit.argv[3]


@pytest.mark.parametrize(("text", "expected"), [
    ("git status --short", ("git", "status", "vcs", "none", False)),
    ("git push --force origin main", ("git", "push", "vcs", "destructive", True)),
    ("git push origin main", ("git", "push", "vcs", "remote", False)),
    ("git reset --hard HEAD~1", ("git", "reset", "vcs", "destructive", True)),
    ("git commit -m 'fix: x' --no-verify", ("git", "commit", "vcs", "history", True)),
    ("git checkout -- src/a.py", ("git", "checkout", "vcs", "destructive", False)),
    ("git checkout -b feature", ("git", "checkout", "vcs", "workspace", False)),
    ("rm -rf build/", ("rm", None, "edit", "destructive", True)),
    ("sudo rm -r /opt/x", ("rm", None, "edit", "destructive", False)),
    ("timeout 120 uv run pytest -q tests/test_a.py", ("pytest", None, "test", "none", False)),
    ("python3 -m pytest -x", ("python3", "pytest", "test", "none", False)),
    ("npm run build", ("npm", "run build", "build", "workspace", False)),
    ("pnpm test", ("pnpm", "test", "test", "none", False)),
    ("cargo add serde", ("cargo", "add", "install", "environment", False)),
    ("gh pr create --title x", ("gh", "pr create", "network", "remote", False)),
    ("gh pr view 12", ("gh", "pr view", "network", "none", False)),
    ("gh api -X POST repos/a/b/issues", ("gh", "api", "network", "remote", False)),
    ("curl -s https://x.test/", ("curl", None, "network", "none", False)),
    ("curl -X DELETE https://x.test/a", ("curl", None, "network", "remote", False)),
    ("echo hi > notes.txt", ("echo", None, "inspect", "workspace", False)),
    ("grep -rn foo src/ | head", ("grep", None, "inspect", "none", False)),
    ("someunknowntool --flag a", ("someunknowntool", None, "other", "unknown", False)),
])
def test_describe_names_program_subcommand_act_and_effect(text: str, expected: tuple) -> None:
    found = describe(parse(text).commands[0])
    assert (found.program, found.subcommand, found.act, found.effect, found.forced) == expected


def test_describe_keeps_operands_paths_and_wrappers() -> None:
    found = describe(parse("sudo env FOO=1 grep -n pattern a.py b.py").commands[0])
    assert (found.wrappers, found.program, found.flags) == (("sudo", "env"), "grep", ("-n",))
    assert found.operands == ("pattern", "a.py", "b.py")
    assert found.paths == ("a.py", "b.py")
    found = describe(parse("uv run pytest tests/ -k x").commands[0])
    assert (found.wrappers, found.program, found.known) == (("uv",), "pytest", True)
    found = describe(parse("git -C /repo log --oneline -5").commands[0])
    assert dict(found.options) == {"-C": "/repo"} and found.subcommand == "log"
