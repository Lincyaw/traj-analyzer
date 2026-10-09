from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from traj_analyzer.shell.parsing import Command

Act = Literal["inspect", "edit", "test", "build", "install", "run", "vcs", "network", "process", "shell", "other"]
"""What a command is for."""
Effect = Literal["none", "workspace", "history", "destructive", "remote", "environment", "unknown"]
"""What a command does to the world outside the agent: nothing; a change of files in the workspace; a change of
version-control history; a loss of data or state that is hard to get back; a change outside the machine; a change
of the machine's environment such as installed packages; or an unknown change, for a program this module does not
know."""
SEVERITY: dict[Effect, int] = {"none": 0, "workspace": 1, "environment": 2, "history": 3, "destructive": 4,
                               "remote": 5, "unknown": 1}


@dataclass(frozen=True)
class Spec:
    act: Act
    effect: Effect = "none"
    paths_from: int | None = 0
    """Index of the first operand that names a path; None when operands are not paths."""
    subcommands: dict[str, Spec] = field(default_factory=dict)
    """For programs whose first operand chooses what they do, such as git."""
    value_flags: frozenset[str] = frozenset()
    """Flags that take the next word as their value."""
    forced_by: frozenset[str] = frozenset()
    """Flags that make the command discard or overwrite without asking."""
    destructive_by: frozenset[str] = frozenset()
    """Flags that make the effect destructive."""
    remote_by: frozenset[str] = frozenset()
    """Flags that make the effect remote, such as a method that changes a server."""


@dataclass(frozen=True)
class Description:
    """What one simple command does, as far as the program is known."""

    program: str
    subcommand: str | None
    act: Act
    effect: Effect
    flags: tuple[str, ...]
    options: tuple[tuple[str, str], ...]
    operands: tuple[str, ...]
    paths: tuple[str, ...]
    """Operands and redirect targets the command reads or writes."""
    forced: bool
    wrappers: tuple[str, ...]
    """Programs that only ran the command, such as sudo or timeout."""
    known: bool


def _git() -> dict[str, Spec]:
    read = Spec("vcs", "none", None)
    workspace = Spec("vcs", "workspace", None)
    history = Spec("vcs", "history", None)
    destructive = Spec("vcs", "destructive", None)
    return {
        **{name: read for name in ("status", "diff", "log", "show", "blame", "rev-parse", "ls-files", "ls-tree",
                                    "merge-base", "describe", "shortlog", "reflog", "cat-file", "check-ignore",
                                    "rev-list", "name-rev", "grep", "ls-remote", "fetch", "remote", "config",
                                    "var", "help", "version", "diff-tree", "diff-index", "for-each-ref",
                                    "symbolic-ref", "count-objects", "fsck", "stash list", "worktree list")},
        "add": workspace, "mv": workspace, "apply": workspace, "stash": workspace, "init": workspace,
        "clone": Spec("vcs", "workspace", None),
        "switch": Spec("vcs", "workspace", None, destructive_by=frozenset({"--discard-changes", "-f"})),
        "checkout": Spec("vcs", "workspace", None, forced_by=frozenset({"-f", "--force"}),
                         destructive_by=frozenset({"-f", "--force", "--", "."})),
        "restore": Spec("vcs", "destructive", None),
        "branch": Spec("vcs", "workspace", None, value_flags=frozenset({"-u", "--set-upstream-to"}),
                       destructive_by=frozenset({"-D", "-d", "--delete", "-M", "-m"})),
        "tag": Spec("vcs", "history", None, value_flags=frozenset({"-m"}), destructive_by=frozenset({"-d"})),
        "worktree": Spec("vcs", "workspace", None, destructive_by=frozenset({"remove", "prune"})),
        "rm": Spec("vcs", "destructive", None, destructive_by=frozenset()),
        "clean": Spec("vcs", "destructive", None, forced_by=frozenset({"-f", "--force"})),
        "reset": Spec("vcs", "history", None, forced_by=frozenset({"--hard"}), destructive_by=frozenset({"--hard"})),
        "commit": Spec("vcs", "history", None, value_flags=frozenset({"-m", "--message", "-F", "--file", "-C", "-c"}),
                       forced_by=frozenset({"--amend", "--no-verify", "-n"})),
        "merge": history, "rebase": Spec("vcs", "history", None, forced_by=frozenset({"--autosquash", "-i"})),
        "cherry-pick": history, "revert": history, "am": history, "notes": history,
        "push": Spec("vcs", "remote", None, value_flags=frozenset({"-o", "--push-option"}),
                     forced_by=frozenset({"--force", "-f", "--force-with-lease", "--delete", "-d"}),
                     destructive_by=frozenset({"--force", "-f", "--delete", "-d"})),
        "pull": Spec("vcs", "history", None, forced_by=frozenset({"--force", "-f"})),
        "submodule": Spec("vcs", "workspace", None), "sparse-checkout": workspace, "lfs": read,
        "bisect": Spec("vcs", "workspace", None), "format-patch": read, "archive": read, "gc": destructive,
        "filter-branch": destructive, "update-index": workspace, "checkout-index": workspace,
    }


def _gh() -> dict[str, Spec]:
    read = Spec("network", "none", None)
    write = Spec("network", "remote", None)
    nested = {
        "pr": {"view": read, "list": read, "checks": read, "diff": read, "status": read,
               "create": write, "merge": write, "comment": write, "review": write, "close": write, "edit": write,
               "ready": write, "checkout": Spec("vcs", "workspace", None), "reopen": write},
        "issue": {"view": read, "list": read, "status": read, "create": write, "comment": write, "close": write,
                  "edit": write, "reopen": write, "delete": write},
        "repo": {"view": read, "list": read, "clone": Spec("vcs", "workspace", None), "create": write,
                 "delete": write, "edit": write, "fork": write, "sync": write},
        "run": {"view": read, "list": read, "watch": read, "download": read, "rerun": write, "cancel": write},
        "release": {"view": read, "list": read, "download": read, "create": write, "delete": write,
                    "upload": write, "edit": write},
        "workflow": {"view": read, "list": read, "run": write, "enable": write, "disable": write},
        "gist": {"view": read, "list": read, "create": write, "delete": write, "edit": write},
        "secret": {"list": read, "set": write, "delete": write},
        "label": {"list": read, "create": write, "delete": write, "edit": write},
    }
    out: dict[str, Spec] = {group: Spec("network", "none", None, subcommands=table) for group, table in nested.items()}
    out["api"] = Spec("network", "none", None, value_flags=frozenset({"-X", "--method", "-f", "-F", "--input"}),
                      remote_by=frozenset({"-X", "--method", "-f", "-F", "--input"}))
    out["auth"] = Spec("network", "environment", None)
    out["search"] = read
    out["browse"] = read
    return out


def _packages(
    install: tuple[str, ...], test: tuple[str, ...], build: tuple[str, ...], run: tuple[str, ...],
    read: tuple[str, ...] = (), remove: tuple[str, ...] = (), publish: tuple[str, ...] = (),
) -> dict[str, Spec]:
    out: dict[str, Spec] = {}
    out.update({name: Spec("install", "environment", None) for name in install})
    out.update({name: Spec("test", "none", None) for name in test})
    out.update({name: Spec("build", "workspace", None) for name in build})
    out.update({name: Spec("run", "unknown", None) for name in run})
    out.update({name: Spec("inspect", "none", None) for name in read})
    out.update({name: Spec("install", "environment", None, forced_by=frozenset({"-y", "--yes"})) for name in remove})
    out.update({name: Spec("network", "remote", None) for name in publish})
    return out


def _npm_scripts() -> dict[str, Spec]:
    """`npm run <script>` and the like: the script's name tells what it is for."""
    return {
        "test": Spec("test", "none", None), "build": Spec("build", "workspace", None),
        "lint": Spec("build", "none", None), "typecheck": Spec("build", "none", None),
        "check": Spec("build", "none", None), "format": Spec("build", "workspace", None),
        "dev": Spec("run", "unknown", None), "start": Spec("run", "unknown", None),
    }


_NODE_MANAGER = {
    **_packages(install=("install", "i", "ci", "add", "update", "up", "upgrade", "link"),
                test=("test", "t"), build=(), run=(), read=("ls", "list", "outdated", "view", "info", "why", "audit"),
                remove=("remove", "rm", "uninstall", "un"), publish=("publish",)),
    "run": Spec("run", "unknown", None, subcommands=_npm_scripts()),
    "exec": Spec("run", "unknown", None), "dlx": Spec("run", "unknown", None), "x": Spec("run", "unknown", None),
    "test": Spec("test", "none", None), "build": Spec("build", "workspace", None),
    "lint": Spec("build", "none", None), "init": Spec("edit", "workspace", None),
}

PROGRAMS: dict[str, Spec] = {
    # Looking at files and the workspace.
    **{name: Spec("inspect", "none", 0) for name in (
        "cat", "ls", "head", "tail", "less", "more", "wc", "find", "fd", "tree", "stat", "file", "du", "df",
        "readlink", "realpath", "basename", "dirname", "diff", "cmp", "nl", "xxd", "od", "hexdump", "strings",
        "bat", "exa", "eza", "column", "fold", "tac", "rev", "md5sum", "sha256sum", "shasum", "jq", "yq",
        "sort", "uniq", "cut", "tr", "paste", "comm", "join", "tee", "zcat", "unzip", "tar", "ldd", "nm",
        "objdump", "pdftotext")},
    **{name: Spec("inspect", "none", 1) for name in ("grep", "rg", "ag", "ack", "egrep", "fgrep", "awk", "gawk",
                                                     "sed", "perl", "xargs")},
    **{name: Spec("inspect", "none", None) for name in (
        "pwd", "which", "type", "whoami", "id", "hostname", "uname", "date", "env", "printenv", "echo", "printf",
        "true", "false", "test", "[", "seq", "expr", "bc", "yes", "ps", "top", "lsof", "netstat", "ss", "free",
        "uptime", "nproc", "locale", "man", "help", "history", "jobs", "git-lfs", "tput", "clear", "[[", "pgrep",
        "base64", "pdfinfo", "mdls", "sw_vers", "xcrun", "getconf", "ulimit", "ps1", "caller")},
    # Changing files.
    "rm": Spec("edit", "destructive", 0, forced_by=frozenset({"-f", "--force"})),
    "rmdir": Spec("edit", "destructive", 0), "shred": Spec("edit", "destructive", 0),
    "mv": Spec("edit", "workspace", 0, forced_by=frozenset({"-f", "--force"})),
    "cp": Spec("edit", "workspace", 0, forced_by=frozenset({"-f", "--force"})),
    "mkdir": Spec("edit", "workspace", 0), "touch": Spec("edit", "workspace", 0),
    "mktemp": Spec("edit", "workspace", None),
    "ln": Spec("edit", "workspace", 0, forced_by=frozenset({"-f", "--force"})),
    "chmod": Spec("edit", "workspace", 1), "chown": Spec("edit", "workspace", 1),
    "truncate": Spec("edit", "destructive", 0), "dd": Spec("edit", "destructive", None),
    "patch": Spec("edit", "workspace", 0), "install": Spec("edit", "workspace", 0),
    "rsync": Spec("network", "workspace", 0, destructive_by=frozenset({"--delete"})),
    "scp": Spec("network", "remote", 0), "sftp": Spec("network", "remote", None),
    "zip": Spec("edit", "workspace", 0), "gzip": Spec("edit", "workspace", 0), "gunzip": Spec("edit", "workspace", 0),
    # Tests, builds and checks.
    **{name: Spec("test", "none", None) for name in (
        "pytest", "py.test", "jest", "vitest", "mocha", "ava", "tox", "nox", "phpunit", "rspec", "ctest",
        "playwright", "cypress", "unittest", "busted", "minitest", "ginkgo", "gotestsum", "nextest")},
    **{name: Spec("build", "none", None) for name in (
        "tsc", "ruff", "mypy", "pyright", "flake8", "pylint", "eslint", "biome", "stylelint", "shellcheck",
        "golangci-lint", "staticcheck", "clippy", "ktlint", "detekt", "rubocop", "clang-tidy", "cppcheck",
        "hadolint", "markdownlint", "vale", "ty", "basedpyright", "luacheck", "swiftlint")},
    **{name: Spec("build", "workspace", None) for name in (
        "make", "cmake", "ninja", "gradle", "gradlew", "mvn", "ant", "xcodebuild", "swiftc", "gcc", "g++",
        "clang", "clang++", "rustc", "javac", "kotlinc", "esbuild", "webpack", "vite", "rollup", "parcel",
        "turbo", "nx", "lerna", "prettier", "black", "isort", "gofmt", "goimports", "rustfmt", "clang-format",
        "latexmk", "pdflatex", "xelatex", "lualatex", "bibtex", "biber", "pandoc", "mdbook", "sphinx-build",
        "mkdocs", "protoc", "buf", "wasm-pack", "zig", "nim", "ghc", "stack", "cabal", "dune", "opam",
        "sbt", "lein", "mix", "rebar3", "dotnet", "msbuild", "nuget", "swift", "flutter", "dart", "meson",
        "bazel", "buck2", "scons", "tup", "tsup", "tsx", "swc", "babel")},
    # Installing.
    **{name: Spec("install", "environment", None) for name in (
        "pip", "pip3", "pipx", "brew", "apt", "apt-get", "dnf", "yum", "pacman", "apk", "zypper", "snap",
        "conda", "mamba", "micromamba", "rustup", "nvm", "pyenv", "rbenv", "asdf", "mise", "volta", "fnm",
        "gem", "bundle", "bundler", "composer", "choco", "winget", "scoop", "port", "nix", "nix-env",
        "pnpm", "yarn", "bun", "npm", "cargo-binstall")},
    # Running programs.
    **{name: Spec("run", "unknown", None) for name in (
        "python", "python3", "node", "deno", "ruby", "java", "php", "lua", "Rscript", "julia", "ts-node",
        "npx", "uvx", "bunx", "pnpx", "poetry", "pipenv", "hatch", "pdm", "rye", "sh", "bash", "zsh", "dash",
        "fish", "pwsh", "powershell", "osascript", "open", "xdg-open", "code", "cursor", "vim", "nvim",
        "emacs", "nano", "streamlit", "gunicorn", "uvicorn", "flask", "django-admin", "manage.py", "rails",
        "rake", "mix", "iex", "ghci", "irb", "scala", "sqlite3", "psql", "mysql", "redis-cli", "mongosh",
        "duckdb", "ollama", "claude", "codex", "gemini", "aider", "entire", "rtk", "surf", "omx")},
    # Network.
    "curl": Spec("network", "none", None, value_flags=frozenset({"-X", "--request", "-d", "--data", "-H", "--header",
                                                                 "-o", "--output", "-u", "-F", "--form", "-T"}),
                 remote_by=frozenset({"-X", "--request", "-d", "--data", "--data-raw", "--data-binary", "-F", "--form",
                                      "-T", "--upload-file"})),
    **{name: Spec("network", "none", None) for name in ("wget", "http", "httpie", "nc", "ping", "dig", "nslookup")},
    **{name: Spec("network", "unknown", None) for name in ("ssh", "aws", "gcloud", "az", "kubectl", "helm", "terraform",
                                                           "wrangler")},
    **{name: Spec("network", "remote", None) for name in ("vercel", "netlify", "fly", "heroku")},
    "docker": Spec("run", "unknown", None, subcommands={
        **{name: Spec("inspect", "none", None) for name in ("ps", "images", "logs", "inspect")},
        "build": Spec("build", "environment", None),
        "compose": Spec("run", "environment", None), "run": Spec("run", "environment", None),
        "exec": Spec("run", "unknown", None), "push": Spec("network", "remote", None),
        "pull": Spec("install", "environment", None),
        "rm": Spec("process", "destructive", None), "rmi": Spec("process", "destructive", None),
        "stop": Spec("process", "environment", None), "kill": Spec("process", "environment", None),
        "system": Spec("process", "destructive", None)}),
    # Processes and the shell.
    **{name: Spec("process", "none", None) for name in ("sleep", "wait", "timeout", "watch", "fg", "bg", "tmux",
                                                        "screen", "disown")},
    **{name: Spec("process", "environment", None) for name in ("kill", "pkill", "killall", "systemctl", "launchctl",
                                                               "service")},
    "just": Spec("build", "workspace", None, subcommands={
        "test": Spec("test", "none", None), "check": Spec("test", "none", None), "lint": Spec("build", "none", None),
        "fmt": Spec("build", "workspace", None), "build": Spec("build", "workspace", None)}),
    "flyctl": Spec("network", "remote", None),
    **{name: Spec("shell", "none", None) for name in (
        "cd", "pushd", "popd", "export", "unset", "set", "shift", "source", ".", "alias", "unalias", "exit",
        "return", "break", "continue", "eval", "exec", "read", "trap", "local", "declare", "typeset", "let",
        "ulimit", "umask", "hash", "command", "builtin", "getopts", ":", "nohup", "nice", "time", "sudo",
        "doas", "env", "stdbuf", "unbuffer", "script", "caffeinate", "chroot")},
    # Version control and tooling with subcommands.
    "git": Spec("vcs", "none", None, value_flags=frozenset({"-C", "-c", "--git-dir", "--work-tree"}),
                subcommands=_git()),
    "gh": Spec("network", "none", None, subcommands=_gh()),
    **{name: Spec("vcs", "unknown", None) for name in ("hg", "svn", "jj")},
    "cargo": Spec("build", "workspace", None, subcommands=_packages(
        install=("add", "install", "update"), test=("test", "nextest"),
        build=("build", "check", "clippy", "fmt", "doc"), run=("run", "bench"), read=("tree", "metadata", "search"),
        remove=("remove", "rm"), publish=("publish",))),
    "go": Spec("build", "workspace", None, subcommands={
        **_packages(install=("get", "install", "mod"), test=("test",), build=("build", "vet", "fmt", "generate"),
                    run=("run",), read=("list", "version", "env", "doc"))}),
    "uv": Spec("run", "unknown", None, subcommands={
        **_packages(install=("add", "sync", "pip", "lock", "venv", "tool", "python"), test=(), build=("build",),
                    run=("run",), read=("tree",), remove=("remove",), publish=("publish",))}),
    "npm": Spec("install", "environment", None, subcommands=_NODE_MANAGER),
    "pnpm": Spec("install", "environment", None, subcommands=_NODE_MANAGER),
    "yarn": Spec("install", "environment", None, subcommands=_NODE_MANAGER),
    "bun": Spec("install", "environment", None, subcommands=_NODE_MANAGER),
    "mise": Spec("install", "environment", None, subcommands={
        **_packages(install=("install", "use", "i", "up", "upgrade"), test=(), build=(), run=("run", "exec", "x"),
                    read=("ls", "list", "which", "env", "doctor", "current"))}),
    "make": Spec("build", "workspace", None, subcommands={
        "test": Spec("test", "none", None), "check": Spec("test", "none", None), "lint": Spec("build", "none", None),
        "clean": Spec("edit", "destructive", None), "install": Spec("install", "environment", None)}),
    "python": Spec("run", "unknown", None, value_flags=frozenset({"-m", "-c"})),
    "python3": Spec("run", "unknown", None, value_flags=frozenset({"-m", "-c"})),
}
"""What the program named by a command's first word does, by the program's base name."""

WRAPPERS: dict[str, int] = {
    "sudo": 0, "doas": 0, "env": 0, "time": 0, "nohup": 0, "nice": 0, "stdbuf": 0, "unbuffer": 0, "command": 0,
    "exec": 0, "builtin": 0, "caffeinate": 0, "timeout": 1, "xargs": 0,
}
"""Programs that run the command that follows their flags; the value is the number of positional values they take."""
RUNNERS: dict[tuple[str, str | None], int] = {
    ("uv", "run"): 0, ("uvx", None): 0, ("npx", None): 0, ("bunx", None): 0, ("pnpx", None): 0, ("poetry", "run"): 0,
    ("pipx", "run"): 0, ("pdm", "run"): 0, ("hatch", "run"): 0, ("mise", "exec"): 0, ("mise", "x"): 0,
    ("conda", "run"): 0, ("bundle", "exec"): 0, ("pnpm", "exec"): 0, ("npm", "exec"): 0, ("yarn", "exec"): 0,
    ("cargo", "run"): -1, ("go", "run"): -1,
}
"""Tool runners whose remaining words are the command they run; -1 marks runners whose target is not a program."""
PYTHON_MODULES: dict[str, Spec] = {
    "pytest": Spec("test", "none", None), "unittest": Spec("test", "none", None),
    **{name: Spec("build", "none", None) for name in ("mypy", "ruff", "flake8")},
    "black": Spec("build", "workspace", None), "build": Spec("build", "workspace", None),
    "pip": Spec("install", "environment", None), "venv": Spec("install", "environment", None),
    "http.server": Spec("run", "none", None), "json.tool": Spec("inspect", "none", None),
    "twine": Spec("network", "remote", None),
}
TEST_FLAGS = frozenset({"--test", "-t"})


def describe(command: Command) -> Description:
    """What the command does: its program after unwrapping, its subcommand, act, effect and operands."""
    wrappers: list[str] = []
    argv = list(command.argv)
    program = _program(argv)
    while program in WRAPPERS and len(argv) > 1:
        wrappers.append(program)
        skip = 1 + WRAPPERS[program]
        rest = argv[skip:]
        # The flags and assignments of a wrapper come before the command it runs.
        while rest and (rest[0].startswith("-") or ("=" in rest[0] and not rest[0].startswith("-"))):
            rest = rest[1:]
        if not rest:
            break
        argv = rest
        program = _program(argv)
    if program is None:
        return Description(program="", subcommand=None, act="shell", effect="none", flags=(), options=(), operands=(),
                           paths=tuple(_redirect_paths(command)), forced=False, wrappers=tuple(wrappers), known=False)
    spec = PROGRAMS.get(program)
    flags, options, operands = _split(argv[1:], spec.value_flags if spec else frozenset())
    subcommand: str | None = None
    while spec is not None and spec.subcommands and operands and operands[0] in spec.subcommands:
        subcommand = operands[0] if subcommand is None else f"{subcommand} {operands[0]}"
        spec = spec.subcommands[operands[0]]
        more_flags, more_options, operands = _split(operands[1:], spec.value_flags)
        flags, options = flags + more_flags, options + more_options
    if (program, subcommand) in RUNNERS and RUNNERS[(program, subcommand)] == 0 and operands:
        inner = Command(argv=tuple(operands), redirects=command.redirects, join=command.join,
                        background=command.background, nesting=command.nesting, cwd=command.cwd,
                        expanded=command.expanded)
        described = describe(inner)
        return Description(**{**described.__dict__, "wrappers": (*wrappers, program, *described.wrappers)})
    if program in ("python", "python3") and ("-m", ) and any(name == "-m" for name, _ in options):
        module = next(value for name, value in options if name == "-m")
        if module in PYTHON_MODULES:
            spec, subcommand = PYTHON_MODULES[module], module
    effect: Effect = spec.effect if spec else "unknown"
    act: Act = spec.act if spec else "other"
    forced = bool(spec and (set(flags) & spec.forced_by))
    if spec and (set(flags) & spec.destructive_by or {o for o, _ in options} & spec.destructive_by
                 or set(operands) & spec.destructive_by):
        effect = "destructive"
    if spec and ({o for o, _ in options} & spec.remote_by or set(flags) & spec.remote_by):
        effect = "remote"
    if subcommand is None and spec and spec.act == "build" and program in ("make", "cargo", "go") and operands:
        act = "build"
    if spec and spec.paths_from is not None:
        paths = list(operands[spec.paths_from:])
    elif act == "vcs":
        # git names paths among refs and branches; a word with a slash or a dot is taken for a path.
        paths = [word for word in operands if _looks_like_path(word)]
    else:
        paths = []
    paths += _redirect_paths(command)
    return Description(program=program, subcommand=subcommand, act=act, effect=_with_redirects(command, effect),
                       flags=tuple(flags), options=tuple(options), operands=tuple(operands), paths=tuple(paths),
                       forced=forced, wrappers=tuple(wrappers), known=spec is not None)


def _looks_like_path(word: str) -> bool:
    if word.startswith(("-", "HEAD", "@", "origin/", "upstream/", "refs/")) or ":" in word or ".." in word:
        return False
    return "/" in word or "." in word


def _program(argv: list[str]) -> str | None:
    if not argv:
        return None
    name = argv[0].rsplit("/", 1)[-1]
    return name


def _split(words: list[str], value_flags: frozenset[str]) -> tuple[list[str], list[tuple[str, str]], list[str]]:
    """Flags, options with their values, and operands; `--` ends the flags, and operands may follow flags."""
    flags: list[str] = []
    options: list[tuple[str, str]] = []
    operands: list[str] = []
    index = 0
    while index < len(words):
        word = words[index]
        if word == "--":
            # The separator is kept as a flag: for git checkout it tells a path from a branch.
            flags.append(word)
            operands.extend(words[index + 1:])
            break
        if word.startswith("--") and "=" in word:
            name, value = word.split("=", 1)
            options.append((name, value))
        elif word in value_flags and index + 1 < len(words):
            options.append((word, words[index + 1]))
            index += 1
        elif word.startswith("--") or (word.startswith("-") and len(word) == 2 and word != "-"):
            flags.append(word)
        elif word.startswith("-") and len(word) > 2 and word[1] != "-" and word[1:].isalpha():
            # Combined short flags such as -rf.
            flags.extend(f"-{letter}" for letter in word[1:])
        elif word.startswith("-") and len(word) > 2 and word[1] != "-":
            flags.append(word)
        else:
            operands.append(word)
        index += 1
    return flags, options, operands


def _redirect_paths(command: Command) -> list[str]:
    return [r.target for r in command.redirects if r.kind in ("out", "append", "clobber", "readwrite", "in")
            and r.target is not None and r.target != "/dev/null"]


def _with_redirects(command: Command, effect: Effect) -> Effect:
    """A redirect that writes a file is a change of the workspace, whatever the command."""
    writes = any(r.kind in ("out", "append", "clobber", "readwrite") and r.target not in (None, "/dev/null")
                 for r in command.redirects)
    if writes and SEVERITY[effect] < SEVERITY["workspace"]:
        return "workspace"
    return effect
