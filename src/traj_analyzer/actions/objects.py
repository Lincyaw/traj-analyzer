from __future__ import annotations

import posixpath
from dataclasses import dataclass
from typing import Literal

Kind = Literal["file", "directory", "glob", "pattern", "url", "agent", "task", "skill", "ref", "unknown"]
Role = Literal["source", "test", "config", "lock", "docs", "script", "data", "asset", "notebook", "generated",
               "harness", "other"]
"""What a file is for in a project, read off its path."""

SYSTEM_PREFIXES = ("/tmp/", "/usr/", "/etc/", "/opt/", "/proc/", "/dev/", "/nix/", "/var/", "/bin/", "/sbin/",
                   "/System/", "/Library/", "/Applications/", "/private/")
HARNESS_SEGMENTS = {".claude", ".codex", ".omx", ".agents", ".cursor", ".gemini", ".aider", ".windsurf", ".copilot"}
HARNESS_NAMES = {"CLAUDE.md", "AGENTS.md", "GEMINI.md", ".cursorrules", ".clinerules", "CODEX.md"}
GENERATED_SEGMENTS = {"node_modules", "dist", "build", "target", ".venv", "venv", "__pycache__", ".next", ".git",
                      "coverage", ".cache", ".tox", ".mypy_cache", ".ruff_cache", ".pytest_cache", "vendor", "out",
                      ".gradle", "DerivedData", ".terraform", "site-packages"}
LOCK_NAMES = {"package-lock.json", "pnpm-lock.yaml", "yarn.lock", "bun.lockb", "bun.lock", "uv.lock", "poetry.lock",
              "Cargo.lock", "go.sum", "Gemfile.lock", "composer.lock", "Pipfile.lock", "flake.lock"}
TEST_SEGMENTS = {"tests", "test", "__tests__", "spec", "specs", "e2e", "testdata", "fixtures", "integration"}
TEST_SUFFIXES = ("_test.py", "_test.go", "_test.rs", ".test.ts", ".test.tsx", ".test.js", ".test.jsx", ".test.mjs",
                 ".spec.ts", ".spec.tsx", ".spec.js", ".spec.jsx", "_spec.rb", "Test.java", "Tests.cs", "Test.kt",
                 "_test.exs", ".test.php", "Test.php")
CONFIG_NAMES = {"pyproject.toml", "package.json", "Cargo.toml", "go.mod", "Makefile", "Dockerfile", "Justfile",
                "setup.py", "setup.cfg", "tox.ini", "Gemfile", "Rakefile", "build.gradle", "pom.xml", "CMakeLists.txt",
                "flake.nix", "Pipfile", "requirements.txt", "composer.json", "deno.json", "bunfig.toml", "mise.toml",
                ".gitignore", ".gitattributes", ".editorconfig", ".dockerignore", "Procfile", "LICENSE"}
CONFIG_SUFFIXES = (".toml", ".yaml", ".yml", ".ini", ".cfg", ".conf", ".env", ".properties", ".plist", ".nix")
CONFIG_SEGMENTS = {".github", ".gitlab", "config", "configs", ".vscode", ".idea", ".devcontainer", ".circleci"}
DOCS_SUFFIXES = (".md", ".mdx", ".rst", ".txt", ".adoc", ".tex", ".bib", ".org")
DOCS_SEGMENTS = {"docs", "doc", "documentation"}
SCRIPT_SUFFIXES = (".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat", ".cmd")
DATA_SUFFIXES = (".csv", ".tsv", ".jsonl", ".parquet", ".sqlite", ".sqlite3", ".db", ".sql", ".xml", ".arrow",
                 ".feather", ".pkl", ".npy", ".npz", ".h5", ".hdf5", ".log")
ASSET_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp", ".pdf", ".mp3", ".mp4", ".wav", ".ttf",
                  ".woff", ".woff2", ".zip", ".tar", ".gz", ".bin", ".wasm")
SOURCE_SUFFIXES = (".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".go", ".rs", ".java", ".kt", ".kts",
                   ".scala", ".rb", ".php", ".c", ".h", ".cc", ".cpp", ".hpp", ".cs", ".swift", ".m", ".mm", ".dart",
                   ".ex", ".exs", ".erl", ".hs", ".ml", ".clj", ".lua", ".r", ".jl", ".zig", ".nim", ".vue", ".svelte",
                   ".html", ".css", ".scss", ".sass", ".less", ".graphql", ".proto", ".sol", ".tf", ".elm", ".fs",
                   ".v", ".sv", ".cu", ".astro", ".hbs", ".ejs", ".pug", ".wgsl", ".glsl", ".metal")


@dataclass(frozen=True)
class Object:
    """What an action acts on: a file or directory by its resolved path, a search pattern, a URL, an agent."""

    kind: Kind
    name: str
    role: Role | None = None
    """For a file or directory, what it is for."""
    relative: str | None = None
    """The path inside the project, when the project root is known and the path lies under it."""


def resolve(path: str, cwd: str | None, root: str | None) -> str:
    """The path as the project sees it: relative paths are joined to the directory the command ran in."""
    path = path.strip()
    if path.startswith("~"):
        return posixpath.normpath(path)
    if posixpath.isabs(path) or (cwd is None and root is None):
        return posixpath.normpath(path) if path else path
    base = cwd if cwd is not None and posixpath.isabs(cwd) else root
    if base is None:
        return posixpath.normpath(path)
    return posixpath.normpath(posixpath.join(base, path))


def object_of(path: str, cwd: str | None, root: str | None) -> Object:
    text = path.strip()
    if not text:
        return Object(kind="unknown", name=text)
    if any(mark in text for mark in "*?") or "{" in text and "}" in text:
        return Object(kind="glob", name=text)
    resolved = resolve(text, cwd, root)
    relative = None
    if root is not None and posixpath.isabs(resolved) and resolved.startswith(root.rstrip("/") + "/"):
        relative = resolved[len(root.rstrip("/")) + 1:]
    elif root is not None and resolved == root.rstrip("/"):
        relative = "."
    kind: Kind = "directory" if text.endswith("/") or text in (".", "..") or resolved.rsplit("/", 1)[-1] == "." \
        else "file"
    return Object(kind=kind, name=resolved, role=role_of(relative or resolved), relative=relative)


def role_of(path: str) -> Role:
    """What a file is for, from the segments and the name of its path; the earliest rule that matches decides."""
    segments = [segment for segment in path.split("/") if segment]
    name = segments[-1] if segments else path
    parents = set(segments[:-1])
    lower = name.lower()
    if name in HARNESS_NAMES or parents & HARNESS_SEGMENTS:
        return "harness"
    if parents & GENERATED_SEGMENTS:
        return "generated"
    if name in LOCK_NAMES or lower.endswith(".lock"):
        return "lock"
    if name.endswith(TEST_SUFFIXES) or lower.startswith("test_") or name == "conftest.py" or parents & TEST_SEGMENTS:
        return "test"
    if lower.endswith(".ipynb"):
        return "notebook"
    if name in CONFIG_NAMES or lower.endswith(CONFIG_SUFFIXES) or (lower.startswith(".") and lower.endswith("rc")) \
            or ".config." in lower or parents & CONFIG_SEGMENTS:
        return "config"
    if lower.endswith(DOCS_SUFFIXES) or parents & DOCS_SEGMENTS:
        return "docs"
    if lower.endswith(SCRIPT_SUFFIXES):
        return "script"
    if lower.endswith(DATA_SUFFIXES):
        return "data"
    if lower.endswith(ASSET_SUFFIXES):
        return "asset"
    if lower.endswith(SOURCE_SUFFIXES) or lower.endswith(".json"):
        return "source"
    return "other"


def project_root(paths: list[str]) -> str | None:
    """The directory the absolute paths of a trajectory share, leaving out system and cache directories.

    The root is the longest common directory of the paths; with a single path it is that path's directory.
    """
    kept = [posixpath.normpath(p) for p in paths if posixpath.isabs(p) and not p.startswith(SYSTEM_PREFIXES)
            and "/.cache/" not in p]
    if not kept:
        return None
    parts = [p.split("/") for p in kept]
    common = parts[0][:-1]
    for other in parts[1:]:
        length = 0
        while length < min(len(common), len(other) - 1) and common[length] == other[length]:
            length += 1
        common = common[:length]
    # A root at the file system's top or at a home directory tells nothing about the project.
    if len(common) <= 1 or (len(common) == 3 and common[1] in ("Users", "home")):
        return None
    return "/".join(common) or "/"
