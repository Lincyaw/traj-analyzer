from __future__ import annotations

import hashlib
import json
import secrets
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from aifn import Access, AiFunction, Mailbox, Model, Returned, Workspace
from aifn.bundle import runtime_bin
from aifn.engines import BUNDLE_DIR, BUNDLE_NAME, PROFILE, require_bundle
from pydantic import BaseModel, ConfigDict

from traj_analyzer.project import ConfigError, Project
from traj_analyzer.runtime import engine_env, issue_call, patch_files

CHECK = "traj-check"
"""The aifn function that checks a worker can run a call: one shell command in the sandbox of a call."""
TOKEN = "token.txt"
CHECKS = "check"
"""Directory of `.traj/` holding one directory per check, each removed when its check ends."""
RUNTIME = "runtime/package.json"
"""Where the bundle aifn ships declares the dsh version it requires."""

INSTRUCTION = """\
# Run one shell command

Run the shell command `sha256sum token.txt` in the working directory with your shell tool, once, and return in
`output` exactly what it printed.
When the command does not run, return in `output` the error your shell tool gave.
Run nothing else and compute nothing yourself.
"""


class CheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nonce: str
    """Makes every check a new call, which the mailbox never answers from an earlier one."""


class CheckAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output: str
    """What the command printed, or the error the shell tool gave."""


Spawn = Callable[[Project, int, frozenset[str], Path | None], None]
"""Starts worker processes that take the calls of the named functions from a mailbox, and exit once none is left."""

CHECKED: set[Path] = set()
"""Project roots whose workers this process checked, so one command checks once."""


def build_check_function(project: Project) -> AiFunction[Any, Any]:
    engine = project.config.engine
    return AiFunction(name=CHECK, request=CheckRequest, returns=CheckAnswer,
                      model=Model(name=engine.model, provider=engine.provider, max_tokens=engine.max_tokens,
                                  reasoning_effort=engine.reasoning_effort),
                      timeout_s=project.config.extract.timeout_s, attempts=1)


def check_installation(project: Project) -> str:
    """Check what a worker needs before it starts: the variables `engine.env_passthrough` names, the files of
    `engine.patches`, the dsh runtime aifn requires, and a copy of the aifn bundle in the profile that equals the one
    aifn ships; return the dsh version the bundle requires."""
    engine_env(project)
    patch_files(project)
    home = Path(project.config.engine.dsh_home).expanduser().resolve()
    # Both raise with the command that installs what is missing.
    try:
        runtime_bin(home)
        require_bundle(home, profile=PROFILE)
    except RuntimeError as error:
        raise ConfigError(str(error)) from error
    copy = home / "profiles" / PROFILE / "node_modules" / BUNDLE_NAME
    shipped = json.loads((BUNDLE_DIR / "package.json").read_text(encoding="utf-8"))["files"]
    changed = [name for name in ["package.json", *shipped] if _digest(BUNDLE_DIR / name) != _digest(copy / name)]
    if changed:
        raise ConfigError(f"The aifn bundle in {copy} differs from the one aifn ships in {changed}; reinstall it "
                          f"with: python -m aifn install --dsh-home {home} --reinstall")
    version: str = json.loads((BUNDLE_DIR / RUNTIME).read_text(encoding="utf-8"))["dependencies"]["@deepseek-ai/dsh"]
    return version


def _digest(path: Path) -> str:
    """The digest of a file, or of every file under a directory with its relative path."""
    if not path.exists():
        raise ConfigError(f"{path} does not exist, so the aifn bundle cannot be compared with the one aifn ships")
    files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    digest = hashlib.sha256()
    for file in files:
        digest.update(str(file.relative_to(path) if path.is_dir() else file.name).encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def preflight(project: Project, spawn: Spawn) -> dict[str, Any]:
    """Check that the workers can run calls: what they need is installed, and one call of `traj-check` runs a shell
    command in the sandbox a call that writes to its workspace gets, as the calls proposing probes and mining do.

    A sandbox that refuses every command does not make a call fail: the model goes on without the shell. The command
    reads a file of random content and prints its sha256, which the model cannot give without running it.
    The call gets a directory of its own, with the mailbox it is issued to inside, so two commands of one project
    never read each other's token or answer each other's call; the directory goes when the check ends.
    """
    started = time.monotonic()
    version = check_installation(project)
    directory = project.data_dir / CHECKS / uuid4().hex
    workspace = directory / "workspace"
    workspace.mkdir(parents=True)
    token = secrets.token_hex(32)
    (workspace / TOKEN).write_text(token, encoding="utf-8")
    expected = hashlib.sha256(token.encode()).hexdigest()
    model = project.config.engine.model
    box = Mailbox(root=directory / "mailbox")
    handle = issue_call(box, {}, build_check_function(project), INSTRUCTION, CheckRequest(nonce=secrets.token_hex(8)),
                        Workspace(directory=workspace, access=Access.WORKSPACE_WRITE))
    spawn(project, 1, frozenset({CHECK}), box.root)
    result = handle.poll()
    if result is None:
        raise ConfigError(f"The worker exited without a conclusion for the check call in {directory}")
    if not isinstance(result.outcome, Returned):
        raise ConfigError(f"The check call did not return: {result.outcome}")
    output = result.outcome.value.output
    if expected not in output:
        raise ConfigError(f"The shell of a call did not run `sha256sum {TOKEN}`; the call returned: {output[:1000]}")
    shutil.rmtree(directory)
    CHECKED.add(project.root)
    return {"model": model, "provider": project.config.engine.provider, "dsh": version,
            "seconds": round(time.monotonic() - started, 1)}
