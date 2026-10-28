from __future__ import annotations

import hashlib
import json
import secrets
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from aifn import Access, AiFunction, Model, Returned, Workspace
from aifn.bundle import BUNDLE_DIR, require_bundle, runtime_bin
from aifn.engines.deepseek import PROFILE
from pydantic import BaseModel, ConfigDict

from traj_analyzer.project import ConfigError, Project
from traj_analyzer.runtime import engine_env, issue_call, mailbox

CHECK = "traj-check"
"""The aifn function that checks a worker can run a call: one shell command in the sandbox of a call."""
TOKEN = "token.txt"
BUNDLE = "node_modules/@aifn/dsh-aifn-harness"
"""Where a dsh profile holds its copy of the aifn bundle."""
DSH = "aifn-runtime/node_modules/@deepseek-ai/dsh/package.json"

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
    model: str


class CheckAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output: str
    """What the command printed, or the error the shell tool gave."""


class PreflightError(RuntimeError):
    """The environment cannot run the calls of aifn functions."""


Spawn = Callable[[Project, int, frozenset[str]], None]
"""Starts worker processes that take the calls of the named functions and exit once none is left."""


def check_function(project: Project) -> AiFunction[Any, Any]:
    engine = project.config.engine
    return AiFunction(name=CHECK, request=CheckRequest, returns=CheckAnswer,
                      model=Model(name=engine.model, provider=engine.provider, max_tokens=engine.max_tokens,
                                  reasoning_effort=engine.reasoning_effort),
                      timeout_s=project.config.extract.timeout_s, attempts=1)


def check_installation(project: Project) -> str:
    """Check what a worker needs before it starts: the variables `engine.env_passthrough` names, the files of
    `engine.patches`, the dsh runtime aifn requires, and a copy of the aifn bundle in the profile that equals the one
    aifn ships; return the version of dsh."""
    engine_env(project)
    config = project.config.engine
    missing = [path for path in config.patches if not (project.root / Path(path).expanduser()).is_file()]
    if missing:
        raise ConfigError(f"engine.patches files do not exist: {missing}")
    home = Path(config.dsh_home).expanduser().resolve()
    runtime_bin(home)
    require_bundle(home, profile=PROFILE)
    copy = home / "profiles" / PROFILE / BUNDLE
    shipped = json.loads((BUNDLE_DIR / "package.json").read_text(encoding="utf-8"))["files"]
    changed = [name for name in ["package.json", *shipped] if _digest(BUNDLE_DIR / name) != _digest(copy / name)]
    if changed:
        raise PreflightError(f"The aifn bundle in {copy} differs from the one aifn ships in {changed}; reinstall it "
                             f"with: python -m aifn install --dsh-home {home} --reinstall")
    version: str = json.loads((home / DSH).read_text(encoding="utf-8"))["version"]
    return version


def _digest(path: Path) -> str:
    """The digest of a file, or of every file under a directory with its relative path."""
    if not path.exists():
        return ""
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
    """
    started = time.monotonic()
    version = check_installation(project)
    directory = project.data_dir / "check"
    directory.mkdir(parents=True, exist_ok=True)
    token = secrets.token_hex(32)
    (directory / TOKEN).write_text(token, encoding="utf-8")
    expected = hashlib.sha256(token.encode()).hexdigest()
    model = project.config.engine.model
    box = mailbox(project)
    handle = issue_call(box, {}, check_function(project), INSTRUCTION, CheckRequest(nonce=uuid4().hex, model=model),
                        Workspace(directory=directory, access=Access.WORKSPACE_WRITE))
    spawn(project, 1, frozenset({CHECK}))
    result = handle.poll()
    if result is None:
        raise PreflightError(f"The worker exited without a conclusion for the check call {handle.id}")
    if not isinstance(result.outcome, Returned):
        raise PreflightError(f"The check call {handle.id} did not return: {result.outcome}")
    output = result.outcome.value.output
    if expected not in output:
        raise PreflightError(f"The shell of a call did not run `sha256sum {TOKEN}`; the call returned: {output[:1000]}")
    return {"model": model, "provider": project.config.engine.provider, "dsh": version, "call": handle.id,
            "seconds": round(time.monotonic() - started, 1)}
