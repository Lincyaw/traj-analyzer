from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from aifn import AiFunction, Call, Handle, Mailbox, Model, Policy, Workspace
from pydantic import BaseModel

from traj_analyzer.features.spec import ExtractRequest, output_model, write_instruction
from traj_analyzer.operators.catalog import Group
from traj_analyzer.project import ConfigError, Project


def model_name(project: Project, group: Group, second: bool = False) -> str:
    """The model that answers the group, or with `second` the model of the second measurement."""
    if not second:
        return group.model or project.config.engine.model
    model = project.config.evaluation.model
    if model is None:
        raise ConfigError("evaluation.model is not set; the second measurement needs a model of its own")
    return model


def build_function(
    project: Project, group: Group, instruction: Path | None = None, second: bool = False
) -> AiFunction[Any, Any]:
    """Declare the group's aifn function; the submitting side passes the instruction file it wrote.

    Workers need no instruction file, because each call carries a copy of its instruction text.
    """
    engine = project.config.engine
    extract = project.config.extract
    return AiFunction(
        name=f"eval-{group.name}" if second else group.function_name,
        request=ExtractRequest,
        returns=output_model(group),
        model=Model(
            name=model_name(project, group, second), provider=engine.provider,
            max_tokens=engine.max_tokens, reasoning_effort=engine.reasoning_effort,
        ),
        instruction=instruction,
        timeout_s=extract.timeout_s,
        attempts=extract.attempts,
    )


def submitting_function(project: Project, group: Group, second: bool = False) -> AiFunction[Any, Any]:
    return build_function(project, group, write_instruction(project, group), second)


def mailbox(project: Project) -> Mailbox:
    return Mailbox(root=project.mailbox_dir)


def policy(project: Project) -> Policy:
    return Policy(attempts=project.config.extract.attempts)


def engine_env(project: Project) -> dict[str, str]:
    names = project.config.engine.env_passthrough
    missing = [name for name in names if name not in os.environ]
    if missing:
        raise ConfigError(f"Environment variables required by engine.env_passthrough: {missing}")
    return {name: os.environ[name] for name in names}


def _identity(call: Call) -> str:
    """What aifn compares to reuse a call: everything except the id and the time it was issued."""
    return json.dumps(call.model_dump(mode="json", exclude={"id", "issued_at"}), sort_keys=True)


def known_calls(box: Mailbox) -> dict[str, str]:
    """Identity of every call in the mailbox, read once; aifn's own lookup rescans the mailbox per submission."""
    requests = box.root / "requests"
    if not requests.is_dir():
        return {}
    known = {}
    for path in requests.glob("*.json"):
        call = Call.model_validate_json(path.read_bytes())
        known[_identity(call)] = call.id
    return known


def issue_call(
    box: Mailbox, known: dict[str, str], function: AiFunction[Any, Any], instruction: str, request: BaseModel,
    workspace: Workspace, dry_run: bool = False,
) -> Handle[Any]:
    """Look up or enqueue one call; a dry run only looks the call up and writes nothing."""
    call = Call(id=uuid4().hex, function=function.name,
                request=request.model_dump(mode="json", by_alias=True, exclude_defaults=True),
                issued_at=datetime.now(UTC), workspace=workspace, instruction=instruction)
    identity = _identity(call)
    if identity in known:
        call = call.model_copy(update={"id": known[identity]})
    elif not dry_run:
        box.issue(call)
        known[identity] = call.id
    return Handle(id=call.id, returns=function.returns, mailbox=box)
