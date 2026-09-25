from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from aifn import AiFunction, Mailbox, Model, Policy, Registry, Worker
from aifn.engines import DeepSeekEngine

from traj_analyzer.features.spec import (
    ExtractRequest,
    GroupSpec,
    load_groups,
    output_model,
    write_instruction,
)
from traj_analyzer.project import ConfigError, Project


def build_function(project: Project, group: GroupSpec) -> AiFunction[Any, Any]:
    engine = project.config.engine
    extract = project.config.extract
    return AiFunction(
        name=group.function_name,
        request=ExtractRequest,
        returns=output_model(group),
        model=Model(
            name=group.model or engine.model, provider=engine.provider,
            max_tokens=engine.max_tokens, reasoning_effort=engine.reasoning_effort,
        ),
        instruction=write_instruction(project, group),
        timeout_s=extract.timeout_s,
        attempts=extract.attempts,
    )


def mailbox(project: Project) -> Mailbox:
    return Mailbox(root=project.mailbox_dir)


def registry(project: Project) -> Registry:
    return Registry(functions={
        group.function_name: build_function(project, group)
        for group in load_groups(project) if not group.is_builtin
    })


def policy(project: Project) -> Policy:
    return Policy(attempts=project.config.extract.attempts)


def engine_env(project: Project) -> dict[str, str]:
    names = project.config.engine.env_passthrough
    missing = [name for name in names if name not in os.environ]
    if missing:
        raise ConfigError(f"Environment variables required by engine.env_passthrough: {missing}")
    return {name: os.environ[name] for name in names}


def make_worker(worker_id: str) -> Worker:
    """Factory for `python -m aifn worker traj_analyzer.runtime:make_worker`."""
    project = Project.find()
    return Worker(
        mailbox=mailbox(project),
        registry=registry(project),
        engine=DeepSeekEngine(dsh_home=Path(project.config.engine.dsh_home).expanduser(),
                              env=engine_env(project)),
        policy=policy(project),
        id=worker_id,
    )
