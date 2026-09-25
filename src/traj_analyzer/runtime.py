from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from aifn import AiFunction, Mailbox, Model, Policy, Registry, Worker
from aifn.engines import DeepSeekEngine

from traj_analyzer.features.spec import ExtractRequest, output_model, write_instruction
from traj_analyzer.operators.catalog import Group, load_groups
from traj_analyzer.project import ConfigError, Project


def model_name(project: Project, group: Group) -> str:
    return group.model or project.config.engine.model


def build_function(project: Project, group: Group, instruction: Path | None = None) -> AiFunction[Any, Any]:
    """Declare the group's aifn function; the submitting side passes the instruction file it wrote.

    Workers need no instruction file, because each call carries a copy of its instruction text.
    """
    engine = project.config.engine
    extract = project.config.extract
    return AiFunction(
        name=group.function_name,
        request=ExtractRequest,
        returns=output_model(group),
        model=Model(
            name=model_name(project, group), provider=engine.provider,
            max_tokens=engine.max_tokens, reasoning_effort=engine.reasoning_effort,
        ),
        instruction=instruction,
        timeout_s=extract.timeout_s,
        attempts=extract.attempts,
    )


def submitting_function(project: Project, group: Group) -> AiFunction[Any, Any]:
    return build_function(project, group, write_instruction(project, group))


def mailbox(project: Project) -> Mailbox:
    return Mailbox(root=project.mailbox_dir)


def registry(project: Project) -> Registry:
    return Registry(functions={
        group.function_name: build_function(project, group)
        for group in load_groups(project) if not group.is_code
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
    config = project.config.engine
    patches = tuple((project.root / Path(p).expanduser()).resolve() for p in config.patches)
    missing = [str(p) for p in patches if not p.is_file()]
    if missing:
        raise ConfigError(f"engine.patches files do not exist: {missing}")
    return Worker(
        mailbox=mailbox(project),
        registry=registry(project),
        engine=DeepSeekEngine(dsh_home=Path(config.dsh_home).expanduser(), env=engine_env(project),
                              patches=patches),
        policy=policy(project),
        id=worker_id,
    )
