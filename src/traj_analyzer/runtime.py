from __future__ import annotations

import os
from pathlib import Path
from typing import Any
 
from aifn import AiFunction, Mailbox, Model, Policy, Registry, Worker
from aifn.engines import DeepSeekEngine

from traj_analyzer import proposals
from traj_analyzer.features.spec import ExtractRequest, output_model, write_instruction
from traj_analyzer.operators.catalog import Group, runtime_groups
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


def registry(project: Project) -> Registry:
    functions = dict(proposals.functions(project))
    for group in runtime_groups(project):
        if group.is_code:
            continue
        functions[group.function_name] = build_function(project, group)
        if project.config.evaluation.model is not None:
            second = build_function(project, group, second=True)
            functions[second.name] = second
    return Registry(functions=functions)


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
