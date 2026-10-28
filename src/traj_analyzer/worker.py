from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from aifn import Call, Mailbox, Registry, Worker
from aifn.engines import DeepSeekEngine

from traj_analyzer.extract import FUNCTIONS_VARIABLE
from traj_analyzer.hunting.functions import functions as hunting_functions
from traj_analyzer.mining import functions as mining_functions
from traj_analyzer.operators.catalog import runtime_groups
from traj_analyzer.preflight import check_function
from traj_analyzer.project import ConfigError, Project
from traj_analyzer.runtime import build_function, engine_env, policy


@dataclass(frozen=True, slots=True)
class WorkerMailbox(Mailbox):
    """The project's mailbox as a worker sees it: only the calls of functions the project still defines.

    A mining round that stops midway leaves calls of its trial functions pending, and those functions stop existing
    when the trial ends.
    """

    functions: frozenset[str] = frozenset()

    def pending(self) -> Iterator[Call]:
        for call in Mailbox.pending(self):
            if call.function in self.functions:
                yield call


def registry(project: Project) -> Registry:
    """Every aifn function a worker of this project may be asked to run: feature extraction, mining, hunts and the
    check that a worker can run a call."""
    functions = dict(mining_functions.functions(project))
    functions.update(hunting_functions(project))
    check = check_function(project)
    functions[check.name] = check
    for group in runtime_groups(project):
        if group.is_code:
            continue
        functions[group.function_name] = build_function(project, group)
        if project.config.evaluation.model is not None:
            second = build_function(project, group, second=True)
            functions[second.name] = second
    return Registry(functions=functions)


def make_worker(worker_id: str) -> Worker:
    """Factory for `python -m aifn worker traj_analyzer.worker:make_worker`.

    `TRAJ_FUNCTIONS`, when set by the run that starts the worker, names the functions it takes calls of.
    """
    project = Project.find()
    config = project.config.engine
    patches = tuple((project.root / Path(p).expanduser()).resolve() for p in config.patches)
    missing = [str(p) for p in patches if not p.is_file()]
    if missing:
        raise ConfigError(f"engine.patches files do not exist: {missing}")
    functions = registry(project)
    taken = frozenset(functions.functions)
    if os.environ.get(FUNCTIONS_VARIABLE):
        taken &= frozenset(os.environ[FUNCTIONS_VARIABLE].split(","))
    return Worker(
        mailbox=WorkerMailbox(root=project.mailbox_dir, functions=taken),
        registry=functions,
        engine=DeepSeekEngine(dsh_home=Path(config.dsh_home).expanduser(), env=engine_env(project),
                              patches=patches),
        policy=policy(project),
        id=worker_id,
    )
