from __future__ import annotations

import shutil
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aifn import Access, AiFunction, Handle, Mailbox, Workspace
from pydantic import BaseModel

from traj_analyzer.extract import WorkerRunner, issue_call
from traj_analyzer.project import Project


@dataclass(kw_only=True)
class WorkspaceCalls:
    """Calls of aifn functions that work in one workspace directory: issuing them, waiting for them, and writing the
    files of the workspace they read."""

    project: Project
    runner: WorkerRunner
    workers: int | None
    box: Mailbox
    known: dict[str, str]
    functions: dict[str, AiFunction[Any, Any]]
    """The functions these calls belong to; the workers started for them take only their calls."""
    workspace: Path
    placed: set[str] = field(default_factory=set)
    """The files of the workspace the driver wrote, relative to it."""

    def settle(self, handles: list[Handle[Any]]) -> list[Any]:
        """Run workers until every call has a conclusion, and return the conclusions in order."""
        if any(handle.poll() is None for handle in handles):
            self.runner.run(self.project, self.workers or self.project.config.extract.workers,
                            frozenset(self.functions))
        results = [handle.poll() for handle in handles]
        if any(result is None for result in results):
            raise RuntimeError("aifn workers exited with calls left without a conclusion; a call claimed by a "
                               "worker that was stopped is free again when its lease expires, so run again")
        return results

    def issue(self, name: str, instruction: str, request: BaseModel, access: Access) -> Handle[Any]:
        path = self.project.instructions_dir / f"{name}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(instruction, encoding="utf-8")
        return issue_call(self.box, self.known, self.functions[name], instruction, request,
                          Workspace(directory=self.workspace, access=access))

    def alone(self, name: str, instruction: str, request: BaseModel, leftovers: Iterable[Path]) -> Any:
        """Run one call that writes to the workspace, alone, and return what it submitted.

        When the call has no conclusion yet, the `leftovers` go first: what an earlier run of it wrote before it was
        stopped, so the call does not take them for work already done.
        """
        handle = self.issue(name, instruction, request, Access.WORKSPACE_WRITE)
        if handle.poll() is None:
            for path in leftovers:
                path.unlink(missing_ok=True)
        (result,) = self.settle([handle])
        return result.unwrap()

    def unplaced(self, directory: str) -> list[Path]:
        """The Python files of a workspace directory the driver did not write."""
        return [path for path in (self.workspace / directory).glob("*.py")
                if f"{directory}/{path.name}" not in self.placed]

    def hand_off(self, name: str, text: str) -> None:
        """Write a file of the workspace: what the driver prepares, or the accepted result of a function."""
        path = self.workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        self.placed.add(name)

    def place(self, name: str, source: Path) -> None:
        """Copy a file into the workspace."""
        shutil.copyfile(source, self.workspace / name)
        self.placed.add(name)
