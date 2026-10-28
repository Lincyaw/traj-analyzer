from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from traj_analyzer.hunting.draw import Draw
from traj_analyzer.hunting.driver import (
    FINAL,
    PENDING,
    REPORT,
    WORKSPACE,
    Answers,
    Hunt,
    Reads,
    begin_round,
    closing_draws,
    end_round,
    finish,
    last_run,
    note,
    open_hunt,
    over,
    propose_request,
    record,
    resume,
    save_state,
    start,
    write_reads,
    write_state,
)
from traj_analyzer.hunting.functions import (
    NEW,
    Case,
    Verification,
    answer_section,
    proposal_model,
    propose_instruction,
    verify_instruction,
)
from traj_analyzer.hunting.ledger import HuntState
from traj_analyzer.hunting.spec import load_hunt
from traj_analyzer.hunting.workspace import prepare, write_file
from traj_analyzer.project import ConfigError, Project


class Batch(BaseModel):
    """One group of cases one reader of an agent session reads: its cases, the instruction it follows, the file
    holding its cases, and the file it writes its verdicts to; paths are relative to the workspace. A blind batch
    reads trajectories a probe hit without their hits."""

    index: int
    cases: list[Case]
    instruction: str
    request: str
    answer: str
    blind: bool = False


class Pending(BaseModel):
    """A round of an agent session that waits for its verdicts, with what the round found before the reads."""

    round: int
    final: bool
    """Whether the round is the closing check."""
    state: HuntState
    reads: Reads
    failed: dict[str, str]
    retired: list[str]
    batches: list[Batch]


def _base(number: int, final: bool) -> str:
    return f"rounds/{FINAL}" if final else f"rounds/{number}"


def _parse[M: BaseModel](model: type[M], path: Path) -> M:
    """The file as the model, with the file named in the error of one that does not fit."""
    try:
        return model.model_validate_json(path.read_bytes())
    except ValidationError as error:
        raise ConfigError(f"{path} does not fit its schema: {error}") from error


def start_hunt(project: Project, name: str) -> dict[str, Any]:
    """Start a run of the hunt for an agent session, or continue the one in progress: prepare the workspace and
    write what the next step reads."""
    hunt = open_hunt(project, name, create=True)
    prepare(project, hunt.workspace, name, hunt.spec, hunt.keys)
    if not (hunt.directory / PENDING).is_file():
        state = resume(hunt.directory) or start(project, name)
        if not over(hunt.spec, state):
            open_proposal(hunt, state)
    return status(project, name)


def open_proposal(hunt: Hunt, state: HuntState) -> None:
    """Write the state the proposal of the next round reads, its request, the schema of its answer, and the
    instruction an agent session follows for it: the instruction of `hunt-propose` with where to write the answer."""
    number = state.round + 1
    base = _base(number, False)
    write_state(hunt, state)
    write_file(hunt.workspace, f"{base}/input.json", propose_request(hunt, number).model_dump_json(indent=1))
    schema = json.dumps(proposal_model(hunt.spec.probes_max).model_json_schema(), indent=1)
    write_file(hunt.workspace, f"{base}/proposal.schema.json", schema)
    write_file(hunt.workspace, f"{base}/propose.md", propose_instruction(hunt.spec, number, f"{base}/input.json")
               + answer_section(f"{base}/proposal.json", f"{base}/proposal.schema.json"))
    note(hunt, number, "propose", "open")


def submit_proposal(project: Project, name: str) -> dict[str, Any]:
    """Take the proposal the agent session wrote for the next round, run its probes, draw the reads of the round
    and write their cases, in the groups the readers read."""
    hunt = open_hunt(project, name, create=False)
    _require_no_pending(hunt, name)
    state = (resume(hunt.directory) or start(project, name)).model_copy(deep=True)
    if over(hunt.spec, state):
        raise ConfigError(f"The search of {hunt.directory} has ended; close it with `traj hunt close {name}`")
    number = state.round + 1
    path = hunt.workspace / _base(number, False) / "proposal.json"
    if not path.is_file():
        raise ConfigError(f"{path} does not exist; write the proposal there, as {_base(number, False)}/propose.md "
                          f"says")
    proposal = _parse(proposal_model(hunt.spec.probes_max), path)
    note(hunt, number, "propose", "done")
    failed, retired, draws = begin_round(hunt, state, proposal, number)
    open_verdicts(hunt, number, False, state, draws, failed, sorted(retired))
    return status(project, name)


def open_verdicts(
    hunt: Hunt, number: int, final: bool, state: HuntState, draws: list[Draw], failed: dict[str, str],
    retired: list[str],
) -> None:
    """Write the cases of a round and, per group of cases, the file holding them and the instruction a reader
    follows: the instruction of `hunt-verify` with where to write the verdicts; keep the round as pending."""
    base = _base(number, final)
    answers = hunt.workspace / base / "answers"
    if answers.is_dir():
        shutil.rmtree(answers)
    note(hunt, number, "close" if final else "read", "open")
    reads = write_reads(hunt, draws, base, number)
    schema = f"{base}/verify.schema.json"
    write_file(hunt.workspace, schema, json.dumps(Verification.model_json_schema(), indent=1))
    batches = []
    chunks = [*((chunk, False) for chunk in reads.groups), *((chunk, True) for chunk in reads.blind_groups)]
    for index, (chunk, blind) in enumerate(chunks):
        request, answer = f"{base}/batches/{index}.json", f"{base}/answers/{index}.json"
        write_file(hunt.workspace, request, json.dumps(
            {"hunt": hunt.name, "round": number,
             "cases": [case.model_dump(mode="json", exclude_defaults=True) for case in chunk]},
            ensure_ascii=False, indent=1))
        instruction = f"{base}/batches/{index}.md"
        write_file(hunt.workspace, instruction, verify_instruction(request) + answer_section(answer, schema))
        batches.append(Batch(index=index, cases=chunk, instruction=instruction, request=request, answer=answer,
                             blind=blind))
    pending = Pending(round=number, final=final, state=state, reads=reads, failed=failed, retired=retired,
                      batches=batches)
    (hunt.directory / PENDING).write_text(pending.model_dump_json(indent=1), encoding="utf-8")


def submit_verdicts(project: Project, name: str) -> dict[str, Any]:
    """Take the verdicts the readers wrote for every group of the pending round and record the round; after the
    closing check, write the verdicts table, keep the probes, mark the trajectories and write the report."""
    hunt = open_hunt(project, name, create=False)
    path = hunt.directory / PENDING
    if not path.is_file():
        raise ConfigError(f"No round of {hunt.directory} waits for verdicts; see `traj hunt status {name}`")
    pending = Pending.model_validate_json(path.read_bytes())
    missing = [batch.answer for batch in pending.batches if not (hunt.workspace / batch.answer).is_file()]
    if missing:
        raise ConfigError(f"The verdicts {missing} are not written yet; each reader writes its own, as its "
                          f"instruction says")
    facets = {facet.name for facet in pending.state.facets}
    answers: Answers = []
    blind: Answers = []
    for batch in pending.batches:
        verdicts = {verdict.file: verdict for verdict in _parse(Verification, hunt.workspace / batch.answer).verdicts}
        files = {case.file for case in batch.cases}
        if set(verdicts) != files:
            raise ConfigError(f"{batch.answer} gives verdicts for {sorted(set(verdicts) - files)} its group does "
                              f"not hold, and none for {sorted(files - set(verdicts))}; it gives one per case")
        unknown = sorted({verdict.facet for verdict in verdicts.values() if verdict.found and verdict.facet != NEW
                          and verdict.facet not in facets})
        if unknown:
            raise ConfigError(f"{batch.answer} names facets facets.json lacks: {unknown}; a new kind of behaviour "
                              f"is `{NEW}` with `new_facet`")
        (blind if batch.blind else answers).append((batch.cases, verdicts, "the answer gave no verdict for it"))
    readings, blind_readings, unanswered = record(pending.state, pending.reads, answers, blind, pending.round)
    note(hunt, pending.round, "close" if pending.final else "read", "done")
    if pending.final:
        finish(hunt, pending.state, readings, blind_readings, unanswered)
        path.unlink()
        return status(project, name)
    state = end_round(hunt, pending.state, pending.round, readings, blind_readings, unanswered, pending.failed,
                      pending.retired)
    save_state(hunt, state)
    path.unlink()
    if not over(hunt.spec, state):
        open_proposal(hunt, state)
    return status(project, name)


def close_hunt(project: Project, name: str) -> dict[str, Any]:
    """Close a run whose search has ended: draw the closing check, or finish the run when it reads nothing."""
    hunt = open_hunt(project, name, create=False)
    _require_no_pending(hunt, name)
    state = resume(hunt.directory) or start(project, name)
    if not over(hunt.spec, state):
        raise ConfigError(f"The search of {hunt.directory} goes on until {hunt.spec.patience} rounds in a row find "
                          f"nothing new or {hunt.spec.max_rounds} rounds have run; see `traj hunt status {name}`")
    draws = closing_draws(hunt, state)
    if not draws:
        finish(hunt, state, [], [], [])
        return status(project, name)
    open_verdicts(hunt, state.round + 1, True, state.model_copy(deep=True), draws, {}, [])
    return status(project, name)


def _require_no_pending(hunt: Hunt, name: str) -> None:
    if (hunt.directory / PENDING).is_file():
        raise ConfigError(f"A round of {hunt.directory} waits for its verdicts; hand them in with "
                          f"`traj hunt verdicts {name}`")


def status(project: Project, name: str) -> dict[str, Any]:
    """Where the run of the hunt stands, the files the next step reads and writes, and the command that takes it."""
    spec = load_hunt(project, name)
    directory = last_run(project, name)
    if directory is None:
        return {"hunt": name, "phase": "none", "next": f"traj hunt start {name}"}
    workspace = directory / WORKSPACE
    out: dict[str, Any] = {"hunt": name, "run": int(directory.name), "workspace": str(workspace)}
    if (directory / REPORT).is_file():
        return {**out, "phase": "finished", "report": json.loads((directory / REPORT).read_text(encoding="utf-8")),
                "next": f"`traj hunt start {name}` starts a new run from the probes this one kept"}
    state = resume(directory) or start(project, name)
    out.update(rounds=state.round, quiet=state.quiet, patience=spec.patience, max_rounds=spec.max_rounds,
               found=sum(reading.found for reading in state.readings.values()),
               last_round=state.rounds[-1].model_dump(mode="json") if state.rounds else None)
    pending = directory / PENDING
    if pending.is_file():
        waiting = Pending.model_validate_json(pending.read_bytes())
        batches = [{"index": batch.index, "cases": len(batch.cases), "instruction": batch.instruction,
                    "request": batch.request, "answer": batch.answer,
                    "answered": (workspace / batch.answer).is_file()} for batch in waiting.batches]
        return {**out, "phase": "verdicts", "round": waiting.round, "final": waiting.final, "batches": batches,
                "next": f"every reader writes its answer, then `traj hunt verdicts {name}`"}
    if over(spec, state):
        return {**out, "phase": "close", "next": f"traj hunt close {name}"}
    base = _base(state.round + 1, False)
    proposal = {"instruction": f"{base}/propose.md", "answer": f"{base}/proposal.json",
                "answered": (workspace / base / "proposal.json").is_file()}
    if not (workspace / proposal["instruction"]).is_file():
        return {**out, "phase": "start", "next": f"traj hunt start {name}"}
    return {**out, "phase": "propose", "round": state.round + 1, "proposal": proposal,
            "next": f"write the proposal, then `traj hunt propose {name}`"}
