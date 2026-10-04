from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow as pa
from aifn import Handle, Mailbox, Returned, Workspace
from pydantic import BaseModel, Field
from sklearn.neighbors import NearestNeighbors

from traj_analyzer import proposals
from traj_analyzer import schema as schema_module
from traj_analyzer.evaluation import (
    Evaluation,
    canonical,
    disagreement,
    evaluate,
    labels_of,
    noise,
    ok_values,
    reference_keys,
)
from traj_analyzer.extract import SubprocessWorkers, WorkerRunner, extract, issue_call, known_calls, run_groups
from traj_analyzer.features.spec import value_validator
from traj_analyzer.features.table import KEY, read_group
from traj_analyzer.files import load_module
from traj_analyzer.ingest import IndexRow, load_index, load_trajectory
from traj_analyzer.operators import base as operator_base
from traj_analyzer.operators.base import FeatureSpec, Operator
from traj_analyzer.operators.catalog import LIBRARY, Group, edit_operators, load_groups, trial_group
from traj_analyzer.project import ConfigError, Project
from traj_analyzer.proposals import (
    CONSOLIDATE,
    IMPLEMENT,
    IMPLEMENT_INSTRUCTION,
    PROPOSE,
    REVISE,
    REVISE_INSTRUCTION,
    Case,
    ConsolidateRequest,
    Definition,
    ImplementRequest,
    Proposal,
    ProposeRequest,
    ReviseRequest,
    TrajectoryFile,
    consolidate_instruction,
    mining_model,
    propose_instruction,
)
from traj_analyzer.runtime import mailbox
from traj_analyzer.vectorize import build_frame

MINED = "mined"
"""Instance and call group of the mined LLM features in traj.yaml, and the operator namespace of mined code."""
REPORT = "report.json"
MAX_LABEL_COUNTS = 300
"""Most labels shown to mine-revise when a feature has too many."""


class Verdict(BaseModel):
    feature: str
    reads: Literal["fields", "text"]
    parent: str | None = None
    """The feature this one revises."""
    noise: float | None = None
    compared: int = 0
    distinct: int = 0
    decision: Literal["add", "reject", "remove"]
    reason: str = ""


class RoundReport(BaseModel):
    round: int
    singles: list[str] = Field(default_factory=list)
    pairs: list[tuple[str, str]] = Field(default_factory=list)
    failed_readings: list[str] = Field(default_factory=list)
    proposals: int = 0
    verdicts: list[Verdict] = Field(default_factory=list)
    added: list[str] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)


@dataclass
class _Item:
    """One definition on trial: a candidate, or the revision of a candidate or of an enabled mined feature."""

    definition: Definition
    examples: list[proposals.Example] = field(default_factory=list)
    parent: str | None = None
    replaces: str | None = None
    """The enabled mined feature this definition takes the place of."""
    noisy: bool = False
    """An enabled mined feature above the noise limit, which is removed when no revision of it is accepted."""
    revised: bool = False
    source: Path | None = None
    """The operator file of a code candidate."""
    noise: float | None = None
    compared: int = 0
    distinct: int = 0
    decision: Literal["add", "reject", "revise"] | None = None
    reason: str = ""
    revise_request: ReviseRequest | None = None

    @property
    def name(self) -> str:
        return self.definition.spec.name

    def settle(self, decision: Literal["add", "reject", "revise"], reason: str = "") -> None:
        self.decision, self.reason = decision, reason

    def verdict(self) -> Verdict:
        assert self.decision in ("add", "reject")
        return Verdict(feature=self.name, reads=self.definition.reads, parent=self.parent, noise=self.noise,
                       compared=self.compared, distinct=self.distinct, decision=self.decision, reason=self.reason)


@dataclass
class _Session:
    """What every step of one round shares."""

    project: Project
    runner: WorkerRunner
    workers: int | None
    box: Mailbox
    known: dict[str, str]
    directory: Path
    rows: dict[str, IndexRow]
    reference: list[str]
    chosen: list[str]

    def settle(self, handles: list[Handle[Any]]) -> list[Any]:
        """Run workers until every call has a conclusion, and return the conclusions in order."""
        if any(handle.poll() is None for handle in handles):
            self.runner.run(self.project, self.workers or self.project.config.extract.workers)
        results = [handle.poll() for handle in handles]
        if any(result is None for result in results):
            raise RuntimeError("aifn workers exited with calls left without a conclusion; a call claimed by a "
                               "worker that was stopped is free again when its lease expires, so run again")
        return results

    def issue(self, name: str, instruction: str, request: BaseModel, workspace: Path) -> Handle[Any]:
        path = self.project.instructions_dir / f"{name}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(instruction, encoding="utf-8")
        function = proposals.functions(self.project)[name]
        return issue_call(self.box, self.known, function, instruction, request, Workspace(directory=workspace))

    @property
    def datasets_dir(self) -> Path:
        return self.project.data_dir / "datasets"


def mine(
    project: Project, *, rounds: int = 1, workers: int | None = None, runner: WorkerRunner | None = None
) -> list[RoundReport]:
    """Run mining rounds until one adds no feature or `rounds` is reached, then evaluate the resulting features."""
    if project.config.evaluation.model is None:
        raise ConfigError("evaluation.model is not set; mining measures every candidate with two models")
    runner = runner or SubprocessWorkers()
    reports: list[RoundReport] = []
    for _ in range(rounds):
        report = run_round(project, workers=workers, runner=runner)
        reports.append(report)
        project = Project.load(project.root)
        if not report.added:
            break
    if reports[-1].added or reports[-1].removed:
        evaluate(project, workers=workers, runner=runner)
    return reports


def run_round(project: Project, *, workers: int | None, runner: WorkerRunner) -> RoundReport:
    evaluation = evaluate(project, workers=workers, runner=runner)
    number, directory = _round_directory(project)
    groups = load_groups(project)
    rows = load_index(project)
    session = _Session(project=project, runner=runner, workers=workers, box=mailbox(project),
                       known=known_calls(mailbox(project)), directory=directory,
                       rows={row.key: row for row in rows}, reference=reference_keys(project, rows),
                       chosen=evaluation.evaluation_set)
    report = RoundReport(round=number)
    report.singles, report.pairs = choose_material(project, groups, rows, evaluation, session.reference, number)
    found = _propose(session, groups, report)
    report.proposals = len(found)
    items = _consolidate(session, groups, found, report) if found else []
    _implement(session, [item for item in items if item.definition.reads == "fields"])
    items += _flagged(session, groups, evaluation)
    _trial(session, [item for item in items if item.decision is None], "trial-1")
    taken = {f.name for g in groups for f in g.features} | {item.name for item in items}
    revised = _revise(session, [item for item in items if item.decision == "revise"], taken)
    _trial(session, [item for item in revised if item.decision is None], "trial-2")
    for item in items:
        if item.decision == "revise":
            # Revised definitions carry the verdicts; the definition they replace is rejected with their reason.
            item.settle("reject", item.reason)
    report.verdicts = [item.verdict() for item in [*items, *revised] if item.replaces is None or item.revised]
    _apply(session, [*items, *revised], evaluation, report)
    (directory / REPORT).write_text(report.model_dump_json(indent=1), encoding="utf-8")
    return report


def _round_directory(project: Project) -> tuple[int, Path]:
    """The directory of the round to run: the last one when it has no report yet, a new one otherwise."""
    numbers = sorted(int(p.name) for p in project.mining_dir.glob("[0-9]*") if p.is_dir() and p.name.isdigit())
    number = numbers[-1] if numbers else 1
    if (project.mining_dir / str(number) / REPORT).is_file():
        number += 1
    directory = project.mining_dir / str(number)
    directory.mkdir(parents=True, exist_ok=True)
    return number, directory


def choose_material(
    project: Project, groups: list[Group], rows: list[IndexRow], evaluation: Evaluation, reference: list[str],
    number: int,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Reference trajectories outside the evaluation set to read: the closest pairs under the distinguishing
    features, and single trajectories at random."""
    config = project.config.mining
    held_out = evaluation.held_out()
    readable = [key for key in reference if key not in held_out]
    wanted = set(readable)
    frame = build_frame(project, groups, [row for row in rows if row.key in wanted],
                        project.config.sampling.vector_length)
    roles = evaluation.roles()
    weights = {feature: 1.0 for feature in frame.columns if roles.get(feature) == "distinguishing"}
    pairs: list[tuple[str, str]] = []
    if weights and config.pairs and len(readable) >= 2:
        matrix = frame.matrix(weights)
        distance, neighbour = NearestNeighbors(n_neighbors=1).fit(matrix.x).kneighbors()
        keys = list(frame.distance.index)
        used: set[int] = set()
        for i in np.argsort(distance[:, 0], kind="stable"):
            j = int(neighbour[i, 0])
            if len(pairs) < config.pairs and i not in used and j not in used:
                used.update((int(i), j))
                pairs.append((keys[int(i)], keys[j]))
    paired = {key for pair in pairs for key in pair}
    pool = sorted(wanted - paired)
    rng = np.random.default_rng([project.config.evaluation.seed, number])
    singles = sorted(rng.choice(pool, size=min(config.singles, len(pool)), replace=False).tolist()) if pool else []
    return singles, pairs


def _propose(session: _Session, groups: list[Group], report: RoundReport) -> list[Proposal]:
    instruction = propose_instruction(groups)
    model = mining_model(session.project)
    readings = [[key] for key in report.singles] + [list(pair) for pair in report.pairs]
    handles = []
    for keys in readings:
        files = [TrajectoryFile(file=f"{key}.md", sha256=session.rows[key].sha256,
                                n_chunks=session.rows[key].n_chunks) for key in keys]
        handles.append(session.issue(PROPOSE, instruction, ProposeRequest(files=files, model=model),
                                     session.datasets_dir))
    found: list[Proposal] = []
    for keys, result in zip(readings, session.settle(handles), strict=True):
        if not isinstance(result.outcome, Returned):
            report.failed_readings.append(" + ".join(keys))
            continue
        for proposal in result.outcome.value.proposals:
            # An example is only usable when it names a file of this reading.
            examples = [example for example in proposal.examples if example.key in keys]
            if examples:
                found.append(proposal.model_copy(update={"examples": examples}))
    return found


def _features_json(groups: list[Group]) -> str:
    return json.dumps([f.model_dump(mode="json", include={"name", "type", "description", "labels"})
                       for g in groups for f in g.features], ensure_ascii=False, indent=1)


def _consolidate(
    session: _Session, groups: list[Group], found: list[Proposal], report: RoundReport
) -> list[_Item]:
    files = {"proposals.json": json.dumps([p.model_dump(mode="json") for p in found], ensure_ascii=False, indent=1),
             "features.json": _features_json(groups)}
    for name, text in files.items():
        (session.directory / name).write_text(text, encoding="utf-8")
    digests = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in files.items()}
    request = ConsolidateRequest(
        proposals_file="proposals.json", proposals_sha256=digests["proposals.json"],
        features_file="features.json", features_sha256=digests["features.json"], model=mining_model(session.project))
    limit = session.project.config.mining.candidates
    handle = session.issue(CONSOLIDATE, consolidate_instruction(limit), request, session.directory)
    (result,) = session.settle([handle])
    taken = {f.name for g in groups for f in g.features}
    read = {example.file for proposal in found for example in proposal.examples}
    items = []
    for candidate in result.unwrap().candidates:
        examples = [example for example in candidate.examples if example.file in read]
        item = _Item(definition=Definition(**candidate.model_dump(exclude={"examples"})), examples=examples)
        if item.name in taken:
            item.settle("reject", "the name belongs to another feature")
        elif not examples:
            item.settle("reject", "no example of the candidate names a trajectory read in this round")
        taken.add(item.name)
        items.append(item)
    return items


def _implement(session: _Session, items: list[_Item]) -> None:
    """Have a code operator written for every candidate that structured fields answer, and check the file."""
    pending = [item for item in items if item.decision is None]
    handles = []
    for item in pending:
        workspace = session.directory / "implement" / item.name
        workspace.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(schema_module.__file__, workspace / "schema.py")
        shutil.copyfile(operator_base.__file__, workspace / "base.py")
        shutil.copyfile(LIBRARY / "stats" / "tool_usage.py", workspace / "example_operator.py")
        names = []
        for example in item.examples:
            name = example.key.replace("/", ".") + ".json"
            shutil.copyfile(session.project.trajectory_path(example.key, ".json"), workspace / name)
            names.append(name)
        candidate = Proposal(**item.definition.model_dump(), examples=item.examples)
        request = ImplementRequest(candidate=candidate, example_files=names, model=mining_model(session.project))
        handles.append(session.issue(IMPLEMENT, IMPLEMENT_INSTRUCTION, request, workspace))
    for item, result in zip(pending, session.settle(handles), strict=True):
        if not isinstance(result.outcome, Returned):
            item.settle("reject", f"mine-implement returned no operator: {result.status}")
            continue
        item.source = session.directory / "implement" / item.name / "operator.py"
        item.source.write_text(result.outcome.value.source, encoding="utf-8")


def _flagged(session: _Session, groups: list[Group], evaluation: Evaluation) -> list[_Item]:
    """Enabled mined LLM features that the evaluation sends to revision, each with the cases that show why."""
    mined = next((group for group in groups if group.name == MINED), None)
    if mined is None:
        return []
    (instance,) = mined.instances
    first = read_group(session.project, mined)
    second = read_group(session.project, mined, session.project.evaluation_dir)
    decisions = {card.feature: card for card in evaluation.features}
    items = []
    for feature in mined.features:
        card = decisions[feature.name]
        if card.decision == "keep":
            continue
        definition = Definition(spec=feature, reads="text", guidance=instance.params.features[feature.name].guidance)
        item = _Item(definition=definition, replaces=feature.name, noise=card.noise, compared=card.compared,
                     distinct=card.distinct)
        if card.decision == "revise":
            item.noisy = True
            item.revise_request = _noise_request(session, definition, first, second, evaluation.measured[MINED])
            item.settle("revise", f"noise {card.noise} is above the limit")
        else:
            item.revise_request = _labels_request(session, definition, first)
            item.settle("revise", "more distinct labels than the limit")
        items.append(item)
    return items


def _evidence(table: pa.Table, feature: str) -> dict[str, str | None]:
    column = f"{feature}__evidence"
    values = table[column].to_pylist() if column in table.column_names else [None] * len(table)
    return dict(zip(table[KEY].to_pylist(), values, strict=True))


def _noise_request(
    session: _Session, definition: Definition, first: pa.Table, second: pa.Table, keys: list[str]
) -> ReviseRequest:
    """The cases among `keys`, the trajectories both models measured, where the two values differ most."""
    feature = definition.spec
    wanted = set(keys)
    a, b = ok_values(first, feature, wanted), ok_values(second, feature, wanted)
    scores = {key: disagreement(feature, a[key], b[key]) for key in set(a) & set(b)}
    worst = sorted((key for key, score in scores.items() if score > 0), key=lambda key: (-scores[key], key))
    evidence = (_evidence(first, feature.name), _evidence(second, feature.name))
    cases = [Case(file=f"{key}.md", first=a[key], first_evidence=evidence[0].get(key), second=b[key],
                  second_evidence=evidence[1].get(key))
             for key in worst[:session.project.config.mining.revise_cases]]
    return ReviseRequest(definition=definition, reason="noise", cases=cases, model=mining_model(session.project))


def _labels_request(session: _Session, definition: Definition, first: pa.Table) -> ReviseRequest:
    counts: dict[str, int] = {}
    for value in ok_values(first, definition.spec).values():
        for label in labels_of(definition.spec, [value]):
            counts[label] = counts.get(label, 0) + 1
    top = dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:MAX_LABEL_COUNTS])
    return ReviseRequest(definition=definition, reason="labels", label_counts=top,
                         model=mining_model(session.project))


def _trial(session: _Session, items: list[_Item], name: str) -> None:
    """Extract the definitions under trial and settle each one as add, reject or revise."""
    config = session.project.config
    code = [item for item in items if item.definition.reads == "fields"]
    text = [item for item in items if item.definition.reads == "text"]
    for item in code:
        _trial_code(session, item)
    if not text:
        return
    directory = session.directory / name
    parameters = {"features": {item.name: {
        "type": item.definition.spec.type, "description": item.definition.spec.description,
        "labels": item.definition.spec.labels, "guidance": item.definition.guidance} for item in text}}
    session.project.trial_path.write_text(json.dumps(parameters, ensure_ascii=False, indent=1), encoding="utf-8")
    try:
        group = trial_group(session.project)
        assert group is not None
        example_keys = {example.key for item in text for example in item.examples}
        once = [session.rows[key] for key in sorted(set(session.chosen) | example_keys)]
        twice = [session.rows[key] for key in session.chosen]
        for rows, second in ((once, False), (twice, True)):
            run_groups(session.project, [group], rows, workers=session.workers, runner=session.runner,
                       second=second, directory=directory / ("second" if second else "first"))
        first = read_group(session.project, group, directory / "first")
        second_table = read_group(session.project, group, directory / "second")
    finally:
        session.project.trial_path.unlink()
    noises = noise(group, first, second_table, session.chosen)
    for item in text:
        feature = item.definition.spec
        values = ok_values(first, feature)
        item.noise, item.compared = noises[item.name]
        expected = _expected(item)
        seen = {canonical(feature, value) for value in values.values()} | set(expected.values())
        item.distinct = len(seen)
        revisable = not item.revised
        wrong = [example for example in item.examples
                 if example.key not in values or canonical(feature, values[example.key]) != expected[example.key]]
        labels = labels_of(feature, list(values.values())) if feature.type in ("category", "set") else set()
        if not values or item.noise is None:
            item.settle("reject", "the trial extraction gave no value measured by both models")
        elif item.noise > config.evaluation.noise_max:
            reason = f"noise {round(item.noise, 4)} is above the limit"
            item.revise_request = _noise_request(session, item.definition, first, second_table, session.chosen)
            item.settle("revise" if revisable else "reject", reason)
        elif wrong:
            reason = f"the value of {len(wrong)} of {len(item.examples)} examples is not reproduced"
            evidence = _evidence(first, item.name)
            cases = [Case(file=example.file, first=values.get(example.key), first_evidence=evidence.get(example.key),
                          expected=example.value, steps=example.steps)
                     for example in wrong[:config.mining.revise_cases]]
            item.revise_request = ReviseRequest(definition=item.definition, reason="example", cases=cases,
                                                model=mining_model(session.project))
            item.settle("revise" if revisable else "reject", reason)
        elif item.distinct < 2:
            item.settle("reject", "it takes one value on every trajectory it was extracted from")
        elif not feature.labels and len(labels) > config.evaluation.label_limit:
            item.revise_request = _labels_request(session, item.definition, first)
            item.settle("revise" if revisable else "reject", "more distinct labels than the limit")
        else:
            item.settle("add")


def _expected(item: _Item) -> dict[str, str]:
    """The canonical value each example states, by trajectory key."""
    check = value_validator(item.definition.spec)
    return {example.key: canonical(item.definition.spec, check(example.value)) for example in item.examples}


def load_code_operator(path: Path, spec: FeatureSpec) -> Operator:
    """Load an operator file written for `spec` and require a code operator with exactly that output."""
    operator = getattr(load_module(path), "OPERATOR", None)
    if not isinstance(operator, Operator) or operator.kind != "code":
        raise ValueError("the file does not define OPERATOR as a code operator")
    outputs = operator.outputs(operator.params())
    if [f.model_dump() for f in outputs] != [spec.model_dump()]:
        raise ValueError(f"outputs differ from the candidate's spec: {[f.model_dump() for f in outputs]}")
    return operator


def _trial_code(session: _Session, item: _Item) -> None:
    feature = item.definition.spec
    assert item.source is not None
    check = value_validator(feature)
    values: dict[str, Any] = {}
    # A file written by a model can fail in any way; every failure rejects the candidate and is reported.
    try:
        operator = load_code_operator(item.source, feature)
        params = operator.params()
        for key in sorted(set(session.reference) | {example.key for example in item.examples}):
            trajectory = load_trajectory(session.project, key)
            if operator.requires is None or operator.requires(trajectory):
                value = check(operator.compute(trajectory, params)[feature.name])
                if value is not None:
                    values[key] = value
    except Exception as error:
        item.settle("reject", f"the operator file failed: {type(error).__name__}: {error}")
        return
    item.noise = 0.0
    expected = _expected(item)
    wrong = [example for example in item.examples
             if example.key not in values or canonical(feature, values[example.key]) != expected[example.key]]
    item.distinct = len({canonical(feature, value) for value in values.values()})
    if wrong:
        item.settle("reject", f"the value of {len(wrong)} of {len(item.examples)} examples is not reproduced")
    elif item.distinct < 2:
        item.settle("reject", "it takes one value on every reference trajectory")
    else:
        item.settle("add")


def _revise(session: _Session, items: list[_Item], taken: set[str]) -> list[_Item]:
    """Ask for a revision of every definition sent to revision, and return the revised definitions.

    `taken` holds the names in use; a revision may keep the name of the definition it revises.
    """
    handles = []
    for item in items:
        assert item.revise_request is not None
        handles.append(session.issue(REVISE, REVISE_INSTRUCTION, item.revise_request, session.datasets_dir))
    revised: list[_Item] = []
    taken = set(taken)
    for item, result in zip(items, session.settle(handles), strict=True):
        if not isinstance(result.outcome, Returned):
            item.reason += f"; mine-revise returned no revision: {result.status}"
            continue
        revision = result.outcome.value
        if revision.action == "drop":
            item.reason += f"; dropped in revision: {revision.reason}"
            continue
        for definition in revision.definitions:
            # A rewrite keeps the examples; labels change the values, and a split asks other questions.
            examples = item.examples if revision.action == "rewrite" else []
            child = _Item(definition=definition, examples=examples, parent=item.name, replaces=item.replaces,
                          revised=True)
            if child.name in taken and child.name != item.name:
                child.settle("reject", "the name belongs to another feature")
            taken.add(child.name)
            revised.append(child)
    return revised


def _apply(session: _Session, items: list[_Item], evaluation: Evaluation, report: RoundReport) -> None:
    """Write the accepted definitions to operators/mined/ and traj.yaml, remove what failed, and extract."""
    project = session.project
    added = [item for item in items if item.decision == "add"]
    # An enabled feature leaves when a revision of it is accepted, or when it is noisy and none is.
    replaced = {item.replaces for item in added if item.replaces is not None}
    replaced |= {item.name for item in items if item.noisy}
    text = {item.name: item for item in added if item.definition.reads == "text"}
    code = [item for item in added if item.definition.reads == "fields"]
    noises = {card.feature: card.noise or 0.0 for card in evaluation.features if card.group == MINED}
    noises.update({name: item.noise or 0.0 for name, item in text.items()})
    kept = (set(noises) - replaced) | set(text)
    capacity = project.config.mining.capacity
    dropped = set(sorted(kept, key=lambda name: (-noises[name], name))[:max(0, len(kept) - capacity)])
    for name in sorted(dropped & set(text)):
        report.verdicts.append(Verdict(feature=name, reads="text", noise=noises[name], decision="remove",
                                       reason="the mined LLM features exceed mining.capacity"))
    report.added = sorted([*(set(text) - dropped), *(item.name for item in code)])
    enabled = {card.feature for card in evaluation.features}
    report.removed = sorted(((replaced | dropped) & enabled) - (set(text) - dropped))
    if not report.added and not report.removed:
        return

    def change(operators: list[Any]) -> None:
        entry = next((op for op in operators if op.get("as") == MINED), None)
        if entry is None:
            entry = {"use": "llm.features", "as": MINED, "call": MINED, "params": {"features": {}}}
            operators.append(entry)
        features = entry["params"]["features"]
        for name in (replaced | dropped) & set(features):
            del features[name]
        for name, item in text.items():
            if name in dropped:
                continue
            spec = item.definition.spec
            features[name] = {"type": spec.type, "description": spec.description}
            if spec.labels:
                features[name]["labels"] = dict(spec.labels)
            if item.definition.guidance:
                features[name]["guidance"] = item.definition.guidance
        if not features:
            operators.remove(entry)
        operators.extend({"use": f"{MINED}.{item.name}"} for item in code)

    target = project.operators_dir / MINED
    for item in code:
        assert item.source is not None
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(item.source, target / f"{item.name}.py")
    edited = edit_operators(project, change)
    # Removing a feature needs no extraction: its column stays in the table and is no longer read.
    changed = [f"{MINED}-{item.name}" for item in code]
    if set(text) - dropped:
        changed.append(MINED)
    if changed:
        extract(edited, groups=changed, workers=session.workers, runner=session.runner)
