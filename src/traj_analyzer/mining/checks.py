from __future__ import annotations

import json
import shutil
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
from aifn import Access, Returned

from traj_analyzer.anomalies import consequence_coverage, consequence_lifts, read_events, read_turns, spread
from traj_analyzer.evaluation import Evaluation, canonical, disagreement, labels_of, noise, ok_values
from traj_analyzer.events import CONSEQUENCE_RULES
from traj_analyzer.extract import extract, in_scope, run_groups
from traj_analyzer.features.spec import value_validator
from traj_analyzer.features.table import KEY, read_group
from traj_analyzer.layers.tables import KEY as TABLE_KEY
from traj_analyzer.mining.functions import (
    REVISE,
    REVISE_INSTRUCTION,
    TAIL_CHECK,
    TAIL_INSTRUCTION,
    Candidate,
    Case,
    Definition,
    Finding,
    Review,
    ReviseRequest,
    TailCase,
    TailRequest,
    mining_model,
)
from traj_analyzer.mining.operators import compute_operator, load_code_operator
from traj_analyzer.mining.report import RoundReport, Verdict
from traj_analyzer.mining.session import Item, Session
from traj_analyzer.operators.base import FeatureSpec
from traj_analyzer.operators.catalog import Group, edit_operators, trial_group
from traj_analyzer.reviews import precision_by_feature, read_verdicts

MINED = "mined"
"""Instance and call group of the mined judgment features in traj.yaml, and the operator namespace of mined code."""
MAX_LABEL_COUNTS = 300
"""Most labels shown to mine-revise when a feature has too many."""
MIN_SCOPE = 4
"""Fewest trajectories the scope of a judged candidate holds."""
TAIL_TYPES = ("scalar", "boolean", "category", "set", "event")
"""Feature types with a tail the tail check can look at."""


def candidate_items(
    session: Session, groups: list[Group], candidates: list[Candidate], findings: list[Finding],
    reviews: list[Review],
) -> list[Item]:
    """The candidates as items on trial.

    A candidate is rejected when its name is taken, when it has no usable example, when its finding was abandoned
    or does not exist, or when the review did not keep it; an example is usable when it names a trajectory the
    functions may read.
    """
    taken = {f.name for g in groups for f in g.features}
    readable = set(session.readable)
    held = {finding.hypothesis for finding in findings if finding.status != "abandoned"}
    reviewed = {review.candidate: review for review in reviews}
    items = []
    for candidate in candidates:
        examples = [example for example in candidate.examples if example.key in readable]
        definition = Definition(spec=candidate.spec, reads=candidate.reads, guidance=candidate.guidance)
        item = Item(definition=definition, examples=examples, hypothesis=candidate.hypothesis)
        review = reviewed.get(item.name)
        if item.name in taken:
            item.settle("reject", "the name belongs to another feature")
        elif not examples:
            item.settle("reject", "no example of the candidate names a trajectory of the workspace")
        elif candidate.hypothesis not in held:
            item.settle("reject", f"no supported or revised finding is named {candidate.hypothesis}")
        elif review is None:
            item.settle("reject", "the review returned no verdict on the candidate")
        elif review.decision == "reject":
            item.settle("reject", f"review: {review.reason}")
        elif candidate.source is not None:
            item.source = session.workspace / "operators" / f"{item.name}.py"
            item.source.write_text(candidate.source, encoding="utf-8")
        taken.add(item.name)
        items.append(item)
    return items


def flagged_items(session: Session, groups: list[Group], evaluation: Evaluation) -> list[Item]:
    """Enabled mined judgment features that the evaluation sends to revision, each with the cases that show why."""
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
        guidance = instance.params.features[feature.name].guidance  # type: ignore[attr-defined]
        definition = Definition(spec=feature, reads="text", guidance=guidance)
        item = Item(definition=definition, replaces=feature.name, noise=card.noise, compared=card.compared)
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
    session: Session, definition: Definition, first: pa.Table, second: pa.Table, keys: list[str]
) -> ReviseRequest:
    """The cases among `keys`, the trajectories both models measured, where the two values differ most."""
    feature = definition.spec
    wanted = set(keys)
    a, b = ok_values(first, feature, wanted), ok_values(second, feature, wanted)
    scores = {key: disagreement(feature, a[key], b[key]) for key in set(a) & set(b)}
    worst = sorted((key for key, score in scores.items() if score > 0), key=lambda key: (-scores[key], key))
    evidence = (_evidence(first, feature.name), _evidence(second, feature.name))
    cases = [Case(file=session.link_case(key), first=a[key], first_evidence=evidence[0].get(key), second=b[key],
                  second_evidence=evidence[1].get(key))
             for key in worst[:session.project.config.mining.revise_cases]]
    return ReviseRequest(definition=definition, reason="noise", cases=cases, model=mining_model(session.project))


def _labels_request(session: Session, definition: Definition, first: pa.Table) -> ReviseRequest:
    counts: dict[str, int] = {}
    for value in ok_values(first, definition.spec).values():
        for label in labels_of(definition.spec, [value]):
            counts[label] = counts.get(label, 0) + 1
    top = dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:MAX_LABEL_COUNTS])
    return ReviseRequest(definition=definition, reason="labels", label_counts=top,
                         model=mining_model(session.project))


def trial(session: Session, items: list[Item], name: str) -> None:
    """Extract the definitions under trial and settle each one as add, reject or revise."""
    config = session.project.config
    code = [item for item in items if item.definition.reads == "fields"]
    text = [item for item in items if item.definition.reads == "text"]
    for item in code:
        _trial_code(session, item)
    samples = {item.name: _sample(session, item) for item in text}
    text = [item for item in text if item.decision is None]
    if not text:
        return
    directory = session.directory / name
    parameters = {"features": {item.name: {
        "type": item.definition.spec.type, "description": item.definition.spec.description,
        "labels": item.definition.spec.labels, "guidance": item.definition.guidance,
        "scope": item.definition.spec.scope} for item in text}}
    session.project.trial_path.write_text(json.dumps(parameters, ensure_ascii=False, indent=1), encoding="utf-8")
    try:
        group = trial_group(session.project)
        assert group is not None
        sampled = {key for item in text for key in samples[item.name]}
        example_keys = {example.key for item in text for example in item.examples}
        once = [session.rows[key] for key in sorted(sampled | example_keys)]
        twice = [session.rows[key] for key in sorted(sampled)]
        for rows, second in ((once, False), (twice, True)):
            run_groups(session.project, [group], rows, workers=session.workers, runner=session.runner,
                       second=second, directory=directory / ("second" if second else "first"))
        first = read_group(session.project, group, directory / "first")
        second_table = read_group(session.project, group, directory / "second")
    finally:
        session.project.trial_path.unlink()
    for item in text:
        feature = item.definition.spec
        sample = samples[item.name]
        values = ok_values(first, feature)
        item.values = values
        item.noise, item.compared = noise(group, first, second_table, sample)[item.name]
        # The spread is judged on the sample alone: the examples were chosen for their values.
        item.spread = spread(feature, ok_values(first, feature, set(sample)), session.context, config.mining)
        expected = _expected(item)
        revisable = not item.revised
        wrong = [example for example in item.examples
                 if example.key not in values or canonical(feature, values[example.key]) != expected[example.key]]
        labels = labels_of(feature, list(values.values())) if feature.type in ("category", "set") else set()
        objection = item.spread.objection(config.mining)
        if not values or item.noise is None:
            item.settle("reject", "the trial extraction gave no value measured by both models")
        elif item.noise > config.evaluation.noise_max:
            reason = f"noise {round(item.noise, 4)} is above the limit"
            item.revise_request = _noise_request(session, item.definition, first, second_table, sample)
            item.settle("revise" if revisable else "reject", reason)
        elif wrong:
            reason = f"the value of {len(wrong)} of {len(item.examples)} examples is not reproduced"
            evidence = _evidence(first, item.name)
            cases = [Case(file=session.link_case(example.key), first=values.get(example.key),
                          first_evidence=evidence.get(example.key), expected=example.value, steps=example.steps)
                     for example in wrong[:config.mining.revise_cases]]
            item.revise_request = ReviseRequest(definition=item.definition, reason="example", cases=cases,
                                                model=mining_model(session.project))
            item.settle("revise" if revisable else "reject", reason)
        elif objection is not None:
            item.settle("reject", objection)
        elif not feature.labels and len(labels) > config.evaluation.label_limit:
            item.revise_request = _labels_request(session, item.definition, first)
            item.settle("revise" if revisable else "reject", "more distinct labels than the limit")
        else:
            item.settle("add")


def _sample(session: Session, item: Item) -> list[str]:
    """The trajectories a judged candidate is measured on: `mining.trial_size` of its scope.

    Those of the evaluation set come first, since no function could read them, and a draw from the rest of the
    scope fills the sample. A candidate whose scope cannot be evaluated or holds too few trajectories is rejected.
    """
    project = session.project
    feature = item.definition.spec
    reference = [session.rows[key] for key in session.reference]
    # A condition written by a model can fail in any way; every failure rejects the candidate and is reported.
    try:
        scope = in_scope(project, [feature], reference).get(feature.name, set(session.reference))
    except Exception as error:
        item.settle("reject", f"the scope cannot be evaluated: {type(error).__name__}: {error}")
        return []
    if len(scope) < MIN_SCOPE:
        item.settle("reject", f"its scope holds {len(scope)} trajectories of the reference set")
        return []
    size = project.config.mining.trial_size
    held_out = [key for key in session.chosen if key in scope][:size]
    examples = {example.key for example in item.examples}
    rest = sorted(scope - set(held_out) - examples)
    rng = np.random.default_rng([project.config.evaluation.seed, session.number])
    drawn = rng.choice(rest, size=min(size - len(held_out), len(rest)), replace=False).tolist() if rest else []
    return sorted([*held_out, *drawn])


def _comparable(feature: FeatureSpec, value: Any) -> str:
    """The canonical form of a value as examples state it: for an event feature, whether any event was found."""
    if feature.type == "event":
        return canonical(feature, bool(value))
    return canonical(feature, value)


def _expected(item: Item) -> dict[str, str]:
    """The canonical value each example states, by trajectory key."""
    feature = item.definition.spec
    check = value_validator(feature.model_copy(update={"type": "boolean", "event": None})
                            if feature.type == "event" else feature)
    return {example.key: canonical(feature, check(example.value)) for example in item.examples}


def _trial_code(session: Session, item: Item) -> None:
    feature = item.definition.spec
    assert item.source is not None
    # A file written by a model can fail in any way; every failure rejects the candidate and is reported.
    try:
        operator = load_code_operator(item.source, feature)
        keys = sorted(set(session.reference) | {example.key for example in item.examples})
        values, steps = compute_operator(session.project, operator, keys)
    except Exception as error:
        item.settle("reject", f"the operator file failed: {type(error).__name__}: {error}")
        return
    item.noise = 0.0
    item.values, item.steps = values, steps
    expected = _expected(item)
    wrong = [example for example in item.examples
             if example.key not in values or _comparable(feature, values[example.key]) != expected[example.key]]
    reference = set(session.reference)
    item.spread = spread(feature, {key: value for key, value in values.items() if key in reference},
                         session.context, session.project.config.mining)
    objection = item.spread.objection(session.project.config.mining)
    if wrong:
        item.settle("reject", f"the value of {len(wrong)} of {len(item.examples)} examples is not reproduced")
    elif objection is not None:
        item.settle("reject", objection)
    else:
        item.settle("add")


def tail_cases(session: Session, item: Item) -> list[TailCase]:
    """The trajectories at the tail of the feature: the highest values, the true ones, the rarest labels, the
    largest sets, or the most unrecovered events, with the steps each value rests on."""
    feature = item.definition.spec
    size = session.project.config.alerts.tail_size
    values = {key: value for key, value in item.values.items() if value is not None}
    ranked: list[tuple[Any, str]]
    match feature.type:
        case "scalar":
            ranked = sorted(((-float(v), k) for k, v in values.items()))
        case "boolean":
            ranked = sorted((0, k) for k, v in values.items() if v)
        case "category":
            counts = Counter(values.values())
            ranked = sorted((counts[v], k) for k, v in values.items())
        case "set":
            ranked = sorted((-len(v), k) for k, v in values.items() if v)
        case "event":
            ranked = sorted((-sum(not e["recovered"] for e in v), k) for k, v in values.items()
                            if any(not e["recovered"] for e in v))
        case _:
            return []
    cases = []
    for _, key in ranked[:size]:
        value = values[key]
        steps = list(item.steps.get(key, ()))
        if feature.type == "event":
            steps = [event["step"] for event in value if not event["recovered"]]
            value = len(steps)
        cases.append(TailCase(file=session.link_case(key), value=value, steps=steps))
    return cases


def tail_check(session: Session, items: list[Item]) -> None:
    """Set the tail precision, lift and alert of every accepted item.

    A feature on which a person has given at least `alerts.min_human_verdicts` verdicts takes their precision;
    the others are asked of `mine-tail-check`.
    """
    config = session.project.config
    human = precision_by_feature(read_verdicts(session.project))
    asked = []
    judged = []
    for item in items:
        if item.decision != "add" or item.definition.spec.type not in TAIL_TYPES:
            continue
        share, count = human.get(item.name, (0.0, 0))
        if count >= config.alerts.min_human_verdicts:
            item.tail_precision = round(share, 4)
            judged.append(item)
            continue
        cases = tail_cases(session, item)
        if not cases:
            continue
        item.tail_request = TailRequest(definition=item.definition, cases=cases, model=mining_model(session.project))
        asked.append(item)
    handles = [session.issue(TAIL_CHECK, TAIL_INSTRUCTION, item.tail_request, Access.READ_ONLY)
               for item in asked if item.tail_request is not None]
    for item, result in zip(asked, session.settle(handles), strict=True):
        assert item.tail_request is not None
        if not isinstance(result.outcome, Returned):
            item.reason += f"; mine-tail-check returned no verdict: {result.status}"
            continue
        verdicts = {verdict.file: verdict for verdict in result.outcome.value.verdicts}
        worth = [verdicts[case.file].worth for case in item.tail_request.cases if case.file in verdicts]
        if not worth:
            item.reason += "; mine-tail-check answered none of the cases"
            continue
        item.tail_precision = round(float(np.mean(worth)), 4)
        judged.append(item)
    for item in judged:
        assert item.tail_precision is not None
        if item.definition.spec.type == "event":
            item.lift = _lift(session, item)
        item.alert = item.tail_precision >= config.alerts.min_tail_precision and (
            item.lift is None or item.lift >= config.alerts.min_lift)


def _lift(session: Session, item: Item) -> float | None:
    """The consequence lift of an event candidate over the reference set, against the consequences the project's
    events table holds."""
    project = session.project
    own = pd.DataFrame([{TABLE_KEY: key, "rule": item.name, "step": event["step"], "recovered": event["recovered"]}
                        for key, events in item.values.items() for event in events],
                       columns=[TABLE_KEY, "rule", "step", "recovered"])
    known = read_events(project)
    consequences = known[known["rule"].isin(project.config.alerts.consequences)] if len(known) else known
    events = pd.concat([own, consequences[[c for c in own.columns if c in consequences.columns]]],
                       ignore_index=True)
    if events.empty:
        return None
    lifts = consequence_lifts(events, read_turns(project), session.context["stratum"], project.config,
                              consequence_coverage(project), CONSEQUENCE_RULES)
    found = lifts.get(item.name)
    return None if found is None or found.lift is None else round(found.lift, 4)


def revise(session: Session, items: list[Item], taken: set[str]) -> list[Item]:
    """Ask for a revision of every definition sent to revision, and return the revised definitions.

    `taken` holds the names in use; a revision may keep the name of the definition it revises.
    The calls only read the workspace, so they run side by side.
    """
    handles = []
    for item in items:
        assert item.revise_request is not None
        handles.append(session.issue(REVISE, REVISE_INSTRUCTION, item.revise_request, Access.READ_ONLY))
    revised: list[Item] = []
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
            child = Item(definition=definition, examples=examples, hypothesis=item.hypothesis, parent=item.name,
                         replaces=item.replaces, revised=True)
            if child.name in taken and child.name != item.name:
                child.settle("reject", "the name belongs to another feature")
            taken.add(child.name)
            revised.append(child)
    return revised


def apply(session: Session, items: list[Item], evaluation: Evaluation, report: RoundReport) -> None:
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
                                       reason="the mined judgment features exceed mining.capacity"))
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
            if spec.scope:
                features[name]["scope"] = spec.scope
            if item.tail_precision is not None:
                features[name]["tail_precision"] = item.tail_precision
            if item.alert:
                features[name]["alert"] = True
        if not features:
            operators.remove(entry)
        for item in code:
            use: dict[str, Any] = {"use": f"{MINED}.{item.name}"}
            if item.tail_precision is not None:
                use["tail_precision"] = item.tail_precision
            if item.lift is not None:
                use["lift"] = item.lift
            if item.alert:
                use["alert"] = True
            operators.append(use)

    target = project.operators_dir / MINED
    for item in code:
        assert item.source is not None
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(item.source, target / f"{item.name}.py")
    edited = edit_operators(project, change)
    # Removing a feature needs no extraction: its column stays in the table and is no longer read.
    changed = [f"{MINED}-{item.name}" for item in code]
    if set(text) - dropped:
        report.extracted = project.config.mining.extract
        if report.extracted:
            changed.append(MINED)
    if changed:
        extract(edited, groups=changed, workers=session.workers, runner=session.runner)
