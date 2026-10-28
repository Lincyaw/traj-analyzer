from __future__ import annotations

from pathlib import Path
from typing import Any

from traj_analyzer.features.spec import value_validator
from traj_analyzer.files import load_module
from traj_analyzer.harness import discover_harnesses, harness_for
from traj_analyzer.ingest import load_trajectory
from traj_analyzer.layers import compute_tables, enabled_layers
from traj_analyzer.operators.base import Evidenced, FeatureSpec, Operator
from traj_analyzer.project import Project


def load_code_operator(path: Path, spec: FeatureSpec | None = None) -> Operator:
    """Load an operator file with one output; with `spec`, require exactly that output."""
    operator = getattr(load_module(path), "OPERATOR", None)
    if not isinstance(operator, Operator) or operator.kind != "code":
        raise ValueError("the file does not define OPERATOR as a code operator")
    outputs = operator.outputs(operator.params())
    if len(outputs) != 1:
        raise ValueError(f"the operator has {len(outputs)} outputs; a mined operator has one")
    if spec is not None and outputs[0].model_dump() != spec.model_dump():
        raise ValueError(f"outputs differ from the candidate's spec: {[f.model_dump() for f in outputs]}")
    return operator


def compute_operator(
    project: Project, operator: Operator, keys: list[str]
) -> tuple[dict[str, Any], dict[str, tuple[int, ...]]]:
    """The operator's value on each of these trajectories, leaving out those it does not apply to, and the steps
    each value rests on when the operator gives them."""
    (feature,) = operator.outputs(operator.params())
    check = value_validator(feature)
    params = operator.params()
    layers = enabled_layers(project, set(operator.tables_with(params)))
    harnesses = discover_harnesses(project)
    values: dict[str, Any] = {}
    steps: dict[str, tuple[int, ...]] = {}
    for key in keys:
        trajectory = load_trajectory(project, key)
        if operator.requires is not None and not operator.requires(trajectory):
            continue
        _, tables = compute_tables(trajectory, layers, project.config.render, harness_for(trajectory, harnesses))
        value = operator.compute(trajectory, tables, params)[feature.name]
        cited: tuple[int, ...] = ()
        if isinstance(value, Evidenced):
            value, cited = value.value, value.steps
        value = check(value)
        if value is not None:
            values[key] = value
            steps[key] = cited
    return values, steps
