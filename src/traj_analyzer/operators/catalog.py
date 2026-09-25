from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from traj_analyzer.files import load_module
from traj_analyzer.operators.base import FeatureSpec, Operator
from traj_analyzer.project import CallConfig, ConfigError, OperatorUse, Project
from traj_analyzer.schema import Trajectory

LIBRARY = Path(__file__).parent / "library"
_SEGMENT = re.compile(r"[a-z][a-z0-9_]*")
_GROUP = re.compile(r"[a-z][a-z0-9_-]{0,40}")


@dataclass(frozen=True)
class OperatorRef:
    name: str
    origin: Literal["library", "project"]
    path: Path
    operator: Operator


@dataclass(frozen=True)
class Instance:
    """One enabled use of an operator, with validated params and feature names after `prefix`."""

    ref: OperatorRef
    id: str
    params: BaseModel
    features: list[FeatureSpec]
    original: dict[str, str]
    """Maps each feature name to the output name the operator gives it."""

    def applies(self, trajectory: Trajectory) -> bool:
        requires = self.ref.operator.requires
        return requires is None or requires(trajectory)

    def compute(self, trajectory: Trajectory) -> dict[str, Any]:
        compute = self.ref.operator.compute
        assert compute is not None
        values = compute(trajectory, self.params)
        expected = set(self.original.values())
        if set(values) != expected:
            raise ValueError(f"{self.ref.name} returned {sorted(values)}, expected {sorted(expected)}")
        return {name: values[output] for name, output in self.original.items()}


@dataclass(frozen=True)
class Group:
    """What runs together: one code operator instance, or all LLM operator instances of one call."""

    name: str
    kind: Literal["code", "llm"]
    instances: list[Instance]
    evidence: bool = False
    model: str | None = None

    @property
    def features(self) -> list[FeatureSpec]:
        return [f for i in self.instances for f in i.features]

    @property
    def is_code(self) -> bool:
        return self.kind == "code"

    @property
    def function_name(self) -> str:
        return f"feat-{self.name}"


def discover(project: Project | None) -> dict[str, OperatorRef]:
    """All operators of the built-in library and, when given, of the project; project operators override."""
    roots: list[tuple[Literal["library", "project"], Path]] = [("library", LIBRARY)]
    if project is not None and project.operators_dir.is_dir():
        roots.append(("project", project.operators_dir))
        # Project operators import their shared underscore modules by path from this root, e.g. `rca._parse`.
        if str(project.operators_dir) not in sys.path:
            sys.path.insert(0, str(project.operators_dir))
    refs: dict[str, OperatorRef] = {}
    for origin, root in roots:
        for path in sorted(root.rglob("*.py")):
            if path.name.startswith("_"):
                continue
            parts = path.relative_to(root).with_suffix("").parts
            for part in parts:
                if not _SEGMENT.fullmatch(part):
                    raise ConfigError(f"Operator path segment {part!r} in {path} must be snake_case")
            operator = getattr(load_module(path), "OPERATOR", None)
            if not isinstance(operator, Operator):
                raise ConfigError(f"{path} must define OPERATOR as an Operator")
            name = ".".join(parts)
            refs[name] = OperatorRef(name=name, origin=origin, path=path, operator=operator)
    return refs


def instantiate(ref: OperatorRef, use: OperatorUse) -> Instance:
    params = ref.operator.params.model_validate(use.params)
    outputs = ref.operator.outputs(params)
    if not outputs:
        raise ConfigError(f"{ref.name} produces no outputs")
    original: dict[str, str] = {}
    features = []
    for output in outputs:
        name = f"{use.prefix}_{output.name}" if use.prefix else output.name
        original[name] = output.name
        features.append(FeatureSpec.model_validate({**output.model_dump(), "name": name}))
    return Instance(ref=ref, id=use.alias or ref.name, params=params, features=features, original=original)


def load_groups(project: Project, only: list[str] | None = None) -> list[Group]:
    refs = discover(project)
    code: list[Group] = []
    calls: dict[str, list[Instance]] = {}
    for use in project.config.operators:
        if use.use not in refs:
            raise ConfigError(f"Unknown operator {use.use}; see `traj operators list`")
        ref = refs[use.use]
        instance = instantiate(ref, use)
        if ref.operator.kind == "code":
            if use.call is not None:
                raise ConfigError(f"{use.use} is a code operator and takes no call")
            code.append(Group(name=instance.id.replace(".", "-"), kind="code", instances=[instance]))
        else:
            calls.setdefault(use.call or "default", []).append(instance)
    unknown_calls = sorted(set(project.config.calls) - set(calls))
    if unknown_calls:
        raise ConfigError(f"calls configured without enabled llm operators: {unknown_calls}")
    llm = []
    for call, instances in calls.items():
        config = project.config.calls.get(call, CallConfig())
        llm.append(Group(name=call, kind="llm", instances=instances, evidence=config.evidence, model=config.model))
    groups = [*code, *llm]
    _check_names(groups)
    if only:
        missing = set(only) - {g.name for g in groups}
        if missing:
            raise ConfigError(f"Unknown groups: {sorted(missing)}")
        groups = [g for g in groups if g.name in only]
    return groups


def _check_names(groups: list[Group]) -> None:
    seen_groups: set[str] = set()
    owners: dict[str, str] = {}
    for group in groups:
        if not _GROUP.fullmatch(group.name):
            raise ConfigError(f"Group name {group.name!r} must match {_GROUP.pattern}")
        if group.name in seen_groups:
            raise ConfigError(f"Two groups are named {group.name}; rename a call or use `as`")
        seen_groups.add(group.name)
        for instance in group.instances:
            for feature in instance.features:
                if feature.name in owners:
                    raise ConfigError(f"Feature {feature.name} comes from both {owners[feature.name]} and "
                                      f"{instance.id}; use `prefix` to rename one")
                owners[feature.name] = instance.id
