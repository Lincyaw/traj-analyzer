from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel
from ruamel.yaml import YAML

from traj_analyzer.files import load_module
from traj_analyzer.layers.tables import Tables
from traj_analyzer.operators.base import Aggregate, FeatureSpec, Operator, Unit
from traj_analyzer.project import (
    CONFIG_FILE,
    CallConfig,
    Config,
    ConfigError,
    OperatorUse,
    Project,
    SampleConfig,
    WindowConfig,
)
from traj_analyzer.schema import Trajectory

LIBRARY = Path(__file__).parent / "library"
ROOT = "root"
"""The key of the project root in the context the params of an operator are validated with."""
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
    """One enabled use of an operator, with validated params and feature names after `prefix`.

    `features` are the trajectory features: the outputs of a code operator or of a judgment operator over whole
    trajectories, and the aggregates of a judgment operator over turns or windows, whose per-unit `outputs` go to
    the judgments table.
    """

    ref: OperatorRef
    id: str
    params: BaseModel
    features: list[FeatureSpec]
    original: dict[str, str]
    """Maps each feature name to the output name the operator gives it."""
    outputs: list[FeatureSpec] = field(default_factory=list)
    output_names: dict[str, str] = field(default_factory=dict)
    """Maps each per-unit output name the operator gives to the name after `prefix`."""
    aggregates: list[Aggregate] = field(default_factory=list)
    use: OperatorUse | None = None

    @property
    def unit(self) -> Unit:
        return self.ref.operator.unit

    def applies(self, trajectory: Trajectory) -> bool:
        requires = self.ref.operator.requires
        return requires is None or requires(trajectory)

    def compute(self, trajectory: Trajectory, tables: Tables) -> dict[str, Any]:
        compute = self.ref.operator.compute
        assert compute is not None
        values = compute(trajectory, tables, self.params)
        expected = set(self.original.values())
        if set(values) != expected:
            raise ValueError(f"{self.ref.name} returned {sorted(values)}, expected {sorted(expected)}")
        return {name: values[output] for name, output in self.original.items()}


@dataclass(frozen=True)
class Group:
    """What runs together: one code operator instance, or all judgment operator instances of one call."""

    name: str
    kind: Literal["code", "judgment"]
    instances: list[Instance]
    evidence: bool = False
    model: str | None = None
    unit: Unit = "trajectory"
    sample: SampleConfig | None = None
    window: WindowConfig = field(default_factory=WindowConfig)
    context: int = 8
    turns_per_call: int = 20
    tail: int | None = None

    @property
    def features(self) -> list[FeatureSpec]:
        return [f for i in self.instances for f in i.features]

    def parts(self, units: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
        """The units of a trajectory as the group's calls take them: one list per call."""
        if self.unit == "turn":
            size = self.turns_per_call
            return [units[i:i + size] for i in range(0, len(units), size)]
        return [[unit] for unit in units]

    @property
    def reads_file(self) -> bool:
        """Whether a call reads the rendered trajectory file itself, as a group over whole trajectories without
        `tail` does; the other groups read a file rendered for the call."""
        return self.unit == "trajectory" and self.tail is None

    @property
    def unit_outputs(self) -> list[FeatureSpec]:
        """The per-unit outputs of a group over turns or windows; empty for a group over whole trajectories."""
        return [f for i in self.instances for f in i.outputs] if self.unit != "trajectory" else []

    @property
    def is_code(self) -> bool:
        return self.kind == "code"

    @property
    def tables(self) -> tuple[str, ...]:
        return tuple(sorted({t for i in self.instances for t in i.ref.operator.tables_with(i.params)}))

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


def instantiate(ref: OperatorRef, use: OperatorUse, root: Path) -> Instance:
    """The operator with the params of `use`, validated with the project `root` in their context, for params that
    read files of the project."""
    params = ref.operator.params.model_validate(use.params, context={ROOT: root})
    outputs = ref.operator.outputs(params)
    if not outputs:
        raise ConfigError(f"{ref.name} produces no outputs with the params {use.params}")

    def renamed(spec: FeatureSpec) -> FeatureSpec:
        if len(spec.name) > 48:
            raise ConfigError(f"{ref.name}: output name {spec.name!r} exceeds 48 characters before its prefix")
        name = f"{use.prefix}_{spec.name}" if use.prefix else spec.name
        return FeatureSpec.model_validate({**spec.model_dump(), "name": name})

    aggregates = ref.operator.aggregates(params) if ref.operator.aggregates is not None else []
    unit_outputs: list[FeatureSpec] = []
    output_names: dict[str, str] = {}
    if ref.operator.unit != "trajectory":
        if not aggregates:
            raise ConfigError(f"{ref.name} judges {ref.operator.unit}s and declares no aggregates")
        by_name = {spec.name: spec for spec in outputs}
        features = [renamed(aggregate.spec(by_name)) for aggregate in aggregates]
        original = {renamed(aggregate.spec(by_name)).name: aggregate.name for aggregate in aggregates}
        unit_outputs = [renamed(spec) for spec in outputs]
        output_names = {spec.name: renamed(spec).name for spec in outputs}
    else:
        features = [renamed(spec) for spec in outputs]
        original = {renamed(spec).name: spec.name for spec in outputs}
    return Instance(ref=ref, id=use.alias or ref.name, params=params, features=features, original=original,
                    outputs=unit_outputs, output_names=output_names, aggregates=aggregates, use=use)


def load_groups(project: Project, only: list[str] | None = None) -> list[Group]:
    refs = discover(project)
    code: list[Group] = []
    calls: dict[str, list[Instance]] = {}
    for use in project.config.operators:
        if use.use not in refs:
            raise ConfigError(f"Unknown operator {use.use}; see `traj operators list`")
        ref = refs[use.use]
        instance = instantiate(ref, use, project.root)
        if ref.operator.kind == "code":
            if use.call is not None:
                raise ConfigError(f"{use.use} is a code operator and takes no call")
            code.append(Group(name=instance.id.replace(".", "-"), kind="code", instances=[instance]))
        else:
            calls.setdefault(use.call or "default", []).append(instance)
    unknown_calls = sorted(set(project.config.calls) - set(calls))
    if unknown_calls:
        raise ConfigError(f"calls configured without enabled judgment operators: {unknown_calls}")
    judged = []
    for call, instances in calls.items():
        units = {instance.unit for instance in instances}
        if len(units) > 1:
            raise ConfigError(f"call {call} mixes operators over {sorted(units)}; one call judges one kind of unit")
        config = project.config.calls.get(call, CallConfig())
        judged.append(Group(name=call, kind="judgment", instances=instances, evidence=config.evidence,
                            model=config.model, unit=units.pop(), sample=config.sample, window=config.window,
                            context=config.context, turns_per_call=config.turns_per_call, tail=config.tail))
    groups = [*code, *judged]
    _check_names(groups)
    if only:
        missing = set(only) - {g.name for g in groups}
        if missing:
            raise ConfigError(f"Unknown groups: {sorted(missing)}")
        groups = [g for g in groups if g.name in only]
    return groups


TRIAL = "mined-trial"
"""Name of the group of judged candidates a mining round extracts before deciding on them."""


def trial_group(project: Project) -> Group | None:
    """The candidates under trial, read from the file the mining round wrote; None outside a round."""
    if not project.trial_path.is_file():
        return None
    use = OperatorUse(use="llm.features", alias=TRIAL, params=json.loads(project.trial_path.read_text("utf-8")))
    return Group(name=TRIAL, kind="judgment",
                 instances=[instantiate(discover(project)["llm.features"], use, project.root)], evidence=True)


def runtime_groups(project: Project) -> list[Group]:
    """Every group a worker may be asked to run: the enabled groups and the candidates under trial."""
    trial = trial_group(project)
    return [*load_groups(project), *([trial] if trial else [])]


def edit_operators(project: Project, change: Callable[[list[Any]], None]) -> Project:
    """Apply `change` to the operators list of traj.yaml, validate the result, then write it back.

    The file keeps its comments and layout. Returns the project with the new configuration.
    """
    path = project.root / CONFIG_FILE
    editor = YAML()
    editor.preserve_quotes = True
    editor.width = 4096
    editor.indent(mapping=2, sequence=4, offset=2)
    editor.representer.add_representer(
        type(None), lambda representer, _: representer.represent_scalar("tag:yaml.org,2002:null", "null"))
    document = editor.load(path)
    if "operators" not in document:
        document["operators"] = []
    change(document["operators"])
    edited = Project(project.root, Config.model_validate(json.loads(json.dumps(document))))
    load_groups(edited)
    editor.dump(document, path)
    return edited


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
        outputs = [f.name for f in group.unit_outputs]
        if len(set(outputs)) != len(outputs):
            raise ConfigError(f"Group {group.name} has two per-unit outputs of the same name; use `prefix`")
