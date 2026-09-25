from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from traj_analyzer.operators.base import FeatureSpec, Operator
from traj_analyzer.project import CallConfig, ConfigError, OperatorUse, Project
from traj_analyzer.schema import Trajectory

LIBRARY = Path(__file__).parent / "library"
_SEGMENT = re.compile(r"[a-z][a-z0-9_]*")
_GROUP = re.compile(r"[a-z][a-z0-9_-]{0,40}")
_LOADED: dict[Path, Operator] = {}


@dataclass(frozen=True)
class OperatorRef:
    name: str
    origin: Literal["library", "project"]
    path: Path
    operator: Operator
    source_sha: str


@dataclass(frozen=True)
class Instance:
    """One enabled use of an operator, with validated params and feature names after `as`."""

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
    guidance: str = ""
    features: list[FeatureSpec] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", [f for i in self.instances for f in i.features])

    @property
    def is_code(self) -> bool:
        return self.kind == "code"

    @property
    def function_name(self) -> str:
        return f"feat-{self.name}"

    def spec_hash(self) -> str:
        payload = {
            "kind": self.kind, "evidence": self.evidence, "model": self.model, "guidance": self.guidance,
            "instances": [{"operator": i.ref.name, "source": i.ref.source_sha, "id": i.id,
                           "params": i.params.model_dump(mode="json")} for i in self.instances],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


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
        # Files starting with an underscore hold code shared by operators, so they count toward every source hash.
        shared = b"".join(p.read_bytes() for p in sorted(root.rglob("_*.py")))
        for path in sorted(root.rglob("*.py")):
            if path.name.startswith("_"):
                continue
            parts = path.relative_to(root).with_suffix("").parts
            for part in parts:
                if not _SEGMENT.fullmatch(part):
                    raise ConfigError(f"Operator path segment {part!r} in {path} must be snake_case")
            name = ".".join(parts)
            refs[name] = OperatorRef(name=name, origin=origin, path=path, operator=_load(path),
                                     source_sha=hashlib.sha256(path.read_bytes() + shared).hexdigest()[:12])
    return refs


def _load(path: Path) -> Operator:
    if path not in _LOADED:
        name = f"traj_operator_{hashlib.sha256(str(path).encode()).hexdigest()[:16]}"
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        # Pydantic resolves postponed annotations through sys.modules, so the module must be registered first.
        sys.modules[name] = module
        spec.loader.exec_module(module)
        operator = getattr(module, "OPERATOR", None)
        if not isinstance(operator, Operator):
            raise ConfigError(f"{path} must define OPERATOR as an Operator")
        _LOADED[path] = operator
    return _LOADED[path]


def instantiate(ref: OperatorRef, use: OperatorUse) -> Instance:
    params = ref.operator.params.model_validate(use.params)
    outputs = ref.operator.outputs(params)
    if not outputs:
        raise ConfigError(f"{ref.name} produces no outputs")
    original: dict[str, str] = {}
    features = []
    for output in outputs:
        name = output.name
        if use.alias is not None and ref.operator.rename_outputs:
            name = use.alias if len(outputs) == 1 else f"{use.alias}_{output.name}"
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
        llm.append(Group(name=call, kind="llm", instances=instances, evidence=config.evidence, model=config.model,
                         guidance=config.guidance))
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
                                      f"{instance.id}; use `as` to rename one")
                owners[feature.name] = instance.id
