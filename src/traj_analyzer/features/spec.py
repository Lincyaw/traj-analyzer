from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    create_model,
    field_validator,
    model_validator,
)

from traj_analyzer.features.builtin import REGISTRY
from traj_analyzer.project import ConfigError, Project, read_yaml

FeatureType = Literal["scalar", "boolean", "category", "set", "vector", "distribution"]
_NAME = re.compile(r"[a-z][a-z0-9_]{0,47}")


class FeatureSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: FeatureType
    description: str
    range: tuple[float, float] | None = None
    labels: dict[str, str] | None = None
    per: Literal["chunk"] | None = None
    length: int | None = Field(default=None, ge=1)
    thresholds: dict[str, float] = Field(default_factory=dict)
    fn: str | None = None

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        if not _NAME.fullmatch(value) or "__" in value:
            raise ValueError("feature name must be snake_case without '__'")
        return value

    @model_validator(mode="after")
    def _shape(self) -> FeatureSpec:
        if self.type in ("category", "distribution") and not self.labels:
            raise ValueError(f"{self.name}: type {self.type} needs labels")
        if self.type == "vector" and (self.per is None) == (self.length is None):
            raise ValueError(f"{self.name}: vector needs exactly one of per and length")
        if self.type != "vector" and (self.per or self.length):
            raise ValueError(f"{self.name}: per and length apply only to vectors")
        if self.range and self.range[0] > self.range[1]:
            raise ValueError(f"{self.name}: range is reversed")
        if self.fn is not None and self.fn not in REGISTRY:
            raise ValueError(f"{self.name}: unknown builtin fn {self.fn}")
        return self


class GroupSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group: str
    description: str = ""
    engine: Literal["deepseek", "builtin"] = "deepseek"
    model: str | None = None
    evidence: bool = True
    guidance: str = ""
    features: list[FeatureSpec] = Field(min_length=1)

    @field_validator("group")
    @classmethod
    def _group(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,40}", value):
            raise ValueError("group name must match [a-z][a-z0-9_-]{0,40}")
        return value

    @model_validator(mode="after")
    def _functions(self) -> GroupSpec:
        for feature in self.features:
            if self.is_builtin != (feature.fn is not None):
                raise ValueError(f"{feature.name}: fn is required in builtin groups only")
        return self

    @property
    def is_builtin(self) -> bool:
        return self.engine == "builtin"

    @property
    def function_name(self) -> str:
        return f"feat-{self.group}"

    def spec_hash(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:12]


def load_groups(project: Project, only: list[str] | None = None) -> list[GroupSpec]:
    groups = [GroupSpec.model_validate(read_yaml(path))
              for path in sorted(project.features_dir.glob("*.yaml"))]
    owners: dict[str, str] = {}
    for group in groups:
        if group.group in owners.values():
            raise ConfigError(f"Duplicate group name: {group.group}")
        for feature in group.features:
            if feature.name in owners:
                raise ConfigError(f"Feature {feature.name} is defined in both "
                                  f"{owners[feature.name]} and {group.group}")
            owners[feature.name] = group.group
    if only:
        missing = set(only) - {g.group for g in groups}
        if missing:
            raise ConfigError(f"Unknown groups: {sorted(missing)}")
        groups = [g for g in groups if g.group in only]
    return groups


def _sums_to_one(value: dict[str, float]) -> dict[str, float]:
    total = sum(value.values())
    if abs(total - 1.0) > 0.02:
        raise ValueError(f"probabilities must sum to 1, got {total:.3f}")
    return value


def value_type(feature: FeatureSpec) -> Any:
    number: Any = float
    if feature.range:
        number = Annotated[float, Field(ge=feature.range[0], le=feature.range[1])]
    labels = tuple(feature.labels or ())
    match feature.type:
        case "scalar":
            return number
        case "boolean":
            return bool
        case "category":
            return Literal[labels]  # type: ignore[valid-type]
        case "set":
            item = Literal[labels] if labels else str  # type: ignore[valid-type]
            return Annotated[list[item], Field(json_schema_extra={"uniqueItems": True}),
                             AfterValidator(_distinct)]
        case "vector":
            if feature.length:
                return Annotated[list[number], Field(min_length=feature.length,
                                                     max_length=feature.length)]
            return Annotated[list[number], Field(min_length=1)]
        case "distribution":
            unit = Annotated[float, Field(ge=0, le=1)]
            return Annotated[dict[Literal[labels], unit],  # type: ignore[valid-type]
                             AfterValidator(_sums_to_one)]
    raise AssertionError(feature.type)


def _distinct(value: list[str]) -> list[str]:
    if len(set(value)) != len(value):
        raise ValueError("items must be distinct")
    return value


def output_model(group: GroupSpec) -> type[BaseModel]:
    fields: dict[str, Any] = {}
    for feature in group.features:
        vtype = value_type(feature)
        if group.evidence:
            vtype = create_model(
                f"{_camel(feature.name)}Answer",
                __config__=ConfigDict(extra="forbid"),
                value=(vtype, ...),
                evidence=(str, Field(description="Why, citing step numbers like #12")),
            )
        fields[feature.name] = (vtype, Field(description=feature.description))
    return create_model(
        f"{_camel(group.group)}Features", __config__=ConfigDict(extra="forbid"), **fields
    )


def _camel(name: str) -> str:
    return "".join(part.capitalize() for part in re.split(r"[_-]", name))


class ExtractRequest(BaseModel):
    trajectory_file: str
    n_chunks: int
    sha256: str


_PREAMBLE = """\
# Extract features from one trajectory

`input.json` names a Markdown file, `trajectory_file`, relative to the working directory.
The file is one LLM trajectory, such as a conversation or an agent run.
Every step starts with a heading `### #<n> <role> · <kind>[ · <tool name>]`.
The file has `n_chunks` chunks, each starting with a line `<!-- chunk <k> -->`, where k counts from 0.

Read the whole file in parts, since it may be long.
Fill in every feature below according to its definition, and write the answer to match `output.schema.json`.
Judge only from what the trajectory shows.
Do not read any other file.
"""

_EVIDENCE = """\
Each feature is an object `{"value": ..., "evidence": "..."}`.
The evidence is one or two sentences citing the steps the judgement rests on, written as `#12` or `#12-#15`.
"""

_REFUSE = """\
Refuse with refusal.json and refuse_output only when the file is missing or unreadable.
A trajectory with little content still gets an answer that fits what is there.
"""


def render_instruction(group: GroupSpec) -> str:
    parts = [_PREAMBLE]
    if group.description:
        parts.append(f"This group of features is about the following.\n{group.description.strip()}\n")
    if group.guidance:
        parts.append(f"## Guidance\n\n{group.guidance.strip()}\n")
    parts.append("## Features\n")
    parts += [_render_feature(feature) for feature in group.features]
    if group.evidence:
        parts.append("## Evidence\n\n" + _EVIDENCE)
    parts.append("## Refusing\n\n" + _REFUSE)
    return "\n".join(parts)


def _render_feature(feature: FeatureSpec) -> str:
    lines = [f"### `{feature.name}` ({feature.type})", "", feature.description.strip(), ""]
    if feature.range:
        lines.append(f"- Range: {feature.range[0]} to {feature.range[1]}.")
    if feature.per == "chunk":
        lines.append("- One number per chunk in chunk order, so the list has exactly "
                     "`n_chunks` entries (see input.json).")
    if feature.length:
        lines.append(f"- Exactly {feature.length} numbers, evenly spaced over the trajectory.")
    if feature.type == "boolean":
        lines.append("- true or false.")
    if feature.labels:
        heading = {
            "category": "Exactly one of these labels:",
            "set": "Any subset of these labels, possibly empty:",
            "distribution": "A probability for each of these labels, summing to 1:",
        }[feature.type]
        lines.append(f"- {heading}")
        lines += [f"  - `{label}`: {text}" for label, text in feature.labels.items()]
    elif feature.type == "set":
        lines.append("- A list of short distinct strings, possibly empty.")
    return "\n".join(lines) + "\n"


def write_instruction(project: Project, group: GroupSpec) -> Path:
    """Write the instruction file atomically, since worker processes write it concurrently."""
    text = render_instruction(group)
    path = project.instructions_dir / f"{group.group}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return path
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    return path
