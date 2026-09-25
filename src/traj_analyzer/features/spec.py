from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, TypeAdapter, create_model

from traj_analyzer.operators.base import FeatureSpec
from traj_analyzer.operators.catalog import Group
from traj_analyzer.project import Project


def _sums_to_one(value: dict[str, float]) -> dict[str, float]:
    total = sum(value.values())
    if abs(total - 1.0) > 0.02:
        raise ValueError(f"probabilities must sum to 1, got {total:.3f}")
    return value


def _distinct(value: list[str]) -> list[str]:
    if len(set(value)) != len(value):
        raise ValueError("items must be distinct")
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
            return Literal[labels] if labels else Annotated[str, Field(min_length=1)]  # type: ignore[valid-type]
        case "set":
            item = Literal[labels] if labels else str  # type: ignore[valid-type]
            return Annotated[list[item], Field(json_schema_extra={"uniqueItems": True}),
                             AfterValidator(_distinct)]
        case "vector":
            if feature.length:
                return Annotated[list[number], Field(min_length=feature.length, max_length=feature.length)]
            return Annotated[list[number], Field(min_length=1)]
        case "distribution":
            unit = Annotated[float, Field(ge=0, le=1)]
            return Annotated[dict[Literal[labels], unit],  # type: ignore[valid-type]
                             AfterValidator(_sums_to_one)]
    raise AssertionError(feature.type)


def validate_value(feature: FeatureSpec, value: Any) -> Any:
    """Check a code operator's value against its feature type; None means the feature does not apply."""
    adapter: TypeAdapter[Any] = TypeAdapter(value_type(feature) | None)
    return adapter.dump_python(adapter.validate_python(value), mode="json")


def output_model(group: Group) -> type[BaseModel]:
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
    return create_model(f"{_camel(group.name)}Features", __config__=ConfigDict(extra="forbid"), **fields)


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


def render_instruction(group: Group) -> str:
    parts = [_PREAMBLE, "## Features\n"]
    for instance in group.instances:
        operator = instance.ref.operator
        parts.append(f"### Operator `{instance.id}`\n\n{operator.description}\n")
        if operator.guidance is not None:
            parts.append(operator.guidance(instance.params).strip() + "\n")
        parts += [_render_feature(feature) for feature in instance.features]
    if group.evidence:
        parts.append("## Evidence\n\n" + _EVIDENCE)
    parts.append("## Refusing\n\n" + _REFUSE)
    return "\n".join(parts)


def _render_feature(feature: FeatureSpec) -> str:
    lines = [f"#### `{feature.name}` ({feature.type})", "", feature.description.strip(), ""]
    if feature.range:
        lines.append(f"- Range: {feature.range[0]} to {feature.range[1]}.")
    if feature.per == "chunk":
        lines.append("- One number per chunk in chunk order, so the list has exactly `n_chunks` entries "
                     "(see input.json).")
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
    elif feature.type == "category":
        lines.append("- One short label.")
    return "\n".join(lines) + "\n"


def write_instruction(project: Project, group: Group) -> Path:
    """Write the instruction file atomically, since worker processes write it concurrently."""
    text = render_instruction(group)
    path = project.instructions_dir / f"{group.name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return path
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    return path
