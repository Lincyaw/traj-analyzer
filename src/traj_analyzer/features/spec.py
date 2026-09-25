from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, TypeAdapter, create_model

from traj_analyzer.operators.base import FeatureSpec
from traj_analyzer.operators.catalog import Group
from traj_analyzer.project import Project


def _distinct(value: list[str]) -> list[str]:
    if len(set(value)) != len(value):
        raise ValueError("items must be distinct")
    return value


def value_type(feature: FeatureSpec) -> Any:
    number: Any = float
    if feature.range:
        number = Annotated[float, Field(ge=feature.range[0], le=feature.range[1])]
    labels = tuple(feature.labels or ())
    label: Any = Literal[labels] if labels else Annotated[str, Field(min_length=1)]
    match feature.type:
        case "scalar":
            return number
        case "boolean":
            return bool
        case "category":
            return label
        case "set":
            return Annotated[list[label], Field(json_schema_extra={"uniqueItems": True}), AfterValidator(_distinct)]
        case "vector":
            if feature.length:
                return Annotated[list[number], Field(min_length=feature.length, max_length=feature.length)]
            return Annotated[list[number], Field(min_length=1)]
        case "map":
            return dict[label, number]
    raise AssertionError(feature.type)


def value_validator(feature: FeatureSpec) -> Callable[[Any], Any]:
    """Checker for a code operator's values of this feature; None means the feature does not apply."""
    adapter: TypeAdapter[Any] = TypeAdapter(value_type(feature) | None)
    return lambda value: adapter.dump_python(adapter.validate_python(value), mode="json")


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
    """Input of one LLM call; aifn reuses an earlier answer only when every field and the instruction match."""

    trajectory_file: str
    n_chunks: int
    sha256: str
    """Content hash of the Markdown file, so a re-ingested trajectory is judged again."""
    model: str
    """Model that answers, so changing the model is not served answers of the previous one."""


_PREAMBLE = """\
# Extract features from one trajectory

`input.json` names a Markdown file, `trajectory_file`, relative to the working directory.
The file is one LLM trajectory, such as a conversation or an agent run.
It opens with a description of the dataset, when one is given, and a JSON block of metadata; read both first.
Every step starts with a heading `### #<n> <role> · <kind>[ · <name>]`.
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
    shape = {
        "boolean": "true or false.",
        "category": "One label.",
        "set": "A list of distinct labels, possibly empty.",
        "map": "An object from label to number; leave out labels that do not occur.",
    }
    if feature.type in shape:
        lines.append(f"- {shape[feature.type]}")
    if feature.labels:
        lines.append("- The labels are:")
        lines += [f"  - `{label}`: {text}" for label, text in feature.labels.items()]
    return "\n".join(lines) + "\n"


def write_instruction(project: Project, group: Group) -> Path:
    path = project.instructions_dir / f"{group.name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_instruction(group), encoding="utf-8")
    return path
