from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, TypeAdapter, create_model

from traj_analyzer.operators.base import Event, FeatureSpec, Unit
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
        case "paths":
            depth = len(feature.levels or ())
            return list[Annotated[list[str], Field(min_length=depth, max_length=depth)]]
        case "event":
            return list[Event]
    raise AssertionError(feature.type)


def value_validator(feature: FeatureSpec) -> Callable[[Any], Any]:
    """Checker for a code operator's values of this feature; None means the feature does not apply."""
    adapter: TypeAdapter[Any] = TypeAdapter(value_type(feature) | None)
    return lambda value: adapter.dump_python(adapter.validate_python(value), mode="json")


def _answer_model(name: str, features: list[FeatureSpec], evidence: bool) -> type[BaseModel]:
    fields: dict[str, Any] = {}
    for feature in features:
        vtype = value_type(feature)
        if evidence:
            vtype = create_model(
                f"{_camel(feature.name)}Answer",
                __config__=ConfigDict(extra="forbid"),
                value=(vtype, ...),
                evidence=(str, Field(description="Why, citing step numbers like #12")),
            )
        fields[feature.name] = (vtype, Field(description=feature.description))
    return create_model(name, __config__=ConfigDict(extra="forbid"), **fields)


def output_model(group: Group) -> type[BaseModel]:
    """What a call of the group returns: the features of a trajectory, one object per turn, or the outputs of one
    window."""
    name = _camel(group.name)
    if group.unit == "trajectory":
        return _answer_model(f"{name}Features", group.features, group.evidence)
    unit = _answer_model(f"{name}Unit", group.unit_outputs, group.evidence)
    if group.unit == "window":
        return unit
    return create_model(f"{name}Units", __config__=ConfigDict(extra="forbid"),
                        units=(list[unit], Field(description="One object per unit of input.json, in order")))  # type: ignore[valid-type]


def _camel(name: str) -> str:
    return "".join(part.capitalize() for part in re.split(r"[_-]", name))


class ExtractRequest(BaseModel):
    """Input of one judgment call; aifn reuses an earlier answer only when every field and the instruction match.

    Fields at their default are left out of the request, so a call over a whole trajectory keeps the identity it
    had before units existed.
    """

    trajectory_file: str
    n_chunks: int
    sha256: str
    """Content hash of the Markdown file, so a re-ingested trajectory is judged again."""
    model: str
    """Model that answers, so changing the model is not served answers of the previous one."""
    unit: Unit = "trajectory"
    units: list[tuple[int, int]] = Field(default_factory=list)
    """For a call over turns: the first and last step of every turn, in order."""
    index: int | None = None
    """For a call over one window: its position among the windows of the trajectory."""


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

_PREAMBLE_TURN = """\
# Judge every message the user typed in one trajectory

`input.json` names a Markdown file, `trajectory_file`, relative to the working directory.
The file holds turns of one LLM trajectory, such as a conversation or an agent run: all of them, or one part of
them when the trajectory has many.
It opens with a description of the dataset, when one is given, and a JSON block of metadata; read both first.
Then comes one section `## Turn <k>: steps #<start> to #<end>` per message the user typed, in order: under
`### Before the message`, the steps just before it, which show what the assistant did last; under
`### The message`, the message itself. Every step starts with a heading `### #<n> <role> · <kind>[ · <name>]`.

`units` in `input.json` lists the same turns as `[start, end]`.
Return `units`, one object per turn in the same order, so that `units` has exactly as many entries as the file
has turns; count the `## Turn` sections before you answer.
Fill in every feature below for each turn according to its definition, judging the message against the steps
before it, and write the answer to match `output.schema.json`.
Judge only from what the file shows.
Do not read any other file.
"""

_PREAMBLE_TAIL = """\
# Judge the end of one trajectory

`input.json` names a Markdown file, `trajectory_file`, relative to the working directory.
The file holds the end of an LLM trajectory, such as a conversation or an agent run.
It opens with a description of the dataset, when one is given, a JSON block of metadata, the section `## Request`
with the last message the user typed, and the section `## Steps` with the last steps of the trajectory.
Every step starts with a heading `### #<n> <role> · <kind>[ · <name>]`.

Fill in every feature below according to its definition, and write the answer to match `output.schema.json`.
Judge only from what the file shows.
Do not read any other file.
"""

_PREAMBLE_WINDOW = """\
# Judge one window of a trajectory

`input.json` names a Markdown file, `trajectory_file`, relative to the working directory.
The file holds one window of an LLM trajectory, such as a conversation or an agent run.
It opens with a description of the dataset, when one is given, a JSON block of metadata, the section `## Request`
with the last message the user typed before or inside the window, and the section `## Steps` with the steps of the
window. Every step starts with a heading `### #<n> <role> · <kind>[ · <name>]`.

Fill in every feature below for this window according to its definition, and write the answer to match
`output.schema.json`.
Judge only from what the window shows, against the request.
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
    preamble = {"trajectory": _PREAMBLE, "turn": _PREAMBLE_TURN, "window": _PREAMBLE_WINDOW}[group.unit]
    if group.unit == "trajectory" and group.tail is not None:
        preamble = _PREAMBLE_TAIL
    parts = [preamble, "## Features\n"]
    for instance in group.instances:
        operator = instance.ref.operator
        parts.append(f"### Operator `{instance.id}`\n\n{operator.description}\n")
        if operator.guidance is not None:
            parts.append(operator.guidance(instance.params).strip() + "\n")
        parts += [_render_feature(feature) for feature in (instance.outputs or instance.features)]
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
