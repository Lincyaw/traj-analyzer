from __future__ import annotations

from typing import Annotated, Any, Literal

from aifn import AiFunction, Model
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, create_model, model_validator

from traj_analyzer.features.spec import value_type
from traj_analyzer.operators.base import FeatureSpec
from traj_analyzer.operators.catalog import Group
from traj_analyzer.project import Project

PROPOSE = "mine-propose"
CONSOLIDATE = "mine-consolidate"
IMPLEMENT = "mine-implement"
REVISE = "mine-revise"

MAX_PROPOSALS = 5
"""Most proposals one reading returns."""
LLM_TYPES = ("boolean", "category", "set")

Value = bool | float | str | list[str] | list[float] | dict[str, float]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Example(_Strict):
    file: str
    """The trajectory file as the request names it, `<dataset>/<id>.md`."""
    steps: str
    """The steps the value rests on, such as `#12` or `#12-#15`."""
    value: Value

    @property
    def key(self) -> str:
        return self.file.removesuffix(".md")


class Definition(_Strict):
    spec: FeatureSpec
    reads: Literal["fields", "text"]
    """`fields` when structured fields answer the question and code computes it, `text` when a model reads it."""
    guidance: str = ""

    @model_validator(mode="after")
    def _llm_shape(self) -> Definition:
        if self.reads == "text" and self.spec.type not in LLM_TYPES:
            raise ValueError(f"{self.spec.name}: a feature with reads=text is boolean, category or set")
        return self


class Proposal(Definition):
    examples: list[Example] = Field(min_length=1)

    @model_validator(mode="after")
    def _example_values(self) -> Proposal:
        adapter: TypeAdapter[Any] = TypeAdapter(value_type(self.spec))
        for example in self.examples:
            try:
                adapter.validate_python(example.value)
            except ValidationError as error:
                raise ValueError(f"{self.spec.name}: the example value in {example.file} does not fit the "
                                 f"feature type: {error}") from error
        return self


class TrajectoryFile(_Strict):
    file: str
    sha256: str
    n_chunks: int


class ProposeRequest(_Strict):
    files: list[TrajectoryFile]
    model: str


class Proposals(_Strict):
    proposals: list[Proposal] = Field(max_length=MAX_PROPOSALS)


class ConsolidateRequest(_Strict):
    proposals_file: str
    proposals_sha256: str
    features_file: str
    features_sha256: str
    model: str


class ImplementRequest(_Strict):
    candidate: Proposal
    example_files: list[str]
    model: str


class Implementation(_Strict):
    source: str = Field(min_length=1)
    """The whole text of the operator file."""


class Case(_Strict):
    file: str
    first: Any = None
    first_evidence: str | None = None
    second: Any = None
    second_evidence: str | None = None
    expected: Any = None
    steps: str | None = None


class ReviseRequest(_Strict):
    definition: Definition
    reason: Literal["noise", "example", "labels"]
    cases: list[Case] = Field(default_factory=list)
    label_counts: dict[str, int] = Field(default_factory=dict)
    model: str


class Revision(_Strict):
    action: Literal["rewrite", "split", "labels", "drop"]
    definitions: list[Definition] = Field(default_factory=list)
    reason: str = ""

    @model_validator(mode="after")
    def _shape(self) -> Revision:
        count = len(self.definitions)
        if self.action in ("rewrite", "labels") and count != 1:
            raise ValueError(f"{self.action} returns exactly one definition")
        if self.action == "split" and count < 2:
            raise ValueError("split returns at least two definitions")
        if self.action == "drop" and (count or not self.reason.strip()):
            raise ValueError("drop returns no definition and a reason")
        if self.action == "labels" and not self.definitions[0].spec.labels:
            raise ValueError("labels returns a definition whose spec has labels")
        if any(d.reads != "text" for d in self.definitions):
            raise ValueError("revised definitions have reads=text")
        return self


def candidates_model(limit: int) -> type[BaseModel]:
    return create_model("Candidates", __base__=_Strict,
                        candidates=(Annotated[list[Proposal], Field(max_length=limit)], ...))


def functions(project: Project) -> dict[str, AiFunction[Any, Any]]:
    """The aifn functions of a mining round, declared the same way by the driver and by every worker."""
    engine = project.config.engine
    mining = project.config.mining
    model = Model(name=mining_model(project), provider=engine.provider, max_tokens=engine.max_tokens,
                  reasoning_effort=engine.reasoning_effort)
    shapes: dict[str, tuple[type[BaseModel], type[BaseModel]]] = {
        PROPOSE: (ProposeRequest, Proposals),
        CONSOLIDATE: (ConsolidateRequest, candidates_model(mining.candidates)),
        IMPLEMENT: (ImplementRequest, Implementation),
        REVISE: (ReviseRequest, Revision),
    }
    return {
        name: AiFunction(name=name, request=request, returns=returns, model=model, timeout_s=mining.timeout_s,
                         attempts=project.config.extract.attempts)
        for name, (request, returns) in shapes.items()
    }


def mining_model(project: Project) -> str:
    return project.config.mining.model or project.config.engine.model


_FILES = """\
Each trajectory file is one LLM trajectory, such as a conversation or an agent run.
It opens with a description of the dataset, when one is given, and a JSON block of metadata; read both first.
Every step starts with a heading `### #<n> <role> · <kind>[ · <name>][ · ERROR]`.
A file has chunks, each starting with a line `<!-- chunk <k> -->`; read a long file in parts.
"""

_RULES = """\
## Rules for a feature

- A feature is one value per trajectory, and it answers one question that can be read off the trajectory directly:
  by finding one statement, or by counting structured fields.
- Its value is a yes or no, one label, or a set of labels copied from the text.
  A feature with `reads: fields` may also be a number, a list of numbers, or a map from label to number.
- A question that needs judgement across the whole trajectory is not a feature: whether something was handled well,
  why something happened, whether evidence is sufficient, how severe or how frustrated something is.
- The description is a literal question about what is written or done, such as "The user writes that ..." or
  "The agent calls ... again after ...".
- The description says what does not count wherever the boundary is unclear, and never asks whether a statement
  is correct.
- The definition holds for the whole dataset: it names no file, topic or person of one trajectory.
  When such names matter, a set feature copies them.
- The feature takes different values on different trajectories of this dataset.
- A name is snake_case and is a noun, such as `tool_calls`, or a noun followed by a verb, such as `user_corrects`.
  It never starts with a verb, has at most 48 characters, holds no double underscore, and differs from every
  existing feature name.
- Give `labels`, a map from label to its meaning, when the possible labels are known.

## `reads`

- `fields`: the step headings and the metadata answer the question: roles, kinds, tool names, error marks, step
  order, counts and timestamps. Code computes such a feature from the structured trajectory.
- `text`: answering needs reading what a user, a model or a tool wrote. A model answers such a feature, and its
  type is boolean, category or set. Put the judging criteria into `guidance`.
"""


def _render_features(groups: list[Group]) -> str:
    lines = []
    for group in groups:
        for feature in group.features:
            labels = f" Labels: {', '.join(feature.labels)}." if feature.labels else ""
            lines.append(f"- `{feature.name}` ({feature.type}): {feature.description.strip()}{labels}")
    return "\n".join(lines) or "No feature is extracted yet."


def propose_instruction(groups: list[Group]) -> str:
    return f"""\
# Propose trajectory features

`input.json` names one or two Markdown files in `files`, relative to the working directory.
{_FILES}Read every named file whole, and do not read any other file.

The features listed at the end are already extracted from every trajectory of this dataset.
Your task is to find what they do not express.

- With one file: state what this trajectory shows that no listed feature records.
- With two files: the listed features give the two trajectories the same or nearly the same values.
  State how the two trajectories differ.

Return at most {MAX_PROPOSALS} proposals, the most notable first, and an empty list when you find none.

{_RULES}
## Examples

Every proposal carries `examples`, one for each named file in which you determined the value.
An example holds `file` exactly as `input.json` names it, `steps` citing the steps the value rests on as `#12` or
`#12-#15`, and `value`, the value of the feature in that file.
With two files, give an example for both, and the two values differ.

## Features already extracted

{_render_features(groups)}
"""


def consolidate_instruction(limit: int) -> str:
    return f"""\
# Consolidate feature proposals

`input.json` names two JSON files in the working directory.
`proposals_file` holds proposals from independent readings of trajectories; each has `spec`, `reads`, `guidance`
and `examples`.
`features_file` holds the features already extracted.
Read both files whole, and do not read any other file.

Return candidates:

- Merge the proposals that ask the same question into one candidate, with the clearest definition among them.
- Drop a proposal whose question an already extracted feature asks, or answers through one comparison, such as
  a count being zero.
- Drop a proposal that breaks the rules below.
- Keep at most {limit} candidates, those that most readings proposed first.

A candidate keeps the examples of every proposal merged into it: copy `file` and `steps` exactly, and keep an
example only when its `value` fits the type of the candidate.
Never write an example that no proposal holds.
Candidate names differ from each other and from the names in `features_file`.

{_RULES}"""


IMPLEMENT_INSTRUCTION = """\
# Write a code operator

`input.json` holds `candidate`, a feature definition with examples, and `example_files`, trajectories as JSON in
the working directory.
The working directory also holds `schema.py`, which defines `Trajectory` and `Step`, `base.py`, which defines
`FeatureSpec` and `Operator`, and `example_operator.py`, an operator of the built-in library.
Read these files, and do not read any other file.

Return `source`, the whole text of one Python file:

- It defines `OPERATOR = Operator(kind="code", description=..., outputs=outputs, compute=compute)`.
- `outputs(params)` returns exactly one `FeatureSpec`, equal to `candidate.spec` in every field.
- `compute(trajectory, params)` returns a dict from the feature name to its value, or to None when the feature
  does not apply to the trajectory.
- The value has the feature's type: bool for boolean, float for scalar, str for category, a list of distinct str
  for set, a list of float for vector, a dict from str to float for map.
- It reads only structured fields: `step.index`, `step.role`, `step.kind`, `step.name`, `step.is_error`,
  `step.timestamp`, the JSON arguments of tool calls in `step.content`, and `trajectory.metadata`.
- It never uses regular expressions or keyword matching to interpret text a user or a model wrote.
- It imports only the standard library, `traj_analyzer.operators.base` and `traj_analyzer.schema`.
- It starts with `from __future__ import annotations` and has no module docstring.

For every example of the candidate, the value computed from the example file equals the example's `value`.
Check this by reading the example files before you answer.
"""

REVISE_INSTRUCTION = f"""\
# Revise a feature definition

`input.json` holds `definition`, `reason`, `cases` and `label_counts`.
A case names a trajectory file relative to the working directory.
{_FILES}Read the cited steps of the cases, and do not read files no case names.

What `reason` asks for:

- `noise`: two models extracted this feature from the same trajectories, and the cases are those where `first`
  and `second` differ, each with the evidence the model gave.
  Find what in the definition lets two readers answer differently.
  Return `rewrite` with one definition that states the boundary and what does not count; or `split` with two or
  more simpler definitions, each answerable by finding one statement; or `drop` with the reason when no literal
  question remains.
- `example`: the feature was proposed with the value `expected` at `steps` of a file, and extraction gave `first`.
  Return `rewrite` with one definition that states what the example shows, or `drop` with the reason.
- `labels`: the feature has too many free labels, and `label_counts` gives each label with its frequency.
  Return `labels` with one definition whose `spec.labels` maps every label of a closed set to its meaning; merge
  labels that differ only in wording.

Every returned definition has `reads: text`.
`rewrite` and `labels` keep the feature name; `split` uses new names that differ from each other.

{_RULES}"""
