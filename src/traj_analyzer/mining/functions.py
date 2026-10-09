from __future__ import annotations

from typing import Annotated, Any, Literal

from aifn import AiFunction, Model
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, create_model, model_validator

from traj_analyzer.features.spec import value_type
from traj_analyzer.operators.base import Edge, FeatureSpec
from traj_analyzer.project import Project

SURVEY = "mine-survey"
HYPOTHESIZE = "mine-hypothesize"
EXPLORE = "mine-explore"
DEFINE = "mine-define"
REVISE = "mine-revise"
REVIEW = "mine-review"
TAIL_CHECK = "mine-tail-check"

MIN_SEEN = 2
"""Least number of trajectories a finding needs on each side: with the behaviour and without it."""

SCENARIO = "scenario.md"
REVIEWS = "reviews.json"
HYPOTHESES = "hypotheses.json"
FINDINGS = "findings.json"
CANDIDATES = "candidates.json"
CHECKS = "checks.json"

LLM_TYPES = ("boolean", "category", "set")
NAME = r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$"

Value = bool | float | str | list[str] | list[float] | dict[str, float]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Example(_Strict):
    file: str
    """The trajectory file below `trajectories/`, `<dataset>/<id>.md`."""
    steps: str
    """The steps the value rests on, such as `#12` or `#12-#15`."""
    value: Value
    """The value of the feature in the file; for an event feature, true where the rule finds an event at `steps`
    and false where its precondition holds and the expectation is met."""

    @property
    def key(self) -> str:
        return self.file.removeprefix("trajectories/").removesuffix(".md")


class Definition(_Strict):
    spec: FeatureSpec
    reads: Literal["fields", "text"]
    """`fields` when the tables answer the question and code computes it, `text` when a model reads it."""
    guidance: str = ""

    @model_validator(mode="after")
    def _llm_shape(self) -> Definition:
        if self.reads == "text" and self.spec.type not in LLM_TYPES:
            raise ValueError(f"{self.spec.name}: a feature with reads=text is boolean, category or set")
        return self


class Candidate(Definition):
    hypothesis: str
    """The hypothesis the feature measures."""
    examples: list[Example] = Field(min_length=1)
    source: str | None = None
    """For `reads: fields`, the whole text of the operator file that computes the feature."""

    @model_validator(mode="after")
    def _shape(self) -> Candidate:
        adapter: TypeAdapter[Any] = TypeAdapter(bool if self.spec.type == "event" else value_type(self.spec))
        for example in self.examples:
            try:
                adapter.validate_python(example.value)
            except ValidationError as error:
                raise ValueError(f"{self.spec.name}: the example value in {example.file} does not fit the "
                                 f"feature type: {error}") from error
        if (self.reads == "fields") != bool(self.source and self.source.strip()):
            raise ValueError(f"{self.spec.name}: `source` holds the operator file exactly when reads=fields")
        if self.reads == "fields" and self.spec.scope:
            raise ValueError(f"{self.spec.name}: an operator returns None where its question cannot arise, so "
                             f"`spec.scope` stays empty for reads=fields")
        return self


class StepRequest(_Strict):
    """What a function of the sequence is asked: the round, and the digest of every file it builds on."""

    round: int
    files: dict[str, str]
    """Workspace file to its sha256; a changed file makes a new call."""
    model: str


class Scenario(_Strict):
    notes: str = Field(min_length=200)
    """The whole text of `scenario.md`."""


class Hypothesis(_Strict):
    name: str = Field(pattern=NAME, max_length=48)
    kind: Literal["measure", "expectation"]
    """`measure`: a quantity worth a panel, whose change between batches tells something. `expectation`: what
    should follow a given situation, so that a position where it does not is an event."""
    behaviour: str = Field(min_length=1)
    """What some trajectories do and others do not; for an expectation, the situation and what should follow."""
    reason: str = Field(min_length=1)
    """Why it matters in this scenario: what a change of the measure, or a broken expectation, means for how the
    trajectory goes."""
    signal: str = Field(min_length=1)
    """What in a trajectory or its tables shows the behaviour."""
    scope: str = Field(min_length=1)
    """The trajectories in which the behaviour can occur at all, such as those with tool calls."""
    observed: list[Example] = Field(min_length=2)
    """Trajectories of the scope the hypothesis comes from, with `value` true where the behaviour shows and false
    where it could have shown and does not."""

    @model_validator(mode="after")
    def _both_sides(self) -> Hypothesis:
        if {example.value for example in self.observed} != {True, False}:
            raise ValueError(f"{self.name}: `observed` names a trajectory that shows the behaviour, with value "
                             f"true, and one that does not, with value false")
        return self


class Finding(_Strict):
    hypothesis: str = Field(pattern=NAME, max_length=48)
    status: Literal["supported", "revised", "abandoned"]
    statement: str = Field(min_length=1)
    """What the trajectories show: the behaviour as found, or why the hypothesis does not hold."""
    shows: int = Field(ge=0)
    """Trajectories of the scope found to show the behaviour."""
    lacks: int = Field(ge=0)
    """Trajectories of the scope found not to show it, although it could occur there."""
    spread: str = ""
    """How the trajectories that show the behaviour spread over batches, strata, datasets and other groups."""
    consequence: str = ""
    """What follows where the behaviour shows: what the user does next, how the trajectory ends, what else
    differs."""
    examples: list[Example] = Field(default_factory=list)

    @model_validator(mode="after")
    def _shape(self) -> Finding:
        if self.status == "abandoned":
            return self
        if not (self.examples and self.spread.strip() and self.consequence.strip()):
            raise ValueError(f"{self.hypothesis}: a {self.status} finding has examples and states its spread and "
                             f"its consequence")
        if min(self.shows, self.lacks) < MIN_SEEN:
            raise ValueError(
                f"{self.hypothesis}: a {self.status} finding rests on at least {MIN_SEEN} trajectories that show "
                f"the behaviour and {MIN_SEEN} of its scope that do not; with shows={self.shows} and "
                f"lacks={self.lacks}, look at more trajectories or return the finding as abandoned")
        return self


class Findings(_Strict):
    findings: list[Finding]


class Review(_Strict):
    candidate: str
    decision: Literal["keep", "reject"]
    reason: str = Field(min_length=1)
    """What was checked and what it showed."""


class Reviews(_Strict):
    reviews: list[Review]


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


class TailCase(_Strict):
    file: str
    """The trajectory file, `cases/<dataset>/<id>.md`."""
    value: Any
    """The value of the feature in the file; for an event feature, the number of its unrecovered events."""
    steps: list[int] = Field(default_factory=list)
    """The steps the value rests on."""
    note: str = ""


class TailRequest(_Strict):
    definition: Definition
    cases: list[TailCase]
    model: str


class TailVerdict(_Strict):
    file: str
    worth: bool
    """Whether a person responsible for the agent should look at this trajectory because of what the feature
    found there."""
    why: str = Field(min_length=1)
    edge: Edge | None = None
    """When worth a look: the interaction on which the trajectory went wrong."""


class TailReview(_Strict):
    verdicts: list[TailVerdict]


def _listing(name: str, field: str, item: type[BaseModel], limit: int) -> type[BaseModel]:
    return create_model(name, __base__=_Strict, **{field: (Annotated[list[item], Field(max_length=limit)], ...)})  # type: ignore[call-overload, valid-type]


def hypotheses_model(limit: int) -> type[BaseModel]:
    return _listing("Hypotheses", "hypotheses", Hypothesis, limit)


def candidates_model(limit: int) -> type[BaseModel]:
    return _listing("Candidates", "candidates", Candidate, limit)


def functions(project: Project) -> dict[str, AiFunction[Any, Any]]:
    """The aifn functions of a mining round, declared the same way by the driver and by every worker."""
    engine = project.config.engine
    mining = project.config.mining
    model = Model(name=mining_model(project), provider=engine.provider, max_tokens=engine.max_tokens,
                  reasoning_effort=engine.reasoning_effort)
    shapes: dict[str, tuple[type[BaseModel], type[BaseModel]]] = {
        SURVEY: (StepRequest, Scenario),
        HYPOTHESIZE: (StepRequest, hypotheses_model(mining.hypotheses)),
        EXPLORE: (StepRequest, Findings),
        DEFINE: (StepRequest, candidates_model(mining.candidates)),
        REVIEW: (StepRequest, Reviews),
        REVISE: (ReviseRequest, Revision),
        TAIL_CHECK: (TailRequest, TailReview),
    }
    return {
        name: AiFunction(name=name, request=request, returns=returns, model=model, timeout_s=mining.timeout_s,
                         attempts=project.config.extract.attempts)
        for name, (request, returns) in shapes.items()
    }


def mining_model(project: Project) -> str:
    return project.config.mining.model or project.config.engine.model


_WORKSPACE = f"""\
## The workspace

The working directory is the workspace of one mining round.
Several agents work in it one after another, and each reads what the ones before left.
`input.json` names the files you build on in `files`; read `README.md` first, it describes every file.

- `trajectories/<dataset>/<id>.md`: one LLM trajectory each, such as a conversation or an agent run.
  A file opens with a description of the dataset, when one is given, and a JSON block of metadata.
  Every step starts with a heading `### #<n> <role> · <kind>[ · <name>][ · ERROR]`.
  A file has chunks, each starting with a line `<!-- chunk <k> -->`; read a long file in parts, and search it
  with grep when you look for one thing.
- `tables/*.parquet`: the trajectories parsed into tables, one row per step, per tool action, per relation
  between actions, per SQL query, per event, and so on; `README.md` lists every table with its columns and gives
  queries that answer common questions. The tables are the fastest way to see a thing across all trajectories.
- `features.csv`: one row per trajectory you may read, with its `key`, `<dataset>/<id>`, its `batch` and
  `stratum`, and the value of every feature already extracted. Read only trajectories this file lists.
- `features.md`: the definition of every feature already extracted.
- `{SCENARIO}`, `{HYPOTHESES}`, `{FINDINGS}`, `{CANDIDATES}`, `{REVIEWS}`, `{CHECKS}`: what the agents of this
  round and the checks have produced so far.
- `notes/`: yours to write.

You may run shell commands in the workspace; `README.md` gives the commands.
Write only below `notes/` and `operators/`.

## What the features are for

The features go on a dashboard, one panel each, where people watch how batches of trajectories differ: a new
model, a new release, a new day. The ones that hold up also raise alerts: a feature whose extreme values mark
trajectories that are worth a person's look, and an event rule whose events mark the position where something
went wrong, so that a person can open the trajectory at that step.
A feature is good when a change of it between batches means something, and when the trajectories at its tail
are worth looking at. A feature of style, which varies but tells nothing about how the trajectory goes, is not.

## Work from the data

The trajectories are the source, and every agent of the round works from them.
A file another agent wrote is a set of claims made from the part of the data that agent looked at: it tells you
where to look, and it does not replace looking.

- Query the tables to see a thing across all trajectories, and read trajectory files to see how one goes.
- Before you build on a claim of another agent, check it against the data, and when the data says otherwise,
  follow the data and record the correction.
- State nothing you did not see. Every statement you return rests on trajectories you queried or read, and names
  them.
- Write `notes/<name of your function>.md`, such as `notes/mine-survey.md`, as you work: the queries you ran
  with what they showed, the trajectories you read with what each showed, and every claim of an earlier file
  you corrected.
"""

_RULES = """\
## Rules for a feature

- A feature is one value per trajectory, or one list of events per trajectory, and it answers one question that
  can be read off the trajectory or its tables directly: by finding one statement, or by counting and relating
  rows of the tables.
- A measure is a number, a yes or no, one label, or a set of labels copied from the text.
  An event feature lists the positions where a rule finds an expectation broken, each with the step, the objects
  concerned, and whether a later action made up for it.
- A question that needs judgement across the whole trajectory is not a feature: whether something was handled well,
  why something happened, whether evidence is sufficient, how severe or how frustrated something is.
- The description is a literal question about what is written or done, such as "The user writes that ..." or
  "The agent calls ... again after ...". For an event rule it states the situation, what should follow, and what
  counts as recovery.
- The description says what does not count wherever the boundary is unclear, and never asks whether a statement
  is correct.
- The definition holds for the whole dataset: it names no file, topic, person, project or fixed wording of one
  group of trajectories. When such names matter, a set feature copies them.
- A feature has a value only where its question can arise. A feature about tool calls has none in a trajectory
  without tool calls, and a feature about what follows a failure has none where nothing failed. "No" means the
  behaviour could have shown and did not.
  An operator returns None outside that scope.
  A feature a model reads gives the scope in `spec.scope`, a condition over columns of `features.csv` such as
  `n_tool_calls > 0` or `n_user_turns >= 2 and tool_error_rate > 0`, in the syntax of a pandas query; the model
  is then asked only about the trajectories that meet it. Leave `spec.scope` out only when the question arises
  in every trajectory. When no column expresses the scope, return the code feature that does as a candidate of
  its own, and leave the feature that needs it to a later round.

## Code first

Code computes a feature on every trajectory at once and always gives the same answer, and a model reads one
trajectory at a time, at a cost, and with some disagreement.
Whatever the tables can answer is a `reads: fields` feature: roles, kinds, tool names, the actions of the tool
calls with their objects, effects and outcomes, the relations between actions, order, counts, lengths,
repetition, and what follows what.
A model reads only what is left, the meaning of what someone wrote, and only inside a scope that code computes.
- The feature takes different values on different trajectories across the dataset, and its values do not follow
  from which project, template or harness a trajectory comes from.
- A name is snake_case and is a noun, such as `tool_calls`, or a noun followed by a verb, such as `user_corrects`.
  It never starts with a verb, has at most 48 characters, holds no double underscore, and differs from every
  existing feature name.
- Give `labels`, a map from label to its meaning, when the possible labels are known.

## `reads`

- `fields`: the tables answer the question. Code computes such a feature from the tables of the trajectory.
- `text`: answering needs reading what a user, a model or a tool wrote. A model answers such a feature, and its
  type is boolean, category or set. Put the judging criteria into `guidance`.
"""


def survey_instruction() -> str:
    return f"""\
# Describe the scenario of a trajectory dataset

{_WORKSPACE}
## Your task

Work out what these trajectories are, from the data alone, and return `notes`, the whole text of `{SCENARIO}`.
The agents after you propose features for this scenario, and they rely on your notes to tell behaviour that
matters from text that only looks like behaviour.

`{SCENARIO}` may already hold the notes of an earlier round; keep what still holds, correct what does not, and
add what you learn.

How to work:

- Begin with the whole dataset. Run the queries of `README.md` over the tables: the first user message of every
  trajectory, the tools with their arguments and failures, the actions and their outcomes, the system messages,
  the openings that repeat across trajectories, how trajectories end, which values the metadata takes, and how
  the batches and strata of `features.csv` differ.
  Tool definitions, skill definitions and system prompts show as messages that repeat; find them this way.
- Then read whole trajectories. `start.txt` lists some that differ strongly from each other under the extracted
  features; read them, and read more wherever the queries show a group you have not seen: another value of a
  metadata field, very short and very long trajectories, trajectories with errors or events.
- Give numbers from the queries, such as how many trajectories call a tool or open with a given text.

`notes` is Markdown with these sections:

- `## Participants`: who acts in a trajectory, and what each does.
- `## Task`: what kinds of task the trajectories carry, and how a task is given.
- `## Course`: how a trajectory typically starts, proceeds and ends, and the main ways this varies.
- `## Tools`: the tools called, what each is for, and how a failed call shows.
- `## Tables`: which tables hold what for this scenario, and which columns take which values here.
- `## Good and bad`: what in the data itself shows that a trajectory went well or badly: reactions of the user,
  where the user corrected or abandoned the agent and what came before, a task left open, repeated failures,
  events already found. Name the positions, as trajectory and step.
- `## Batches and strata`: how the batches of `features.csv` differ, and what the strata hold fixed.
- `## Templates and harness text`: text that scripts, templates, harnesses or commands produce and that repeats
  across trajectories, such as generated opening messages, injected context or notices of the rendering like
  cut-off marks. Say how to recognise each, and how many of the trajectories you read carry it.
  A difference inside such text is no behaviour.
- `## Questions`: the questions about behaviour that are worth asking of this scenario: what quantities would
  show on a panel that something changed, and what expectations the agent breaks before things go wrong.

State only what you saw, and name the trajectories a statement rests on.
"""


def hypothesize_instruction(limit: int) -> str:
    return f"""\
# Propose hypotheses about behaviour

{_WORKSPACE}
## Your task

Propose which behaviours matter in this scenario, of two kinds, and return at most {limit} hypotheses, the most
important first.

- `measure`: a quantity whose change between batches would tell a person that the agent behaves differently,
  and whose extreme values mark trajectories worth a look.
- `expectation`: a situation and what should follow it, such that a position where it does not follow is an
  event a person would want to open the trajectory at. Give the situation, the expected follow-up, and what
  counts as recovery.

Look at the trajectories before you propose anything.

- Read `{SCENARIO}` and `features.md`, then go to the data yourself: query the tables for what the scenario
  describes, and read whole trajectories of different kinds, among them ones `{SCENARIO}` does not name.
- Start from where things went wrong: the positions where the user corrected, rejected or abandoned the agent,
  the events already found, the failures that ran on. Look at what the agent did before, and ask what
  expectation it broke; that is an `expectation` hypothesis. Then ask what quantity would have shown the
  difference on a panel; that is a `measure` hypothesis.
- A hypothesis names one behaviour you saw. `behaviour` says what some trajectories do, `reason` why it matters
  here, and `signal` what in a trajectory or its tables shows it.
- `reason` is a claim the data can bear out: it names what should differ where the behaviour shows, such as the
  user correcting the agent afterwards, a task left open, fewer failed calls, or a batch where it is more common.
- `scope` names the trajectories in which the behaviour can occur at all, such as those with tool calls, those
  with a failed call, or those with more than one user message. A trajectory outside the scope tells nothing
  about the hypothesis.
- `observed` gives the trajectories the hypothesis comes from: at least one where you saw the behaviour and one
  of the same scope where it could have shown and does not, each with `file` as `<dataset>/<id>.md`, `steps`
  citing the steps as `#12` or `#12-#15`, and `value`, true where the behaviour shows and false where it does
  not.
- Cover the scenario: every participant and every kind of trajectory `{SCENARIO}` describes gets hypotheses,
  in proportion to how many trajectories are of that kind.
- Leave out what `features.md` already measures, what belongs to templates and harness text, and style that
  tells nothing about how the trajectory goes.
- A hypothesis holds for the scenario: it names no dataset, project or tool, and you expect the behaviour in
  trajectories of different origin. The agent after you tests each hypothesis on more trajectories.

`{CHECKS}` of an earlier round, when it exists in `previous/`, lists the features that were rejected and why;
do not propose them again in the same form.
"""


def explore_instruction() -> str:
    return f"""\
# Test hypotheses against the trajectories

{_WORKSPACE}
## Your task

`{HYPOTHESES}` holds hypotheses about behaviour in the scenario of `{SCENARIO}`.
Test every one against the trajectories and return a finding for each.
You may add findings for behaviours you come across that no hypothesis names.

For each hypothesis:

- Its `observed` trajectories are where it comes from; read them first, and correct the hypothesis when they do
  not show what it says.
- Then go beyond them: look for more trajectories that show the behaviour and for trajectories that clearly do
  not. Choose them across different batches, strata and values of the extracted features, so that the
  trajectories you compare differ in origin; `features.csv` and `tables/trajectories.parquet` let you pick them.
- When the signal is in the tables, query them over all trajectories and count. Express the scope and the signal
  as a query over `tables/`, or as an operator file run with the command of `README.md`, run it over the whole
  dataset, and read some of the trajectories on each side to check that the query finds what the hypothesis
  means. For an expectation, the query finds the positions where the situation holds and tells whether the
  follow-up came.
- Only when the signal lies in the meaning of what someone wrote, read trajectories: narrow to the scope with a
  query first, draw the trajectories to read from the result across different origins, and read enough of them
  to count both sides.
- Count inside the scope of the hypothesis: `shows` is the number of trajectories of the scope you found to show
  the behaviour, and `lacks` the number of the scope where it could occur and does not. A trajectory outside the
  scope counts on neither side. Say in `spread` whether the counts cover every trajectory of the scope or the
  ones you read.
- Write into `spread` how the trajectories that show the behaviour and those that lack it spread over batches,
  strata and other groups inside the scope. Judge this inside the scope. A scope that itself lies in one group
  is no objection; the objection is a behaviour whose two sides fall apart along a group, so that the group
  already tells the value.
- Test the reason of the hypothesis: compare the trajectories that show the behaviour with those of the scope
  that lack it, and look for what the reason says should differ: what the user does next, how the trajectory
  ends, what else differs. For an expectation, look at what follows the positions where it is broken. Write into
  `consequence` what you compared and what you found, and say plainly when nothing differs or when the data
  cannot tell.
- Sharpen the behaviour where the data asks for it: state the boundary, and what does not count.

`status` of a finding, by the same measure for every hypothesis:

- `supported`: at least {MIN_SEEN} trajectories show the behaviour and at least {MIN_SEEN} of its scope lack it,
  and inside the scope no group tells the two sides apart.
- `revised`: the same holds for another form of the behaviour than the hypothesis says; `statement` gives the
  form found, and the counts are those of that form.
- `abandoned`: the behaviour shows in fewer than {MIN_SEEN} trajectories, shows in all of its scope but fewer
  than {MIN_SEEN}, cannot be told from the trajectory, or inside its scope shows in one group and is absent from
  the others; `statement` says which. A behaviour seen once is abandoned however interesting it is.

A supported or revised finding carries `examples`: at least one trajectory that shows the behaviour and one of
the scope that does not, each with `file` as `<dataset>/<id>.md`, `steps` citing the steps as `#12` or
`#12-#15`, and `value`, true or false, or the label when the behaviour has kinds.

"""


def define_instruction(limit: int, operator_guide: str) -> str:
    return f"""\
# Define features from findings

{_WORKSPACE}
## Your task

`{FINDINGS}` holds what the trajectories showed for each hypothesis, and `notes/` how it was found.
Turn every supported or revised finding into one feature, or into several when the finding holds several
questions, and return at most {limit} candidates, the best supported first.
A finding of an expectation becomes an event feature; a finding of a measure becomes a scalar, boolean, category
or set feature.
An abandoned finding gives no candidate; return an empty list when no finding is supported or revised.
Leave out a finding whose question an extracted feature of `features.md` already asks.

Write each definition from the trajectories: read the examples of the finding, and query the tables for
further trajectories with and without the behaviour, so that the definition states the boundary the data shows.
Leave out a finding the data does not bear out, and say so in your notes.

A candidate holds `hypothesis`, the name of its finding; `spec`, `reads` and `guidance`; `examples`; and for
`reads: fields` the operator file in `source`.

{_RULES}
## Examples

Every candidate carries `examples` from the trajectories its finding names, each with `file` as
`<dataset>/<id>.md`, `steps`, and `value`, the value of the feature in that file; for an event feature, true
where the rule finds an event at `steps` and false where the situation holds and the follow-up came.
Give at least two examples with different values.
Check each value against the trajectory file before you return it.

## Operator files

{operator_guide}"""


def review_instruction() -> str:
    return f"""\
# Review candidate features

{_WORKSPACE}
## Your task

`{CANDIDATES}` holds feature definitions another agent wrote from `{FINDINGS}`.
You own the quality of the feature set: a feature you keep goes on the dashboard as a panel, is extracted from
every trajectory, and may raise alerts that send people to trajectories, so keep a candidate only when you have
tried to show that it is not worth that and failed.
Return one review per candidate, with `candidate` its name, `decision` keep or reject, and `reason` saying what
you checked and what it showed.

Test every candidate against the data; do not judge it from its wording.

- Evidence: read the examples, and find further trajectories with and without the behaviour by querying the
  tables. Reject when the definition gives another value than a reader of the trajectory would, or when the
  finding behind it rests on fewer trajectories than it states.
- Rule: for an event feature, run the operator file with the command of `README.md`, open some of the positions
  it reports, and check that the situation holds there, that the follow-up is missing, and that recovery is
  marked where a later action made up for it. Reject when the positions are not what the description says.
- Scope: a feature has a value where its question can arise and none elsewhere. That an operator returns no value
  outside the scope, or that `spec.scope` leaves trajectories out, is as it should be and is no ground to reject.
  Reject when the candidate gives "no" in trajectories where its question cannot arise, so that its values mostly
  tell which kind of trajectory it is. For an operator file, look at which trajectories get which value.
- Origin: look only at the trajectories with a value. Reject when among them the dataset, project, template,
  harness or tool of a trajectory already tells the value. Compare the values across the groups
  `features.csv` and `tables/trajectories.parquet` show.
- Redundancy: reject when an extracted feature of `features.csv`, or a simple comparison of one, already gives
  the same values. For an operator file, compare its values with those columns.
- Question: reject when answering needs judgement across the trajectory, or when two careful readers could
  answer differently.
- Use: `consequence` of the finding says whether anything else differs where the behaviour shows, and what
  follows a broken expectation. Reject a candidate whose finding found no such difference and whose behaviour
  tells nothing by itself about how a trajectory goes; a panel of it would show change that means nothing.

A rejection states the check that failed and the trajectories or numbers that show it; the next round reads it.
"""


def operator_guide(tables: str) -> str:
    return f"""\
`operators/` holds `schema.py`, which defines `Trajectory` and `Step`; `base.py`, which defines `FeatureSpec`,
`Evidenced`, `Event` and `Operator`; `tables.py`, which defines `Tables`; `rules.py`, which defines `Rule` and
`Context` for event features; `example_operator.py`, a measure operator of the built-in library;
`example_actions.py`, an operator that reads the actions table; and `example_rules.py`, the event rules of the
built-in library.
A step of a trajectory file corresponds to one `Step`: the heading gives `index`, `role`, `kind`, `name` and
`is_error`, and the text below it is `content`, cut when long.

The tables an operator may read, by declaring their names in `tables=(...)`:

{tables}
For a candidate with `reads: fields`, write the operator file to `operators/<name>.py`, try it, and return its
whole text in `source`:

- It defines `OPERATOR = Operator(kind="code", description=..., outputs=outputs, compute=compute, tables=(...))`.
- `outputs(params)` returns exactly one `FeatureSpec`, equal to the candidate's `spec` in every field.
- `compute(trajectory, tables, params)` returns a dict from the feature name to its value, or to None when the
  feature does not apply to the trajectory. `tables["actions"]` is a pandas frame of the trajectory's rows of
  that table, and `tables.records("actions")` the same rows as dicts.
- The value has the feature's type: bool for boolean, float for scalar, str for category, a list of distinct str
  for set, a list of float for vector, a dict from str to float for map, and for event a list of `Event` dicts,
  each with `step`, `objects`, `recovered`, `recovered_at` and `note`.
- A measure may be returned as `Evidenced(value, steps, note)` so that the steps it rests on go to the evidence
  table and a person can open the trajectory there; do so whenever the value comes from particular steps.
- An event feature is written as a `Rule` with `precondition`, `expectation` and `recovery` over a `Context`, as
  `example_rules.py` shows, and `compute` returns `events_of([RULE], tables)`.
- It reads only the tables and the structured fields: `step.index`, `step.role`, `step.kind`, `step.name`,
  `step.is_error`, `step.timestamp`, the JSON arguments of tool calls in `step.content`, and the keys of
  `trajectory.metadata` that the metadata block of the trajectory files shows.
- It never uses regular expressions or keyword matching to interpret text a user or a model wrote.
- It imports only the standard library, pandas, `traj_analyzer.operators.base`, `traj_analyzer.schema`,
  `traj_analyzer.layers.tables` and `traj_analyzer.events`.
- It starts with `from __future__ import annotations` and has no module docstring.

`README.md` gives the command that computes an operator file on every trajectory.
It prints how the values spread, whether the trajectories that differ from the most common value sit in one
group, how much of the variation the batches explain, and the value for each trajectory you name.
Run it for every operator file, compare the values with your examples, and correct the file or the examples.
A feature that takes one value almost everywhere, or varies inside one group only, will be rejected, so drop it
or change the question.
"""


REVISE_INSTRUCTION = f"""\
# Revise a feature definition

`input.json` holds `definition`, `reason`, `cases` and `label_counts`.
A case names a trajectory file in the working directory, `cases/<dataset>/<id>.md`.
Every step of a trajectory file starts with a heading `### #<n> <role> · <kind>[ · <name>][ · ERROR]`, and a file
has chunks, each starting with a line `<!-- chunk <k> -->`.
Read the cited steps of the cases; `{SCENARIO}` in the working directory describes the dataset.

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


TAIL_INSTRUCTION = f"""\
# Check the tail of a feature

`input.json` holds `definition`, a feature, and `cases`, the trajectories at its tail: those with the highest or
rarest values of a measure, or with the most unrecovered events of an event feature. Each case names a trajectory
file in the working directory, `cases/<dataset>/<id>.md`, its `value`, and `steps`, the steps the value rests on.
Every step of a trajectory file starts with a heading `### #<n> <role> · <kind>[ · <name>][ · ERROR]`, and a file
has chunks, each starting with a line `<!-- chunk <k> -->`.
`{SCENARIO}` in the working directory describes the dataset.

The feature is meant to raise alerts: its tail is shown to the people responsible for the agent as trajectories
to look at. Decide for every case whether it is worth their look because of what the feature found there.

For each case, open the file at `steps`, read what happens there and what follows, and return a verdict:

- `worth` is true when a person who opens the trajectory at those steps sees something going wrong that they
  would want to know about: the agent breaking an expectation, the user correcting or giving up, work that does
  not serve the request, damage, a failure left unrecovered.
- `worth` is false when what the feature found is style, a harmless difference, a template or harness effect, or
  something the agent recovered from without consequence.
- `why` says what you saw at the steps, in one or two sentences naming the steps.
- `edge`, when worth a look, is the interaction on which it went wrong: owner (with the person who gave the
  task), context, memory, tool, peer, local_env, external_env, grader or third_party.

Judge every case on its own; return exactly one verdict per case, with `file` as given.
"""
