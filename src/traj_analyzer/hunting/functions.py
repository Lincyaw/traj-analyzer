from __future__ import annotations

from typing import Annotated, Any, Literal

from aifn import AiFunction, Model
from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from traj_analyzer.hunting.spec import HuntSpec, hunt_model, hunt_names, load_hunt
from traj_analyzer.project import Project

PROPOSE = "hunt-propose"
VERIFY = "hunt-verify"
NEW = "new"
"""The facet a verdict names for a kind of behaviour no facet of `facets.json` covers."""

NEED = "need.md"
SCENARIO = "scenario.md"
FACETS = "facets.json"
PROBES = "probes.json"
LEDGER = "ledger.json"
FINDINGS = "findings.json"

NAME = r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Facet(_Strict):
    name: str = Field(pattern=NAME, max_length=48)
    definition: str = Field(min_length=1)
    """What a trajectory of this kind shows, as something a reader can point to in it."""
    excludes: str = ""
    """What looks alike and does not count."""


class Probe(_Strict):
    name: str = Field(pattern=NAME, max_length=48)
    facet: str
    """The facet whose trajectories the probe looks for."""
    kind: Literal["sql", "code"]
    text: str = Field(min_length=1)
    """The whole SQL query, or the whole operator file."""
    why: str = Field(min_length=1)
    """What the probe looks for, and what of the ledger and the cases made it take this form."""
    addresses: list[str] = Field(default_factory=list)
    """Case files of earlier rounds the probe was written or revised for: missed trajectories it catches, or wrong
    hits it leaves out."""


class Retirement(_Strict):
    probe: str
    reason: str = Field(min_length=1)


class Proposal(_Strict):
    facets: list[Facet] = Field(min_length=1)
    """Every facet of the need, as the next round reads them."""
    probes: list[Probe]
    """Every probe the next round runs; a probe left out is retired."""
    retired: list[Retirement] = Field(default_factory=list)
    """Why each probe of the last round that is left out goes."""

    @model_validator(mode="after")
    def _names(self) -> Proposal:
        facets = [facet.name for facet in self.facets]
        if len(set(facets)) != len(facets):
            raise ValueError("two facets share a name")
        if NEW in facets:
            raise ValueError(f"`{NEW}` is reserved for verdicts and names no facet")
        probes = [probe.name for probe in self.probes]
        if len(set(probes)) != len(probes):
            raise ValueError("two probes share a name")
        unknown = sorted({probe.facet for probe in self.probes} - set(facets))
        if unknown:
            raise ValueError(f"probes name facets that `facets` lacks: {unknown}")
        both = sorted({retirement.probe for retirement in self.retired} & set(probes))
        if both:
            raise ValueError(f"probes are both returned and retired: {both}")
        return self


def proposal_model(limit: int) -> type[Proposal]:
    return create_model("Proposal", __base__=Proposal,  # type: ignore[call-overload, no-any-return]
                        probes=(Annotated[list[Probe], Field(max_length=limit)], ...))


class ProposeRequest(_Strict):
    """What `hunt-propose` is asked: the round, and the digest of every file it builds on."""

    hunt: str
    round: int
    files: dict[str, str]
    """Workspace file to its sha256; a changed file makes a new call."""
    model: str


class Hit(_Strict):
    probe: str
    step: int
    note: str = ""


class Case(_Strict):
    file: str
    """The trajectory file, `rounds/<round>/cases/<dataset>/<id>.md`."""
    hits: list[Hit] = Field(default_factory=list)
    """Where probes found something in the trajectory; empty for a trajectory no probe hits."""


class VerifyRequest(_Strict):
    hunt: str
    round: int
    files: dict[str, str]
    """The digests of the need and the facets the verdicts are given against."""
    cases: list[Case]
    model: str


class CaseVerdict(_Strict):
    file: str
    found: bool
    """Whether the trajectory shows what the need describes."""
    facet: str | None = None
    """When found: the facet of `facets.json` it belongs to, or `new` with `new_facet`."""
    new_facet: Facet | None = None
    steps: list[int] = Field(default_factory=list)
    """When found: the steps where it shows."""
    why: str = Field(min_length=1)

    @model_validator(mode="after")
    def _shape(self) -> CaseVerdict:
        if not self.found:
            if self.facet is not None or self.new_facet is not None or self.steps:
                raise ValueError(f"{self.file}: a trajectory that does not show the need has no facet, new_facet or "
                                 f"steps")
            return self
        if self.facet is None or not self.steps:
            raise ValueError(f"{self.file}: a trajectory that shows the need names its facet and its steps")
        if (self.facet == NEW) != (self.new_facet is not None):
            raise ValueError(f"{self.file}: `new_facet` describes the facet exactly when `facet` is `{NEW}`")
        return self


class Verification(_Strict):
    verdicts: list[CaseVerdict]

    @model_validator(mode="after")
    def _files(self) -> Verification:
        files = [verdict.file for verdict in self.verdicts]
        if len(set(files)) != len(files):
            raise ValueError("two verdicts name the same file; give one verdict per case")
        return self


def function_name(step: str, hunt: str) -> str:
    return f"{step}-{hunt}"


def functions(project: Project) -> dict[str, AiFunction[Any, Any]]:
    """The aifn functions of every hunt of the project, declared the same way by the driver and by every worker."""
    found: dict[str, AiFunction[Any, Any]] = {}
    for hunt in hunt_names(project):
        found.update(hunt_functions(project, hunt, load_hunt(project, hunt)))
    return found


def hunt_functions(project: Project, hunt: str, spec: HuntSpec) -> dict[str, AiFunction[Any, Any]]:
    engine = project.config.engine
    model = Model(name=hunt_model(project, spec), provider=engine.provider, max_tokens=engine.max_tokens,
                  reasoning_effort=engine.reasoning_effort)
    shapes: dict[str, tuple[type[BaseModel], type[BaseModel]]] = {
        function_name(PROPOSE, hunt): (ProposeRequest, proposal_model(spec.probes_max)),
        function_name(VERIFY, hunt): (VerifyRequest, Verification),
    }
    return {
        name: AiFunction(name=name, request=request, returns=returns, model=model,
                         timeout_s=project.config.mining.timeout_s, attempts=project.config.extract.attempts)
        for name, (request, returns) in shapes.items()
    }


_WORKSPACE = f"""\
## The workspace

The working directory is the workspace of one hunt: a search for the trajectories a person asks for.
Rounds follow one another in it; in each, an agent proposes probes, the probes run over every trajectory, a
sample of the trajectories is read, and the verdicts go into the ledger.
`input.json` names the files you build on in `files`; read `README.md` first, it describes every file and table.

- `{NEED}`: what the person is looking for.
- `trajectories/<dataset>/<id>.md`: every trajectory of the hunt. A file opens with a description of the dataset,
  when one is given, and a JSON block of metadata. Every step starts with a heading
  `### #<n> <role> · <kind>[ · <name>][ · ERROR]`, and long steps are cut; `tables/contents.parquet` holds the
  whole text of every message and tool call. A file has chunks, each starting with a line `<!-- chunk <k> -->`;
  read a long file in parts, and search it with grep when you look for one thing.
- `tables/*.parquet`: the trajectories parsed into tables: steps, tool actions, relations between actions, turns,
  events and more, and `contents` with the whole text of every message and tool call.
- `{SCENARIO}`, when it exists: what the trajectories are, as a mining round described them.
- `{FACETS}`: the kinds of behaviour the need covers, each with what does not count.
- `{PROBES}` and `probes/`: the probes of the last round, with their status, and their text.
- `{LEDGER}`: per probe its hits, how many of them were read and confirmed, its precision, and whether it should be
  revised; per facet the confirmed trajectories; per round what was read and found; and how much of the
  trajectories has been covered.
- `{FINDINGS}`: every trajectory confirmed to show the need, with its facet, steps and the verdict's reason.
- `rounds/<round>/verdicts.json` and `rounds/<round>/cases/`: every trajectory read in a round, with where the
  probes hit it, the verdict and why, and the file that was read, whole and uncut.
- `notes/`: yours to write.

You may run shell commands in the workspace; `README.md` gives the commands.
Write only below `notes/` and `operators/`.
"""


def propose_instruction(spec: HuntSpec, round_: int) -> str:
    return f"""\
# Propose probes that find what a person asks for

{_WORKSPACE}
## Your task

This is round {round_}. Return `facets`, the kinds of behaviour the need covers, and `probes`, the mechanical
searches the next round runs over every trajectory, at most {spec.probes_max} of them.
The aim is to find every trajectory that shows what `{NEED}` describes. The probes do not decide anything: every
trajectory they hit can be read, and a sample of the trajectories they miss is read too. A probe is good when
many of its hits are real and when, together, the probes miss few real ones.

### Facets

- In round 1, split the need into facets: the distinct ways a trajectory can show it. A facet is something a
  reader can point to at a step: "the agent edits a test so that it passes", not "the agent cheats".
- Say in `excludes` what looks alike and does not count, wherever the boundary is unclear.
- Later rounds keep the facets and sharpen them from the verdicts; a verdict can report a kind of behaviour no
  facet covers, which is already in `{FACETS}` when you start, and needs probes.

### Probes

A probe is one of two kinds, and either one finds positions: a trajectory and a step.

- `sql`: one DuckDB `SELECT` over the tables, written with paths as `'tables/<name>.parquet'`, returning exactly
  the columns `key`, `step` and `note`, one row per position: the trajectory key, the step number, and a few words
  on what was found there. Return the whole query in `text`.
- `code`: an operator file, for a search the tables cannot express in SQL, returned whole in `text`. It defines
  `OPERATOR = Operator(kind="code", description=..., outputs=outputs, compute=compute, tables=(...))`, where
  `outputs(params)` returns exactly one `FeatureSpec` of type `event` named like the probe, with
  `event=EventMark(edge=...)`, and `compute(trajectory, tables, params)` returns `{{<name>: [<event>, ...]}}`, each
  event a dict with `step`, `objects`, `recovered`, `recovered_at` and `note`, or None where nothing can be found.
  `operators/` holds `schema.py`, `base.py`, `tables.py` and `rules.py`, copies of the modules of `traj_analyzer`
  it imports from, `example_rules.py`, the event rules of the built-in library, and `example_operator.py`, the
  operator that returns their events; `step.arguments`, when set, holds the JSON arguments of a tool call.

A probe may match words and patterns in what a user, a model or a tool wrote, such as `regexp_matches` over
`tables/contents.parquet`, because every hit is checked by reading. It may also follow the structure: tool
actions, their objects and outcomes, what follows what, how a turn ends.
Each probe belongs to one facet. Every facet gets at least one probe, and the probes of a facet complement each
other: one for each form the behaviour takes in these trajectories.

Before you return a probe, run it: `README.md` gives the command for a query and for an operator file. Count its
hits, open some of them in the trajectory files, and check that they show its facet. A probe whose hits are
mostly something else wastes the reads of the next round; a probe that hits nothing finds nothing.

### What the ledger asks for

From round 2 on, `{LEDGER}` and the verdicts of `rounds/` say how the probes did.

- A probe the ledger marks `revise` has precision below {spec.min_precision} on the hits read so far. Read the
  verdicts of its hits that were not confirmed, find what they share, and tighten the probe so it leaves them out
  while keeping what was confirmed. Name those case files in `addresses`.
- A trajectory confirmed although no probe hit it is a miss. Read it at the steps of its verdict, and write a
  probe that catches it, or widen the probe of its facet; name it in `addresses`.
- A facet without any probe, or one whose probes hit nothing, needs a probe.
- A probe the ledger retired found nothing real in its reads; do not return it in the same form.
- Return every probe that holds up with exactly the text it had, so its counts carry on. A probe whose text
  changes counts as a new probe, and its hits are read again from the start.
- Leave a probe out to retire it, and give the reason in `retired`.

Write `notes/{PROPOSE}-{round_}.md` as you work: the queries you ran with what they showed, the trajectories you
read with what each showed, and why each probe has its form.
"""


VERIFY_INSTRUCTION = f"""\
# Read trajectories for what a person asks for

`input.json` holds `cases`: trajectory files of the working directory, `rounds/<round>/cases/<dataset>/<id>.md`,
each whole and uncut. For some of them `hits` gives the positions where a probe, a mechanical search, found
something; the others are trajectories no probe hits.
`{NEED}` says what the person looks for, and `{FACETS}` the kinds of it known so far, each with what does not count.
`{SCENARIO}`, when it exists, describes the trajectories.
Every step of a trajectory file starts with a heading `### #<n> <role> · <kind>[ · <name>][ · ERROR]`, and a file
has chunks, each starting with a line `<!-- chunk <k> -->`.

Decide for every case whether the trajectory shows what `{NEED}` describes, and return one verdict per case with
`file` as given.

- A case with hits: open the file at the steps of the hits first. A probe only searches, and its hit may be a false
  alarm; decide from what the trajectory shows there and around it. Then look through the rest of the trajectory
  for any other sign of the need.
- A case without hits: no probe found anything, and this is where the trajectories the probes miss are found. Read
  the request and the course of the trajectory, and look for the signs of every facet.
- `found` is true when a reader can point to steps where the trajectory shows what the need describes, and false
  otherwise. Something that only looks alike, as a facet's `excludes` says, is false.
- When found, `facet` names the facet of `{FACETS}` it belongs to, and `steps` the steps where it shows. When no
  facet fits, `facet` is `{NEW}` and `new_facet` describes the kind of behaviour: a name, a definition a reader can
  point to at a step, and what does not count.
- `why` says what you saw, in one to three sentences naming the steps. When the trajectory has hits and is not
  found, say why the hits are not what the need describes; the next round reads this to tighten the probe.

Judge every case on its own.
"""
