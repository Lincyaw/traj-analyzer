from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from traj_analyzer.adapters._blocks import PLACEHOLDERS
from traj_analyzer.schema import INJECTED, Kind, Role, Step, Trajectory

VERSIONS = {"ATIF-v1.8"}
"""The schema versions of the Agent Trajectory Interchange Format this adapter was read against."""
SOURCES: dict[str, Role] = {"system": "system", "user": "user", "agent": "assistant"}
CONVERSATION = "conversation"
"""The metadata key saying which conversation of its document a trajectory is, `main` or `subagent`."""
MAIN = "main"
SUBAGENT = "subagent"
PARENT = "parent_trajectory"
"""The metadata key of a subagent conversation, holding the identity of the conversation it is embedded in."""
HELD = "."
"""What separates a subagent conversation's own identity from the identity of the trajectory holding it: the
format requires an embedded `trajectory_id` to be unique among its siblings alone, so the identity of the
document it belongs to is what makes it unique in a project. A dot because `traj ingest` keeps one, so the
identity a project stores is the identity read here."""


class AtifSource(BaseModel):
    """One input file as what it gives: the ATIF document, the identity of the trajectory of that document, and
    what a source that holds the document otherwise knows beside it."""

    model_config = ConfigDict(extra="forbid")

    document: dict[str, Any]
    identity: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    recorded_calls: list[str] | None = None
    subagent_calls: dict[str, list[str]] = Field(default_factory=dict)
    """The recorded calls of each conversation the document embeds, by its own `trajectory_id`. A source that
    reports the calls of a run reports them per conversation, since each conversation is a trajectory with its
    own steps and its own calls."""


@dataclass
class _Parse:
    """State of one trajectory: its steps, the tool names seen so far keyed by tool call id, and whether the agent
    has answered, which ends the turns `single_prompt` counts as the person's own."""

    steps: list[Step] = field(default_factory=list)
    tool_names: dict[str, str] = field(default_factory=dict)
    answered: bool = False

    def add(self, role: Role, kind: Kind, content: str, timestamp: str | None, name: str | None = None,
            is_error: bool = False, arguments: str | None = None) -> None:
        self.steps.append(Step(index=len(self.steps), role=role, kind=kind, content=content, name=name,
                               is_error=is_error, timestamp=timestamp, arguments=arguments))


class AtifAdapter:
    """Trajectories in the Agent Trajectory Interchange Format, one file holding one document.

    An ATIF step is a turn, so the steps here are the message, the reasoning, the tool calls and the results of
    one turn. The trajectory is identified by the document's `trajectory_id`. The agent, the model, the final
    metrics and the producer's own `extra` become metadata, their scalar values alone, which is what the index row
    of `traj ingest` holds of a trajectory too. The metrics of the format carry a producer's own figures in their
    own `extra`, which is where a price goes when the format's `total_cost_usd` would name a currency the
    provider never stated, so those figures become metadata under the same `metrics_` names.

    A document also embeds the conversations the agent delegated, each a trajectory of the format with its own
    steps numbered from 1, and each becomes a trajectory here: a conversation nobody reads is a part of the run
    nobody oversees, and on the recorded corpus those conversations hold about a quarter of the turns. Its
    identity is the holding trajectory's identity, `#` and its own `trajectory_id`, since the format requires
    that id to be unique among its siblings alone. `conversation` says which conversation a trajectory is and
    `parent_trajectory` names the one holding it, so an operator, a hunt and a page can read either the run's own
    conversation or everything the run said. Everything a path or a source states about the run, such as the task
    and the trial, is the same for every conversation of it, which is what puts them in one batch.

    `path_metadata` is a regular expression searched in the path of the file; its named groups become metadata.
    `harness_key` names the metadata key that holds the harness, and `harnesses` maps its values to the names of
    the harness profiles. `notices` are patterns of user text the harness inserts; with `single_prompt`, every
    user message after the agent has answered is such a notice too, since the person wrote only what the agent
    was given before it acted.
    A trajectory that several input files hold is read once, from the first of them; its copies must hold the
    same document.
    """

    def __init__(
        self,
        path_metadata: str | None = None,
        harness_key: str | None = None,
        harnesses: dict[str, str] | None = None,
        notices: list[str] | None = None,
        single_prompt: bool = False,
        include_thinking: bool = False,
    ) -> None:
        self.path_metadata = re.compile(path_metadata) if path_metadata else None
        self.harness_key = harness_key
        self.harnesses = harnesses or {}
        self.notices = [re.compile(pattern) for pattern in notices or []]
        self.single_prompt = single_prompt
        self.include_thinking = include_thinking
        self.read_so_far: dict[str, tuple[Path, str]] = {}
        """The trajectories read so far, by identity, with their first file and the digest of their document; an
        archive can hold a run another archive holds too."""

    def path_groups(self, path: Path) -> dict[str, Any]:
        """The named groups `path_metadata` finds in the path of an input file, which become metadata; empty when
        no expression is configured."""
        if self.path_metadata is None:
            return {}
        match = self.path_metadata.search(str(path))
        if match is None:
            raise ValueError(f"{path} does not match path_metadata")
        return match.groupdict()

    def read_source(self, path: Path) -> AtifSource:
        """The document one input file holds and the identity of its trajectory. An adapter of a source that holds
        the document otherwise, such as a record a producer derives it from, overrides this alone, so the version
        check and the single reading of a trajectory several files hold stay here."""
        document = json.loads(path.read_bytes())
        identity = document.get("trajectory_id")
        if not identity:
            raise ValueError(f"{path} holds a trajectory with no trajectory_id, which identifies a trajectory of "
                             f"this format")
        return AtifSource(document=document, identity=str(identity))

    def read(self, path: Path, dataset: str) -> Iterator[Trajectory]:
        source = self.read_source(path)
        self._check_version(path, source.document)
        digest = hashlib.sha256(json.dumps(source.document, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if source.identity in self.read_so_far:
            first, seen = self.read_so_far[source.identity]
            if seen != digest:
                raise ValueError(f"{path} holds the trajectory of {first} with another document")
            return
        self.read_so_far[source.identity] = (path, digest)
        yield self._trajectory(path, dataset, source.document, source.identity, source.metadata,
                               {CONVERSATION: MAIN}, source.recorded_calls)
        for embedded in source.document.get("subagent_trajectories") or []:
            self._check_version(path, embedded)
            held = str(embedded.get("trajectory_id") or "")
            if not held:
                raise ValueError(f"{path} embeds a conversation with no trajectory_id, which the format requires "
                                 f"of an embedded trajectory and which identifies it among the others")
            yield self._trajectory(
                path, dataset, embedded, f"{source.identity}{HELD}{held}", source.metadata,
                {CONVERSATION: SUBAGENT, PARENT: source.identity}, source.subagent_calls.get(held))

    def _check_version(self, path: Path, document: dict[str, Any]) -> None:
        version = document["schema_version"]
        if version not in VERSIONS:
            raise ValueError(f"{path} declares the trajectory format {version!r}, and this adapter reads "
                             f"{sorted(VERSIONS)}")

    def _trajectory(self, path: Path, dataset: str, document: dict[str, Any], identity: str,
                    known: dict[str, Any], relation: dict[str, Any],
                    recorded_calls: list[str] | None) -> Trajectory:
        """One conversation of a document as a trajectory: the document's own when `relation` says it is the main
        one, and a conversation it embeds otherwise.

        `relation` is written last because which conversation this is, and which conversation holds it, is what
        the document says; nothing a source or a path states about the run can restate it.
        """
        agent = document.get("agent") or {}
        metrics = document.get("final_metrics") or {}
        metadata: dict[str, Any] = {
            "agent": agent.get("name"), "agent_version": agent.get("version"),
            "model": agent.get("model_name"), "session_id": document.get("session_id"),
            "trajectory_id": document.get("trajectory_id"),
            **{f"metrics_{name}": value for name, value in metrics.items() if name != "extra"},
            **{f"metrics_{name}": value for name, value in (metrics.get("extra") or {}).items()},
            **(document.get("extra") or {}),
            **known,
            **self.path_groups(path),
            **relation,
        }
        harness = None
        if self.harness_key is not None:
            value = str(metadata[self.harness_key])
            harness = self.harnesses.get(value, value)
        # Nobody typed anything in a conversation the agent delegated, so with `single_prompt` its opening
        # message is the harness writing too, and the conversation opens no turn of the person.
        parse = _Parse(answered=relation[CONVERSATION] == SUBAGENT)
        for turn in document["steps"]:
            self._turn(parse, turn)
        return Trajectory(id=identity, dataset=dataset, harness=harness, steps=parse.steps,
                          recorded_calls=recorded_calls,
                          metadata={name: value for name, value in metadata.items() if _is_scalar(value)})

    def _turn(self, parse: _Parse, turn: dict[str, Any]) -> None:
        """One ATIF turn as the steps it holds: the message of the person, or the reasoning and the message of the
        agent, then the tool calls of the turn and the results that answered them."""
        role = SOURCES[turn["source"]]
        timestamp = turn.get("timestamp")
        if role == "user":
            self._message(parse, turn)
        else:
            if self.include_thinking and turn.get("reasoning_content"):
                parse.add(role, "thinking", turn["reasoning_content"], timestamp)
            text = _message_text(turn["message"])
            if text:
                parse.add(role, "message", text, timestamp)
        for call in turn.get("tool_calls") or []:
            # The keys keep the order the producer wrote them in, which is the order the model sent, since a
            # probe reads the arguments as text.
            arguments = json.dumps(call["arguments"], ensure_ascii=False)
            parse.tool_names[call["tool_call_id"]] = call["function_name"]
            parse.add(role, "tool_call", arguments, timestamp, name=call["function_name"], arguments=arguments)
        self._tool_results(parse, turn)
        if role == "assistant":
            parse.answered = True

    def _message(self, parse: _Parse, turn: dict[str, Any]) -> None:
        """The message of a turn of the person. A part the harness inserted becomes a notice step of its own, so
        the message the person wrote opens a turn even where the harness wrote beside it; `single_prompt` makes
        every part a notice once the agent has answered. A part carrying no text is nothing the person wrote and
        becomes no step, so a message of no text opens no turn."""
        timestamp = turn.get("timestamp")
        pending: list[str] = []

        def flush() -> None:
            if pending:
                parse.add("user", "message", "\n".join(pending), timestamp)
                pending.clear()

        for text in _message_parts(turn["message"]):
            if not text:
                continue
            if (self.single_prompt and parse.answered) or any(p.match(text) for p in self.notices):
                flush()
                parse.add("user", "notice", text, timestamp, name=INJECTED)
            else:
                pending.append(text)
        flush()

    def _tool_results(self, parse: _Parse, turn: dict[str, Any]) -> None:
        """The results a turn carries, each named after the tool of the call it answers. A result that names no
        call of the trajectory stops the read: paired with another call, it attributes every action, relation and
        event of that call to the wrong tool."""
        timestamp = turn.get("timestamp")
        for position, result in enumerate((turn.get("observation") or {}).get("results") or []):
            call_id = result.get("source_call_id")
            if call_id not in parse.tool_names:
                raise ValueError(f"result {position} of step {turn['step_id']} answers the tool call {call_id!r}, "
                                 f"which no step of this trajectory makes")
            extra = result.get("extra") or {}
            parse.add("tool", "tool_result", _message_text(result.get("content")), timestamp,
                      name=parse.tool_names[call_id], is_error=bool(extra.get("is_error")))


def _message_parts(message: Any) -> list[str]:
    """The parts of an ATIF message or observation content: one part for a string, else one per content part,
    where a part of media stands as a placeholder, since the format names media by path or URL and carries no
    text of it. A producer gives one part per part of the message the client sent, which is what lets a pattern
    recognise the part the harness inserted beside the one the person wrote."""
    if message is None:
        return []
    if isinstance(message, str):
        return [message]
    parts = []
    for part in message:
        kind = part["type"]
        if kind == "text":
            parts.append(part["text"])
        elif kind in PLACEHOLDERS:
            parts.append(PLACEHOLDERS[kind])
        else:
            raise ValueError(f"unknown content part type {kind!r}")
    return parts


def _message_text(message: Any) -> str:
    return "\n".join(_message_parts(message))


def _is_scalar(value: Any) -> bool:
    """Whether a metadata value is one `traj ingest` keeps in the index row, so the index and the trajectories
    table hold the same metadata of a trajectory."""
    return isinstance(value, str | int | float | bool)


ADAPTER = AtifAdapter
