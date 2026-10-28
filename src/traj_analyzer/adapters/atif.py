from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from traj_analyzer.adapters._blocks import PLACEHOLDERS
from traj_analyzer.schema import INJECTED, Kind, Role, Step, Trajectory

VERSIONS = {f"ATIF-v1.{minor}" for minor in range(9)}
"""The schema versions of the Agent Trajectory Interchange Format this adapter reads."""
SOURCES: dict[str, Role] = {"system": "system", "user": "user", "agent": "assistant"}


class Steps:
    """The steps of one ATIF trajectory, in the order its turns hold them.

    One ATIF step is a turn: a message with its reasoning, its tool calls and the results of those calls. It
    becomes several steps here, each carrying the `step_id` of that turn as `source_step`, so a position found
    here can be named in the numbering a reader of the same file shows.
    """

    def __init__(self, notices: list[re.Pattern[str]], single_prompt: bool, include_thinking: bool) -> None:
        self.notices = notices
        self.single_prompt = single_prompt
        self.include_thinking = include_thinking
        self.steps: list[Step] = []
        self.tools: dict[str, str] = {}
        self.answered = False
        """Whether the agent has answered, which ends the turns `single_prompt` counts as the person's own."""

    def add(self, role: Role, kind: Kind, content: str, turn: dict[str, Any], name: str | None = None,
            is_error: bool = False, arguments: str | None = None) -> None:
        self.steps.append(Step(index=len(self.steps), role=role, kind=kind, content=content, name=name,
                               is_error=is_error, timestamp=turn.get("timestamp"), arguments=arguments,
                               source_step=turn["step_id"]))

    def message(self, turn: dict[str, Any]) -> None:
        """The message of a turn of the person. A part the harness inserted becomes a notice step of its own, so
        the message the person wrote opens a turn even where the harness wrote beside it; `single_prompt` makes
        every part a notice once the agent has answered."""
        pending: list[str] = []

        def flush() -> None:
            if pending:
                self.add("user", "message", "\n".join(pending), turn)
                pending.clear()

        for text in message_parts(turn["message"]):
            if (self.single_prompt and self.answered) or any(p.match(text) for p in self.notices):
                flush()
                self.add("user", "notice", text, turn, name=INJECTED)
            else:
                pending.append(text)
        flush()

    def turn(self, turn: dict[str, Any]) -> None:
        role = SOURCES[turn["source"]]
        if role == "user":
            self.message(turn)
        else:
            if self.include_thinking and turn.get("reasoning_content"):
                self.add(role, "thinking", turn["reasoning_content"], turn)
            text = message_text(turn["message"])
            if text:
                self.add(role, "message", text, turn)
        for call in turn.get("tool_calls") or []:
            arguments = json.dumps(call["arguments"], ensure_ascii=False, sort_keys=True)
            self.tools[call["tool_call_id"]] = call["function_name"]
            self.add(role, "tool_call", arguments, turn, name=call["function_name"], arguments=arguments)
        for result in (turn.get("observation") or {}).get("results") or []:
            extra = result.get("extra") or {}
            self.add("tool", "tool_result", message_text(result.get("content")), turn,
                     name=self.tools.get(result.get("source_call_id") or ""), is_error=bool(extra.get("is_error")))
        if role == "assistant":
            self.answered = True


def message_parts(message: Any) -> list[str]:
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
        else:
            parts.append(PLACEHOLDERS.get(kind, f"[{kind}]"))
    return parts


def message_text(message: Any) -> str:
    return "\n".join(message_parts(message))


def flatten(document: dict[str, Any], notices: list[re.Pattern[str]], single_prompt: bool,
            include_thinking: bool) -> list[Step]:
    steps = Steps(notices=notices, single_prompt=single_prompt, include_thinking=include_thinking)
    for turn in document["steps"]:
        steps.turn(turn)
    return steps.steps


class AtifAdapter:
    """Trajectories in the Agent Trajectory Interchange Format, one file holding one trajectory.

    An ATIF step is a turn, so the steps here are the message, the reasoning, the tool calls and the results of
    one turn, each carrying that turn's `step_id` as `source_step`. The agent, the model, the final metrics and
    the producer's own `extra` become metadata. A subagent trajectory the document embeds is read as a trajectory
    of its own, identified by its `trajectory_id`.

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
        subagents: bool = False,
    ) -> None:
        self.path_metadata = re.compile(path_metadata) if path_metadata else None
        self.harness_key = harness_key
        self.harnesses = harnesses or {}
        self.notices = [re.compile(pattern) for pattern in notices or []]
        self.single_prompt = single_prompt
        self.include_thinking = include_thinking
        self.subagents = subagents
        """Whether an embedded subagent trajectory is read as a trajectory of its own."""
        self.read_so_far: dict[str, tuple[Path, str]] = {}
        """The trajectories read so far, by id, with their first file and the digest of its document; an archive
        can hold a run another archive holds too."""

    def read(self, path: Path, dataset: str) -> Iterator[Trajectory]:
        raw = path.read_bytes()
        document = json.loads(raw)
        version = document["schema_version"]
        if version not in VERSIONS:
            raise ValueError(f"{path} declares the trajectory format {version!r}, and this adapter reads "
                             f"{sorted(VERSIONS)}")
        identity = path.parent.name
        digest = hashlib.sha256(raw).hexdigest()
        if identity in self.read_so_far:
            first, seen = self.read_so_far[identity]
            if seen != digest:
                raise ValueError(f"{path} holds the trajectory of {first} with another document")
            return
        self.read_so_far[identity] = (path, digest)
        yield self._trajectory(path, dataset, document, identity)
        if not self.subagents:
            return
        for embedded in document.get("subagent_trajectories") or []:
            identity = embedded.get("trajectory_id")
            if not identity:
                raise ValueError(f"{path} embeds a subagent trajectory with no trajectory_id")
            yield self._trajectory(path, dataset, embedded, identity)

    def _trajectory(self, path: Path, dataset: str, document: dict[str, Any], identity: str) -> Trajectory:
        agent = document.get("agent") or {}
        metadata: dict[str, Any] = {
            "agent": agent.get("name"), "agent_version": agent.get("version"),
            "model": agent.get("model_name"), "session_id": document.get("session_id"),
            "trajectory_id": document.get("trajectory_id"), "turns": len(document["steps"]),
            **{f"metrics_{name}": value for name, value in (document.get("final_metrics") or {}).items()},
            **{name: value for name, value in (document.get("extra") or {}).items()
               if not isinstance(value, (dict, list))},
        }
        if self.path_metadata is not None:
            match = self.path_metadata.search(str(path))
            if match is None:
                raise ValueError(f"{path} does not match path_metadata")
            metadata.update(match.groupdict())
        harness = None
        if self.harness_key is not None:
            value = str(metadata[self.harness_key])
            harness = self.harnesses.get(value, value)
        return Trajectory(id=identity, dataset=dataset, metadata={k: v for k, v in metadata.items() if v is not None},
                          steps=flatten(document, self.notices, self.single_prompt, self.include_thinking),
                          harness=harness)


ADAPTER = AtifAdapter
