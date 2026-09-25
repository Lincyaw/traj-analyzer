from __future__ import annotations

import ast
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from traj_analyzer.adapters._blocks import PLACEHOLDERS, block_text
from traj_analyzer.files import iter_jsonl
from traj_analyzer.schema import Kind, Role, Step, Trajectory


class RoleMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Role
    kind: Kind = "message"


DEFAULT_ROLES = {
    "system": RoleMapping(role="system"),
    "developer": RoleMapping(role="system"),
    "user": RoleMapping(role="user"),
    "human": RoleMapping(role="user"),
    "assistant": RoleMapping(role="assistant"),
    "gpt": RoleMapping(role="assistant"),
    "tool": RoleMapping(role="tool", kind="tool_result"),
    "function": RoleMapping(role="tool", kind="tool_result"),
}

# Message keys this adapter reads besides the configured role and content keys.
KNOWN_KEYS = {"name", "tool_calls", "tool_call_id", "tool_call_ids", "function_call", "refusal",
              "reasoning_content", "reasoning", "audio"}


@dataclass
class _Parse:
    """State of one trajectory: its steps and the tool names seen so far, keyed by tool call id."""

    steps: list[Step] = field(default_factory=list)
    tool_names: dict[str, str] = field(default_factory=dict)

    def add(self, role: Role, kind: Kind, content: str, name: str | None = None, is_error: bool = False) -> None:
        self.steps.append(Step(index=len(self.steps), role=role, kind=kind, content=content, name=name,
                               is_error=is_error))


class MessagesAdapter:
    """Records holding a list of messages, one trajectory per record.

    Files are `.jsonl` with one record per line, or `.json` holding a record or a list of records.
    Messages follow the OpenAI chat format, the Anthropic messages format, or ShareGPT with configured keys.
    Content is a string or a list of parts; parts cover OpenAI content parts and Anthropic content blocks.
    `role_map` extends the default role mapping; an unmapped role is an error.
    Message keys other than the known ones are an error unless listed in `ignore_keys`; keys whose value is null count
    as absent.
    `tool_call_format` says how the content of a message mapped to kind `tool_call` encodes the tool name.
    A record whose message list is null or empty is an error unless `skip_empty` is set.
    """

    def __init__(
        self,
        messages_key: str = "messages",
        messages_encoding: Literal["list", "json_string"] = "list",
        role_key: str = "role",
        content_key: str = "content",
        id_key: str | None = "id",
        metadata_keys: list[str] | None = None,
        role_map: dict[str, dict[str, str]] | None = None,
        tool_call_format: Literal["text", "json", "python_literal"] = "text",
        include_thinking: bool = False,
        ignore_keys: list[str] | None = None,
        skip_empty: bool = False,
    ) -> None:
        self.messages_key = messages_key
        self.messages_encoding = messages_encoding
        self.role_key = role_key
        self.content_key = content_key
        self.id_key = id_key
        self.metadata_keys = metadata_keys
        self.roles = dict(DEFAULT_ROLES)
        for source, target in (role_map or {}).items():
            self.roles[source] = RoleMapping.model_validate(target)
        self.tool_call_format = tool_call_format
        self.include_thinking = include_thinking
        self.allowed_keys = {role_key, content_key, *KNOWN_KEYS, *(ignore_keys or [])}
        self.skip_empty = skip_empty

    def read(self, path: Path, dataset: str) -> Iterator[Trajectory]:
        for n, record in enumerate(_records(path)):
            tid = str(record[self.id_key]) if self.id_key else f"{path.stem}-{n}"
            messages = record[self.messages_key]
            if not messages:
                if self.skip_empty:
                    continue
                raise ValueError(f"{path}: record {tid} has no {self.messages_key}; set skip_empty to skip it")
            if self.messages_encoding == "json_string":
                messages = json.loads(messages)
            metadata = {k: v for k, v in record.items() if k != self.messages_key}
            if self.metadata_keys is not None:
                metadata = {k: record[k] for k in self.metadata_keys}
            parse = _Parse()
            for position, message in enumerate(messages):
                try:
                    self._message(parse, message)
                except (KeyError, ValueError, TypeError) as error:
                    raise ValueError(f"{path}: record {tid}, message {position}: {error!r}") from error
            yield Trajectory(id=tid, dataset=dataset, metadata=metadata, steps=parse.steps)

    def _message(self, parse: _Parse, message: dict[str, Any]) -> None:
        present = {k: v for k, v in message.items() if v is not None}
        unknown = sorted(set(present) - self.allowed_keys)
        if unknown:
            raise ValueError(f"unknown message keys {unknown}; list them in ignore_keys to skip them")
        source = present[self.role_key]
        if source not in self.roles:
            raise ValueError(f"role {source!r} is not in role_map")
        mapping = self.roles[source]
        name = self._result_name(parse, present) if mapping.kind == "tool_result" else None

        reasoning = present.get("reasoning_content", present.get("reasoning"))
        if self.include_thinking and reasoning:
            parse.add("assistant", "thinking", _as_text(reasoning))

        content = present.get(self.content_key)
        if isinstance(content, str):
            if content:
                self._text(parse, mapping, content, name)
        elif isinstance(content, list):
            self._parts(parse, mapping, content, name)
        elif content is not None:
            raise TypeError(f"content must be a string, a list of parts or null, got {type(content).__name__}")

        if "refusal" in present:
            parse.add(mapping.role, "message", f"[refusal] {present['refusal']}")
        if "audio" in present:
            parse.add(mapping.role, "message", f"[audio] {present['audio'].get('transcript', '')}".rstrip())
        if "function_call" in present:
            call = present["function_call"]
            parse.add("assistant", "tool_call", _arguments(call.get("arguments")), name=call["name"])
        for call in present.get("tool_calls", []):
            function = call["function"]
            if "id" in call and call["id"] is not None:
                parse.tool_names[call["id"]] = function["name"]
            parse.add("assistant", "tool_call", _arguments(function.get("arguments")), name=function["name"])

    def _result_name(self, parse: _Parse, message: dict[str, Any]) -> str | None:
        """Name a tool result after the call its id answers; the message's own name is used only without an id.

        Some exports fill `name` with a placeholder such as `unknown_tool`, while the call id always links a real call.
        """
        ids = [message["tool_call_id"]] if "tool_call_id" in message else message.get("tool_call_ids", [])
        if ids:
            return ",".join(dict.fromkeys(parse.tool_names[i] for i in ids))
        return str(message["name"]) if "name" in message else None

    def _text(self, parse: _Parse, mapping: RoleMapping, text: str, name: str | None) -> None:
        if mapping.kind == "tool_call" and self.tool_call_format != "text":
            parsed = json.loads(text) if self.tool_call_format == "json" else ast.literal_eval(text)
            name = parsed["name"]
        parse.add(mapping.role, mapping.kind, text, name=name)

    def _parts(self, parse: _Parse, mapping: RoleMapping, parts: list[dict[str, Any]], name: str | None) -> None:
        texts: list[str] = []

        def flush() -> None:
            if texts:
                self._text(parse, mapping, "\n".join(texts), name)
                texts.clear()

        for part in parts:
            kind = part["type"]
            if kind == "text":
                texts.append(part["text"])
            elif kind == "refusal":
                texts.append(f"[refusal] {part['refusal']}")
            elif kind in PLACEHOLDERS:
                texts.append(PLACEHOLDERS[kind])
            elif kind in ("thinking", "redacted_thinking"):
                flush()
                text = part.get("thinking") or "[redacted thinking]"
                if self.include_thinking and (kind == "redacted_thinking" or part["thinking"]):
                    parse.add("assistant", "thinking", text)
            elif kind == "tool_use":
                flush()
                parse.tool_names[part["id"]] = part["name"]
                parse.add("assistant", "tool_call", json.dumps(part["input"], ensure_ascii=False), name=part["name"])
            elif kind == "tool_result":
                flush()
                parse.add("tool", "tool_result", block_text(part.get("content", "")),
                          name=parse.tool_names[part["tool_use_id"]], is_error=bool(part.get("is_error")))
            else:
                raise ValueError(f"unknown content part type {kind!r}")
        flush()


def _arguments(arguments: Any) -> str:
    if arguments is None:
        return ""
    return arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)


def _as_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _records(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix == ".jsonl":
        yield from iter_jsonl(path)
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    yield from data if isinstance(data, list) else [data]
