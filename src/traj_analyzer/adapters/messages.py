from __future__ import annotations

import ast
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from traj_analyzer.schema import Kind, Role, Step, Trajectory


class RoleMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Role
    kind: Kind = "message"


DEFAULT_ROLES = {
    "system": RoleMapping(role="system"),
    "user": RoleMapping(role="user"),
    "assistant": RoleMapping(role="assistant"),
    "tool": RoleMapping(role="tool", kind="tool_result"),
}


class MessagesAdapter:
    """Records holding a list of `{role, content}` messages, one trajectory per record.

    Files are `.jsonl` with one record per line, or `.json` holding a record or a list of records.
    `role_map` maps every source role to a unified role and kind; an unmapped role is an error.
    `tool_call_format` says how the content of a `tool_call` message encodes the tool name.
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

    def read(self, path: Path, dataset: str) -> Iterator[Trajectory]:
        for n, record in enumerate(_records(path)):
            tid = str(record[self.id_key]) if self.id_key else f"{path.stem}-{n}"
            messages = record[self.messages_key]
            if self.messages_encoding == "json_string":
                messages = json.loads(messages)
            metadata = {k: v for k, v in record.items() if k != self.messages_key}
            if self.metadata_keys is not None:
                metadata = {k: record[k] for k in self.metadata_keys}
            yield Trajectory(id=tid, dataset=dataset, metadata=metadata,
                             steps=[self._step(i, m) for i, m in enumerate(messages)])

    def _step(self, index: int, message: dict[str, Any]) -> Step:
        source = message[self.role_key]
        if source not in self.roles:
            raise ValueError(f"Role {source!r} is not in role_map")
        mapping = self.roles[source]
        content = message[self.content_key]
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        name = None
        if mapping.kind == "tool_call" and self.tool_call_format != "text":
            parsed = json.loads(text) if self.tool_call_format == "json" else ast.literal_eval(text)
            name = parsed["name"]
        return Step(index=index, role=mapping.role, kind=mapping.kind, content=text, name=name)


def _records(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as lines:
            for line in lines:
                yield json.loads(line)
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    yield from data if isinstance(data, list) else [data]
