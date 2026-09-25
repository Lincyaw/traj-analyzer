from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from traj_analyzer.adapters._blocks import PLACEHOLDERS, block_text
from traj_analyzer.files import iter_jsonl
from traj_analyzer.schema import Role, Step, Trajectory

_LOCAL_COMMAND = re.compile(
    r"\s*<(command-name|command-message|local-command-stdout|local-command-stderr|bash-input|bash-stdout|bash-stderr)>"
)


class ClaudeCodeAdapter:
    """Claude Code session files, `~/.claude/projects/<project>/<session>.jsonl`.

    Conversation records are read in file order, so turns abandoned by a rewind stay in the trajectory.
    User records not typed by a person, such as task notifications and local commands, become system steps.
    """

    def __init__(
        self,
        include_thinking: bool = False,
        include_sidechains: bool = False,
        include_meta: bool = False,
        min_user_turns: int = 1,
    ) -> None:
        self.include_thinking = include_thinking
        self.include_sidechains = include_sidechains
        self.include_meta = include_meta
        self.min_user_turns = min_user_turns

    def read(self, path: Path, dataset: str) -> Iterator[Trajectory]:
        steps: list[Step] = []
        tool_names: dict[str, str] = {}
        metadata: dict[str, Any] = {"file": str(path)}
        for record in iter_jsonl(path):
            kind = record["type"]
            if kind == "ai-title":
                metadata["title"] = record["aiTitle"]
            if kind not in ("user", "assistant"):
                continue
            if record.get("isSidechain") and not self.include_sidechains:
                continue
            if record.get("isMeta") and not self.include_meta:
                continue
            for key in ("cwd", "gitBranch", "version"):
                metadata.setdefault(key, record.get(key))
            message = record["message"]
            if kind == "assistant":
                metadata.setdefault("model", message["model"])
            speaker = kind
            if kind == "user" and record.get("origin") is not None and record["origin"]["kind"] != "human":
                speaker = "system"
            for step in self._steps(speaker, message["content"], record["timestamp"], tool_names):
                steps.append(step.model_copy(update={"index": len(steps)}))
        user_turns = sum(1 for s in steps if s.is_user_message)
        if user_turns < self.min_user_turns:
            return
        metadata["started_at"] = steps[0].timestamp
        metadata["ended_at"] = steps[-1].timestamp
        yield Trajectory(id=path.stem, dataset=dataset, metadata=metadata, steps=steps)

    def _steps(
        self, kind: str, content: Any, timestamp: str, tool_names: dict[str, str]
    ) -> Iterator[Step]:
        if isinstance(content, str):
            yield Step(index=0, role=_text_role(kind, content), content=content, timestamp=timestamp)
            return
        for block in content:
            match block["type"]:
                case "text":
                    yield Step(index=0, role=_text_role(kind, block["text"]), content=block["text"],
                               timestamp=timestamp)
                case "image":
                    yield Step(index=0, role=kind, content=PLACEHOLDERS["image"],  # type: ignore[arg-type]
                               timestamp=timestamp)
                case "thinking":
                    # Signed thinking blocks often carry an empty text; they produce no step.
                    if self.include_thinking and block["thinking"]:
                        yield Step(index=0, role="assistant", kind="thinking",
                                   content=block["thinking"], timestamp=timestamp)
                case "tool_use":
                    tool_names[block["id"]] = block["name"]
                    yield Step(index=0, role="assistant", kind="tool_call", name=block["name"],
                               content=json.dumps(block["input"], ensure_ascii=False),
                               timestamp=timestamp)
                case "tool_result":
                    yield Step(index=0, role="tool", kind="tool_result",
                               name=tool_names[block["tool_use_id"]],
                               content=block_text(block["content"]),
                               is_error=bool(block.get("is_error")), timestamp=timestamp)
                case "fallback":
                    yield Step(index=0, role="system",
                               content=f"model fallback from {block['from']['model']} "
                                       f"to {block['to']['model']}",
                               timestamp=timestamp)
                case other:
                    raise ValueError(f"Unknown Claude Code content block type: {other}")


def _text_role(speaker: str, text: str) -> Role:
    """Local slash commands and `!` shell commands are recorded as user text but are not typed prose."""
    if speaker == "user" and _LOCAL_COMMAND.match(text):
        return "system"
    return speaker  # type: ignore[return-value]
