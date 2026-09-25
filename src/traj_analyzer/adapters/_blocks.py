from __future__ import annotations

from typing import Any

PLACEHOLDERS = {"image": "[image]", "image_url": "[image]", "input_image": "[image]", "input_audio": "[audio]",
                "file": "[file]", "document": "[document]"}


def block_text(content: Any) -> str:
    """Text of an Anthropic tool_result content: a string, or a list of text, media and tool_reference blocks."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        kind = block["type"]
        if kind == "text":
            parts.append(block["text"])
        elif kind in PLACEHOLDERS:
            parts.append(PLACEHOLDERS[kind])
        elif kind == "tool_reference":
            parts.append(f"[tool_reference: {block['tool_name']}]")
        else:
            raise ValueError(f"unknown tool_result block type {kind!r}")
    return "\n".join(parts)
