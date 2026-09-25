from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from traj_analyzer.adapters.claude_code import ClaudeCodeAdapter
from traj_analyzer.adapters.messages import MessagesAdapter
from traj_analyzer.schema import Trajectory

DATA = Path(__file__).parent / "data" / "messages"
SWESMITH_EXTRA = ["action", "agent", "message_type", "thought", "cache_control"]


def read(name: str, **options: Any) -> list[Trajectory]:
    return list(MessagesAdapter(**options).read(DATA / f"{name}.jsonl", name))


def raw_messages(name: str, key: str) -> list[list[dict[str, Any]]]:
    rows = [json.loads(line) for line in (DATA / f"{name}.jsonl").open(encoding="utf-8")]
    return [json.loads(r[key]) if isinstance(r[key], str) else r[key] for r in rows]


def assert_results_named_after_calls(trajectories: list[Trajectory]) -> None:
    assert any(s.kind == "tool_result" for t in trajectories for s in t.steps)
    for trajectory in trajectories:
        calls = {s.name for s in trajectory.steps if s.kind == "tool_call"}
        assert all(s.name in calls for s in trajectory.steps if s.kind == "tool_result")
        assert [s.index for s in trajectory.steps] == list(range(len(trajectory.steps)))


def test_openai_tool_calls_with_reasoning() -> None:
    plain = read("unified", id_key=None)
    thinking = read("unified", id_key=None, include_thinking=True)
    assert_results_named_after_calls(thinking)
    assert not any(s.kind == "thinking" for t in plain for s in t.steps)
    assert any(s.kind == "thinking" for t in thinking for s in t.steps)
    messages = raw_messages("unified", "messages")
    expected_calls = sum(len(m.get("tool_calls") or []) for ms in messages for m in ms)
    assert sum(s.kind == "tool_call" for t in plain for s in t.steps) == expected_calls
    call = next(s for t in plain for s in t.steps if s.kind == "tool_call")
    assert isinstance(json.loads(call.content), dict)


def test_openai_messages_with_null_keys_and_named_tool_results() -> None:
    trajectories = read("openhands", id_key="run_id")
    assert_results_named_after_calls(trajectories)
    messages = raw_messages("openhands", "messages")
    names = [m["name"] for ms in messages for m in ms if m["role"] == "tool"]
    assert [s.name for t in trajectories for s in t.steps if s.kind == "tool_result"] == names


def test_content_parts_and_tool_call_ids() -> None:
    with pytest.raises(ValueError, match="unknown message keys"):
        read("swesmith", messages_encoding="json_string", id_key="traj_id")
    (trajectory,) = read("swesmith", messages_encoding="json_string", id_key="traj_id", ignore_keys=SWESMITH_EXTRA)
    assert_results_named_after_calls([trajectory])
    (messages,) = raw_messages("swesmith", "messages")
    tool_texts = ["\n".join(p["text"] for p in m["content"]) for m in messages if m["role"] == "tool"]
    assert [s.content for s in trajectory.steps if s.kind == "tool_result"] == tool_texts


def test_anthropic_content_blocks() -> None:
    trajectories = read("nlile", messages_key="messages_json", messages_encoding="json_string")
    assert_results_named_after_calls(trajectories)
    for trajectory, messages in zip(trajectories, raw_messages("nlile", "messages_json"), strict=True):
        blocks = [b for m in messages if isinstance(m["content"], list) for b in m["content"]]
        uses = [b["name"] for b in blocks if b["type"] == "tool_use"]
        assert [s.name for s in trajectory.steps if s.kind == "tool_call"] == uses
        assert all(s.role == "tool" for s in trajectory.steps if s.kind == "tool_result")


def test_records_without_messages_are_an_error_unless_skipped(tmp_path: Path) -> None:
    rows = (DATA / "nlile.jsonl").read_text(encoding="utf-8").splitlines()
    empty = json.loads(rows[0]) | {"messages_json": None}
    path = tmp_path / "nlile.jsonl"
    path.write_text("\n".join([json.dumps(empty), *rows]), encoding="utf-8")
    adapter = MessagesAdapter(messages_key="messages_json", messages_encoding="json_string")
    with pytest.raises(ValueError, match="skip_empty"):
        list(adapter.read(path, "n"))
    skipping = MessagesAdapter(messages_key="messages_json", messages_encoding="json_string", skip_empty=True)
    assert len(list(skipping.read(path, "n"))) == len(rows)


def test_tool_call_id_takes_precedence_over_placeholder_names() -> None:
    trajectories = read("mimo", id_key="session_id", include_thinking=True)
    assert_results_named_after_calls(trajectories)
    placeholders = [m for ms in raw_messages("mimo", "messages") for m in ms if m.get("name") == "unknown_tool"]
    assert placeholders
    assert not any(s.name == "unknown_tool" for t in trajectories for s in t.steps)


def test_anthropic_messages_from_a_claude_code_session_match_the_claude_code_adapter(tmp_path: Path) -> None:
    session = next((DATA.parent / "claude_code").glob("*.jsonl"))
    records = [json.loads(line) for line in session.open(encoding="utf-8")]
    messages = [r["message"] for r in records
                if r["type"] in ("user", "assistant") and not r.get("isMeta") and not r.get("isSidechain")]
    assert any(isinstance(m["content"], list) and any(b["type"] == "thinking" for b in m["content"])
               for m in messages)
    path = tmp_path / "session.jsonl"
    path.write_text(json.dumps({"id": "s", "messages": messages}), encoding="utf-8")
    adapter = MessagesAdapter(include_thinking=True, ignore_keys=["id", "type", "model", "stop_reason",
                                                                  "stop_sequence", "stop_details", "usage",
                                                                  "container", "context_management",
                                                                  "input_transformations"])
    (trajectory,) = adapter.read(path, "s")
    (reference,) = ClaudeCodeAdapter(include_thinking=True).read(session, "s")
    calls = [(s.name, s.content) for s in trajectory.steps if s.kind == "tool_call"]
    assert calls == [(s.name, s.content) for s in reference.steps if s.kind == "tool_call"]
    results = [(s.name, s.is_error) for s in trajectory.steps if s.kind == "tool_result"]
    assert results == [(s.name, s.is_error) for s in reference.steps if s.kind == "tool_result"]


def test_sharegpt_keys() -> None:
    trajectories = read("hermes", messages_key="conversations", role_key="from", content_key="value")
    assert all([s.role for s in t.steps] == ["system", "user", "assistant"] for t in trajectories)
