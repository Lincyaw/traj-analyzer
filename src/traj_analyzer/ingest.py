from __future__ import annotations

import fnmatch
import glob
import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from traj_analyzer.adapters import load_adapter
from traj_analyzer.files import iter_jsonl, write_jsonl
from traj_analyzer.project import ConfigError, DatasetConfig, Project, RenderConfig
from traj_analyzer.schema import Step, Trajectory

INDEX_FILE = "index.jsonl"


class IndexRow(BaseModel):
    key: str
    dataset: str
    id: str
    sha256: str
    """Hash of the rendered Markdown; it goes into LLM requests so a changed file is never served a stale answer."""
    n_steps: int
    n_chunks: int
    chars: int
    metadata: dict[str, Any] = Field(default_factory=dict)


def ingest(project: Project, datasets: list[str] | None = None) -> dict[str, int]:
    names = datasets or list(project.config.datasets)
    counts: dict[str, int] = {}
    for name in names:
        if name not in project.config.datasets:
            raise ConfigError(f"Dataset not configured in traj.yaml: {name}")
        counts[name] = _ingest_one(project, name)
    return counts


def _ingest_one(project: Project, name: str) -> int:
    config = project.config.datasets[name]
    adapter = load_adapter(config.adapter, config.options, project.root)
    project.dataset_dir(name).mkdir(parents=True, exist_ok=True)
    rows: dict[str, IndexRow] = {}
    for path in input_files(config, project.root):
        for trajectory in adapter.read(path, name):
            trajectory = trajectory.model_copy(update={"id": safe_id(trajectory.id)})
            if trajectory.id in rows:
                raise ValueError(f"Duplicate trajectory id {trajectory.id} in {path}")
            check_recorded_calls(trajectory, path)
            rows[trajectory.id] = _write(project, trajectory, config.description)
    write_jsonl(project.dataset_dir(name) / INDEX_FILE, (row.model_dump_json() for row in rows.values()))
    return len(rows)


def check_recorded_calls(trajectory: Trajectory, path: Path) -> None:
    """Stop at this trajectory when its steps lack a call the source records, counted by tool.

    The recorded calls are what the source reports of the agent's calls, which can be a part of them: the recorder
    of a proxy may report the function calls of a response and not the calls a model sends in another form. So every
    recorded call has to appear among the parsed calls, and a parsed call the source does not report is no error.
    """
    if trajectory.recorded_calls is None:
        return
    parsed = Counter(step.name or "" for step in trajectory.steps if step.kind == "tool_call")
    missing = Counter(trajectory.recorded_calls) - parsed
    if missing:
        raise ValueError(f"{trajectory.key} of {path} lacks tool calls the source records, by tool and count "
                         f"{dict(missing)}; the adapter parsed {sum(parsed.values())} calls of "
                         f"{len(trajectory.recorded_calls)} recorded")


def input_files(config: DatasetConfig, root: Path) -> Iterator[Path]:
    excluded = [_absolute(pattern, root) for pattern in config.exclude]
    for pattern in config.input:
        for match in sorted(glob.glob(_absolute(pattern, root), recursive=True)):
            if Path(match).is_file() and not any(fnmatch.fnmatch(match, e) for e in excluded):
                yield Path(match)


def _absolute(pattern: str, root: Path) -> str:
    path = Path(pattern).expanduser()
    return str(path if path.is_absolute() else root / path)


def safe_id(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", raw)[:120]
    if not cleaned:
        raise ValueError("Trajectory id is empty")
    return cleaned


def _write(project: Project, trajectory: Trajectory, description: str) -> IndexRow:
    markdown, n_chunks = render_markdown(trajectory, project.config.render, description)
    project.trajectory_path(trajectory.key, ".json").write_text(trajectory.model_dump_json(), encoding="utf-8")
    project.trajectory_path(trajectory.key, ".md").write_text(markdown, encoding="utf-8")
    scalars = {k: v for k, v in trajectory.metadata.items() if _is_scalar(v)}
    return IndexRow(
        key=trajectory.key, dataset=trajectory.dataset, id=trajectory.id,
        sha256=hashlib.sha256(markdown.encode()).hexdigest(),
        n_steps=len(trajectory.steps), n_chunks=n_chunks, chars=len(markdown), metadata=scalars,
    )


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, str | int | float | bool)


def shown_keys(metadata: dict[str, Any], patterns: list[str]) -> dict[str, Any]:
    """The metadata without the keys `render.hide_metadata` keeps from the models that read a trajectory."""
    return {key: value for key, value in metadata.items()
            if not any(fnmatch.fnmatchcase(key, pattern) for pattern in patterns)}


def visible_metadata(text: str, patterns: list[str]) -> str:
    """The metadata of the trajectories table, JSON text, with the hidden keys left out."""
    return json.dumps(shown_keys(json.loads(text), patterns), ensure_ascii=False, sort_keys=True)


def render_markdown(trajectory: Trajectory, render: RenderConfig, description: str = "") -> tuple[str, int]:
    """Render the dataset description, shown metadata, and one heading per step with chunk markers.

    Chunks break only between steps, so a step longer than the budget forms a chunk of its own.
    """
    header = [f"# {trajectory.key}", ""]
    if description:
        header += [description.strip(), ""]
    shown = shown_keys(trajectory.metadata, render.hide_metadata)
    if shown:
        header += ["```json", json.dumps(shown, ensure_ascii=False, indent=1), "```", ""]
    chunks: list[list[str]] = [[]]
    size = 0
    for step in trajectory.steps:
        block = _render_step(step, render.max_step_chars)
        if chunks[-1] and size + len(block) > render.chunk_chars:
            chunks.append([])
            size = 0
        chunks[-1].append(block)
        size += len(block)
    body = []
    for k, chunk in enumerate(chunks):
        body.append(f"<!-- chunk {k} -->\n")
        body.extend(chunk)
    return "\n".join(header) + "\n" + "\n".join(body), len(chunks)


def turn_units(trajectory: Trajectory) -> list[tuple[int, int]]:
    """The turns of the user: from each message the user typed to the step before the next one."""
    starts = [step.index for step in trajectory.steps if step.is_user_message]
    if not starts:
        return []
    last = trajectory.steps[-1].index
    return [(start, (starts[i + 1] - 1) if i + 1 < len(starts) else last) for i, start in enumerate(starts)]


def window_units(trajectory: Trajectory, steps: int, stride: int) -> list[tuple[int, int]]:
    """Windows of `steps` steps every `stride` steps, the last one ending at the last step."""
    indexes = [step.index for step in trajectory.steps]
    if not indexes:
        return []
    if len(indexes) <= steps:
        return [(indexes[0], indexes[-1])]
    starts = list(range(0, len(indexes) - steps + 1, stride))
    if starts[-1] != len(indexes) - steps:
        starts.append(len(indexes) - steps)
    return [(indexes[start], indexes[start + steps - 1]) for start in starts]


def _header(trajectory: Trajectory, title: str, render: RenderConfig, description: str) -> list[str]:
    header = [f"# {trajectory.key} {title}", ""]
    if description:
        header += [description.strip(), ""]
    shown = shown_keys(trajectory.metadata, render.hide_metadata)
    if shown:
        header += ["```json", json.dumps(shown, ensure_ascii=False, indent=1), "```", ""]
    return header


def render_window(trajectory: Trajectory, start: int, end: int, render: RenderConfig, description: str = "") -> str:
    """Render one window of a trajectory: the last message the user typed before or inside it, then its steps."""
    header = _header(trajectory, f"steps #{start} to #{end}", render, description)
    request = next((step for step in reversed(trajectory.steps) if step.is_user_message and step.index <= end), None)
    body = ["## Request", ""]
    body.append(_render_step(request, render.max_step_chars) if request is not None else "No message from the user.\n")
    body += ["## Steps", "", "<!-- chunk 0 -->", ""]
    body += [_render_step(step, render.max_step_chars) for step in trajectory.steps if start <= step.index <= end]
    return "\n".join(header) + "\n" + "\n".join(body)


def render_turns(
    trajectory: Trajectory, units: list[tuple[int, int]], before: int, render: RenderConfig, description: str = "",
    first: int = 0,
) -> str:
    """Render turns of a trajectory for a call that judges every message the user typed: per turn, the `before`
    steps preceding the message, which show what the assistant did last, and the message itself.

    `first` numbers the turns when the file holds a part of them.
    """
    header = _header(trajectory, f"turns {first} to {first + len(units) - 1} of the user", render, description)
    by_index = {step.index: step for step in trajectory.steps}
    body = ["<!-- chunk 0 -->", ""]
    for number, (start, end) in enumerate(units, start=first):
        body += [f"## Turn {number}: steps #{start} to #{end}", ""]
        shown = [by_index[index] for index in range(max(0, start - before), start) if index in by_index]
        if shown:
            body += ["### Before the message", ""]
            body += [_render_step(step, render.max_step_chars) for step in shown]
        body += ["### The message", ""]
        body.append(_render_step(by_index[start], render.max_step_chars))
    return "\n".join(header) + "\n" + "\n".join(body)


def _render_step(step: Step, limit: int) -> str:
    title = f"### #{step.index} {step.role} · {step.kind}"
    if step.name:
        title += f" · {step.name}"
    if step.is_error:
        title += " · ERROR"
    content = step.content
    if len(content) > limit:
        content = content[:limit] + f"\n…[truncated {len(content) - limit} chars]"
    return f"{title}\n\n{content}\n"


def load_index(project: Project, datasets: list[str] | None = None) -> list[IndexRow]:
    rows: list[IndexRow] = []
    for name in datasets or list(project.config.datasets):
        path = project.dataset_dir(name) / INDEX_FILE
        if not path.is_file():
            raise ConfigError(f"Dataset {name} is not ingested; run `traj ingest {name}`")
        rows.extend(IndexRow.model_validate(value) for value in iter_jsonl(path))
    return rows


def load_trajectory(project: Project, key: str) -> Trajectory:
    return Trajectory.model_validate_json(project.trajectory_path(key, ".json").read_text(encoding="utf-8"))
