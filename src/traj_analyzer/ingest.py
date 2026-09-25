from __future__ import annotations

import fnmatch
import glob
import hashlib
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from traj_analyzer.adapters import load_adapter
from traj_analyzer.project import ConfigError, DatasetConfig, Project, RenderConfig
from traj_analyzer.schema import Step, Trajectory

INDEX_FILE = "index.jsonl"


class IndexRow(BaseModel):
    key: str
    dataset: str
    id: str
    markdown: str
    sha256: str
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
    out = project.dataset_dir(name)
    out.mkdir(parents=True, exist_ok=True)
    rows: dict[str, IndexRow] = {}
    for path in input_files(config, project.root):
        for trajectory in adapter.read(path, name):
            trajectory = trajectory.model_copy(update={"id": safe_id(trajectory.id)})
            if trajectory.id in rows:
                raise ValueError(f"Duplicate trajectory id {trajectory.id} in {path}")
            rows[trajectory.id] = _write(trajectory, out, project.config.render)
    with (out / INDEX_FILE).open("w", encoding="utf-8") as index:
        for row in rows.values():
            index.write(row.model_dump_json() + "\n")
    return len(rows)


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


def _write(trajectory: Trajectory, out: Path, render: RenderConfig) -> IndexRow:
    markdown, n_chunks = render_markdown(trajectory, render)
    (out / f"{trajectory.id}.json").write_text(trajectory.model_dump_json(), encoding="utf-8")
    (out / f"{trajectory.id}.md").write_text(markdown, encoding="utf-8")
    scalars = {k: v for k, v in trajectory.metadata.items() if _is_scalar(v)}
    return IndexRow(
        key=trajectory.key, dataset=trajectory.dataset, id=trajectory.id,
        markdown=f"{trajectory.id}.md",
        sha256=hashlib.sha256(markdown.encode()).hexdigest(),
        n_steps=len(trajectory.steps), n_chunks=n_chunks, chars=len(markdown), metadata=scalars,
    )


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, str | int | float | bool)


def render_markdown(trajectory: Trajectory, render: RenderConfig) -> tuple[str, int]:
    """Render one heading per step, with a `<!-- chunk k -->` marker opening every chunk.

    Chunks break only between steps, so a step longer than the budget forms a chunk of its own.
    """
    header = [f"# {trajectory.key}", ""]
    shown = {k: v for k, v in trajectory.metadata.items()
             if not any(fnmatch.fnmatchcase(k, pattern) for pattern in render.hide_metadata)}
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
        with path.open(encoding="utf-8") as index:
            rows.extend(IndexRow.model_validate_json(line) for line in index)
    return rows


def load_trajectory(project: Project, row: IndexRow) -> Trajectory:
    path = project.dataset_dir(row.dataset) / f"{row.id}.json"
    return Trajectory.model_validate_json(path.read_text(encoding="utf-8"))
