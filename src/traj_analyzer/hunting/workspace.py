from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from traj_analyzer import schema as schema_module
from traj_analyzer.events import library as rules_library
from traj_analyzer.events import rules as rules_module
from traj_analyzer.hunting.functions import NEED, SCENARIO
from traj_analyzer.hunting.probes import CONTENTS, CONTENTS_SCHEMA, contents_rows, readable_tables
from traj_analyzer.hunting.spec import HuntSpec
from traj_analyzer.ingest import load_trajectory, render_markdown
from traj_analyzer.layers import describe_tables
from traj_analyzer.layers import tables as tables_module
from traj_analyzer.layers.tables import with_key
from traj_analyzer.mining.workspace import FILES as MINING_FILES
from traj_analyzer.mining.workspace import write_tables
from traj_analyzer.operators import base as operator_base
from traj_analyzer.operators.catalog import LIBRARY
from traj_analyzer.project import Project

FILES = Path(__file__).parent / "files"
"""Files copied into every workspace of a hunt."""
TABLES = "tables"
OPERATOR_FILES = {
    "operators/schema.py": Path(schema_module.__file__),
    "operators/base.py": Path(operator_base.__file__),
    "operators/tables.py": Path(tables_module.__file__),
    "operators/rules.py": Path(rules_module.__file__),
    "operators/example_rules.py": Path(rules_library.__file__),
    "operators/example_operator.py": LIBRARY / "events" / "generic.py",
}
"""What a code probe imports, and the operators of the built-in library written the same way."""


def write_file(workspace: Path, name: str, text: str) -> None:
    path = workspace / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def prepare(project: Project, workspace: Path, name: str, spec: HuntSpec, keys: list[str]) -> set[str]:
    """Fill the workspace of the hunt `name` with what every round starts from: the trajectories, the tables, the
    need, the files a code probe imports, and the description of it all; return the names of the files written.

    Every file is written from the project's state alone, so a hunt that runs again starts from the same files.
    """
    for directory in ("notes", "operators", "probes", "rounds", TABLES):
        (workspace / directory).mkdir(parents=True, exist_ok=True)
    for key in keys:
        link = workspace / "trajectories" / f"{key}.md"
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.is_symlink():
            link.symlink_to(project.trajectory_path(key, ".md"))
    schemas = write_tables(project, keys, workspace / TABLES)
    write_contents(project, keys, workspace / TABLES / f"{CONTENTS}.parquet")
    schemas[CONTENTS] = CONTENTS_SCHEMA
    write_file(workspace, NEED, spec.need.strip() + "\n")
    written = {NEED, "README.md", *OPERATOR_FILES}
    if project.scenario_path.is_file():
        shutil.copyfile(project.scenario_path, workspace / SCENARIO)
        written.add(SCENARIO)
    else:
        (workspace / SCENARIO).unlink(missing_ok=True)
    for file, source in OPERATOR_FILES.items():
        shutil.copyfile(source, workspace / file)
    python = Path(sys.executable)
    readable = ", ".join(f"`{table}`" for table in sorted(readable_tables(project) & set(schemas)))
    readme = (FILES / "README.md").read_text(encoding="utf-8")
    write_file(workspace, "README.md", readme.replace("{python}", str(python))
               .replace("{traj}", str(python.with_name("traj"))).replace("{hunt}", name)
               .replace("{readable}", readable).replace("{tables}", describe_tables(schemas)))
    shutil.copyfile(MINING_FILES / "query.py", workspace / "query.py")
    return written


def write_contents(project: Project, keys: list[str], path: Path) -> None:
    """The whole text of every step of these trajectories, which the steps table and the trajectory files cut."""
    rows = [{"key": key, **row} for key in keys for row in contents_rows(load_trajectory(project, key))]
    pq.write_table(pa.Table.from_pylist(rows, schema=with_key(CONTENTS_SCHEMA)), path)


def write_case(project: Project, workspace: Path, directory: str, key: str) -> str:
    """Write the trajectory whole, with no step cut, as a case file under `directory`, and return its path."""
    trajectory = load_trajectory(project, key)
    longest = max((len(step.content) for step in trajectory.steps), default=1)
    render = project.config.render.model_copy(update={"max_step_chars": max(longest, 1)})
    text, _ = render_markdown(trajectory, render, project.config.datasets[trajectory.dataset].description)
    file = f"{directory}/{key}.md"
    write_file(workspace, file, text)
    return file
