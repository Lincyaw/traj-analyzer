from __future__ import annotations

import fnmatch
import json
import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from traj_analyzer import schema as schema_module
from traj_analyzer.evaluation import CONTEXT_OPERATOR, Evaluation
from traj_analyzer.events import library as rules_library
from traj_analyzer.events import rules as rules_module
from traj_analyzer.features.table import KEY
from traj_analyzer.layers import describe_tables, read_table, table_names
from traj_analyzer.layers import tables as tables_module
from traj_analyzer.mining.functions import CHECKS, SCENARIO
from traj_analyzer.mining.session import WORKSPACE, Session
from traj_analyzer.operators import base as operator_base
from traj_analyzer.operators.catalog import LIBRARY, Group
from traj_analyzer.project import Project
from traj_analyzer.sampling import diverse
from traj_analyzer.vectorize import build_frame

FILES = Path(__file__).parent / "files"
"""Files copied into every workspace."""
TABLES = "tables"
"""Directory of the workspace holding the tables of the readable trajectories."""


def hidden_features(project: Project, groups: list[Group]) -> set[str]:
    """Context features that copy a metadata key `render.hide_metadata` keeps from models."""
    patterns = project.config.render.hide_metadata
    return {name for group in groups for instance in group.instances if instance.ref.name == CONTEXT_OPERATOR
            for name, spec in instance.params.fields.items()  # type: ignore[attr-defined]
            if any(fnmatch.fnmatchcase(spec.key, pattern) for pattern in patterns)}


def prepare_workspace(session: Session, groups: list[Group], evaluation: Evaluation) -> tuple[list[str], str]:
    """Fill the workspace with what the functions of the round start from, and return the starting trajectories
    and the description of the tables.

    Every file is written from the project's state alone, so a round that runs again starts from the same files.
    """
    project, workspace = session.project, session.workspace
    for name in ("notes", "operators", "previous", "cases", TABLES):
        (workspace / name).mkdir(parents=True, exist_ok=True)
    for key in session.readable:
        link = workspace / "trajectories" / f"{key}.md"
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.is_symlink():
            link.symlink_to(project.trajectory_path(key, ".md"))
    tables = _write_tables(session)

    hidden = hidden_features(project, groups)
    frame = build_frame(project, groups, [session.rows[key] for key in session.readable],
                        project.config.sampling.vector_length)
    # The batch and stratum come from the context; the query view carries them too once `traj anomalies` has run.
    shown = [column for column in frame.query.columns if column not in ("batch", "stratum")
             and not any(column == name or column.startswith(f"{name}__") for name in hidden)]
    features = session.context[["batch", "stratum"]].reindex(frame.query.index).join(frame.query[shown])
    features.to_csv(workspace / "features.csv", index_label=KEY)
    roles = evaluation.roles()
    lines = []
    for group in groups:
        for feature in group.features:
            if feature.name in hidden:
                continue
            labels = f" Labels: {', '.join(feature.labels)}." if feature.labels else ""
            lines.append(f"- `{feature.name}` ({feature.type}, {roles[feature.name]}): "
                         f"{feature.description.strip()}{labels}")
    session.hand_off("features.md", "\n".join(lines) + "\n" if lines else "No feature is extracted yet.\n")

    weights = {name: 1.0 for name in frame.columns if roles.get(name) == "distinguishing"}
    keys = list(frame.distance.index)
    count = min(project.config.mining.starting, len(keys))
    if weights:
        everything = list(range(len(keys)))
        picks = diverse("start", count, everything, keys, frame.matrix(weights), project.config.evaluation.seed,
                        everything)
        starting = sorted(pick.key for pick in picks)
    else:
        starting = keys[:count]
    session.hand_off("start.txt", "".join(f"{key}\n" for key in starting))

    session.place("operators/schema.py", Path(schema_module.__file__))
    session.place("operators/base.py", Path(operator_base.__file__))
    session.place("operators/tables.py", Path(tables_module.__file__))
    session.place("operators/rules.py", Path(rules_module.__file__))
    session.place("operators/example_rules.py", Path(rules_library.__file__))
    session.place("operators/example_operator.py", LIBRARY / "stats" / "tool_usage.py")
    session.place("operators/example_actions.py", LIBRARY / "stats" / "actions.py")
    for name, source in ((SCENARIO, project.scenario_path),
                         (f"previous/{CHECKS}", project.mining_dir / str(session.number - 1) / WORKSPACE / CHECKS)):
        target = workspace / name
        if source.is_file():
            shutil.copyfile(source, target)
        else:
            target.unlink(missing_ok=True)
    python = Path(sys.executable)
    readme = (FILES / "README.md").read_text(encoding="utf-8")
    session.hand_off("README.md", readme.replace("{python}", str(python))
                     .replace("{traj}", str(python.with_name("traj"))).replace("{tables}", tables))
    shutil.copyfile(FILES / "query.py", workspace / "query.py")
    return starting, tables


def _write_tables(session: Session) -> str:
    """Copy every table of the project into the workspace, restricted to the readable trajectories, with the
    hidden metadata left out; return the description of the tables."""
    return describe_tables(write_tables(session.project, session.readable, session.workspace / TABLES))


def write_tables(project: Project, keys: list[str], directory: Path) -> dict[str, pa.Schema]:
    """Copy every table of the project into `directory`, restricted to the trajectories of `keys`, with the hidden
    metadata left out; return the schemas of the copies without the key column."""
    directory.mkdir(parents=True, exist_ok=True)
    readable = pa.array(sorted(keys), pa.string())
    patterns = project.config.render.hide_metadata
    schemas: dict[str, pa.Schema] = {}
    for name in table_names(project):
        table = read_table(project, name)
        if KEY not in table.column_names:
            continue
        table = table.filter(pc.is_in(table[KEY], value_set=readable))
        if name == "trajectories":
            shown = [json.dumps({k: v for k, v in json.loads(text).items()
                                 if not any(fnmatch.fnmatchcase(k, p) for p in patterns)},
                                ensure_ascii=False, sort_keys=True)
                     for text in table["metadata"].to_pylist()]
            table = table.set_column(table.schema.get_field_index("metadata"), table.schema.field("metadata"),
                                     pa.array(shown, pa.string()))
        pq.write_table(table, directory / f"{name}.parquet")
        schemas[name] = pa.schema([field for field in table.schema if field.name != KEY])
    return schemas
