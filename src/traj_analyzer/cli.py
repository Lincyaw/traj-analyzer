from __future__ import annotations

import json
import sys
from dataclasses import asdict
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import typer
import uvicorn
from pydantic import ValidationError

from traj_analyzer.adapters import discover_adapters
from traj_analyzer.evaluation import evaluate, load_evaluation, reference_keys
from traj_analyzer.extract import coverage, extract
from traj_analyzer.files import parse_yaml
from traj_analyzer.ingest import ingest, load_index
from traj_analyzer.mining import mine
from traj_analyzer.operators.catalog import discover, edit_operators, load_groups
from traj_analyzer.paths import PathTree, describe
from traj_analyzer.project import ConfigError, Project
from traj_analyzer.sampling import Reference, enrich, load_sampler, needs_reference, sample, save
from traj_analyzer.scaffold import InitError, init_project
from traj_analyzer.vectorize import Frame, build_frame
from traj_analyzer.viewer.server import create_app

DESCRIPTION = """\
Batch analysis of LLM trajectories.
Every command except `table` and `view` prints one JSON document to stdout, and logs go to stderr.
Exit codes: 0 success, 1 runtime error, 2 usage or configuration error.
"""

app = typer.Typer(help=DESCRIPTION, no_args_is_help=True, add_completion=False, pretty_exceptions_enable=False)
adapters_app = typer.Typer(help="Data sources.", no_args_is_help=True)
operators_app = typer.Typer(help="Browse, enable and disable operators.", no_args_is_help=True)
app.add_typer(adapters_app, name="adapters")
app.add_typer(operators_app, name="operators")

Datasets = Annotated[list[str] | None, typer.Option("--dataset", help="Restrict to this dataset; repeatable.")]


class OperatorKind(StrEnum):
    code = "code"
    llm = "llm"


def _emit(data: Any) -> None:
    json.dump(data, sys.stdout, ensure_ascii=False, indent=1)
    sys.stdout.write("\n")


@app.command("init")
def cmd_init(directory: Annotated[str, typer.Argument(help="Directory of the new project.")],
             force: Annotated[bool, typer.Option(help="Overwrite an existing traj.yaml.")] = False) -> None:
    """Create an analysis project with shared skills for Codex and Claude Code."""
    _emit(init_project(Path(directory), force=force))


@app.command("ingest")
def cmd_ingest(datasets: Annotated[list[str] | None, typer.Argument(help="Datasets; default all.")] = None) -> None:
    """Read raw data into the unified form."""
    _emit({"ingested": ingest(Project.find(), datasets or None)})


def _enabled(project: Project) -> dict[str, list[str]]:
    enabled: dict[str, list[str]] = {}
    for use in project.config.operators:
        enabled.setdefault(use.use, []).append(use.alias or use.use)
    return enabled


@adapters_app.command("list")
def cmd_adapters_list() -> None:
    """Adapters of the package and the project, and the datasets using them."""
    project = Project.find()
    used = {name: config.adapter for name, config in project.config.datasets.items()}
    _emit({"adapters": [
        {"name": ref.name, "origin": ref.origin, "path": str(ref.path),
         "description": (ref.adapter.__doc__ or "").strip().splitlines()[0] if ref.adapter.__doc__ else "",
         "used_by": sorted(d for d, adapter in used.items() if adapter == ref.name)}
        for ref in discover_adapters(project.root).values()
    ]})


@operators_app.command("list")
def cmd_operators_list(tag: Annotated[str | None, typer.Option(help="Only operators with this tag.")] = None,
                       kind: Annotated[OperatorKind | None, typer.Option(help="Only this kind.")] = None) -> None:
    """Operators in the built-in library and the project."""
    project = Project.find()
    enabled = _enabled(project)
    rows = []
    for name, ref in discover(project).items():
        operator = ref.operator
        if tag and tag not in operator.tags:
            continue
        if kind and kind != operator.kind:
            continue
        rows.append({
            "name": name, "kind": operator.kind, "origin": ref.origin, "tags": list(operator.tags),
            "description": operator.description,
            "outputs": [f.name for f in operator.outputs(operator.params())],
            "enabled_as": enabled.get(name, []),
        })
    _emit({"operators": rows})


@operators_app.command("show")
def cmd_operators_show(name: str) -> None:
    """One operator's parameters, outputs and guidance."""
    project = Project.find()
    refs = discover(project)
    if name not in refs:
        raise ConfigError(f"Unknown operator {name}")
    ref = refs[name]
    operator = ref.operator
    params = operator.params()
    _emit({
        "name": ref.name, "kind": operator.kind, "origin": ref.origin, "path": str(ref.path),
        "tags": list(operator.tags), "description": operator.description,
        "requires": None if operator.requires is None else operator.requires.__name__,
        "params_schema": operator.params.model_json_schema(), "params_default": params.model_dump(mode="json"),
        "outputs": [f.model_dump(mode="json") for f in operator.outputs(params)],
        "guidance": None if operator.guidance is None else operator.guidance(params),
        "enabled_as": _enabled(project).get(ref.name, []),
    })


def _edit_operators(project: Project, change: Any) -> list[dict[str, Any]]:
    """Apply `change` to the operators list of traj.yaml and return the list as written."""
    edited = edit_operators(project, change)
    return [use.model_dump(by_alias=True, exclude_none=True) for use in edited.config.operators]


@operators_app.command("enable")
def cmd_operators_enable(
    name: str,
    alias: Annotated[str | None, typer.Option("--as", help="Instance name for enabling one operator twice.")] = None,
    prefix: Annotated[str | None, typer.Option(help="Prepended as <prefix>_ to every feature name.")] = None,
    call: Annotated[str | None, typer.Option(help="Call group for LLM operators.")] = None,
    param: Annotated[list[str] | None, typer.Option(help="key=value, value parsed as YAML; repeatable.")] = None,
) -> None:
    """Add an operator to traj.yaml, keeping the file's comments and layout."""
    project = Project.find()
    entry: dict[str, Any] = {"use": name}
    for option, value in (("as", alias), ("prefix", prefix), ("call", call)):
        if value:
            entry[option] = value
    params = {}
    for item in param or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise ConfigError(f"--param must be key=value: {item}")
        params[key] = parse_yaml(value)
    if params:
        entry["params"] = params
    _emit({"operators": _edit_operators(project, lambda ops: ops.append(entry))})


@operators_app.command("disable")
def cmd_operators_disable(name: str) -> None:
    """Remove the operators with this name or instance name from traj.yaml."""
    project = Project.find()

    def remove(ops: list[Any]) -> None:
        keep = [op for op in ops if name not in (op["use"], op.get("as"))]
        if len(keep) == len(ops):
            raise ConfigError(f"No enabled operator is named {name}")
        ops[:] = keep

    _emit({"operators": _edit_operators(project, remove)})


def _frame(project: Project, datasets: list[str] | None) -> Frame:
    return build_frame(project, load_groups(project), load_index(project, datasets),
                       project.config.sampling.vector_length)


@app.command("extract")
def cmd_extract(
    group: Annotated[list[str] | None, typer.Option(help="Execution group; repeatable.")] = None,
    dataset: Datasets = None,
    key: Annotated[list[str] | None, typer.Option(help="Trajectory key; repeatable.")] = None,
    limit: Annotated[int | None, typer.Option(help="At most this many trajectories.")] = None,
    workers: Annotated[int | None, typer.Option(help="Worker processes for LLM groups.")] = None,
    dry_run: Annotated[bool, typer.Option(help="Only look up the cache.")] = False,
) -> None:
    """Check traj.yaml, the operators and every sampler, compute features, and report coverage per group."""
    project = Project.find()
    for path in sorted(project.samplers_dir.glob("*.yaml")):
        load_sampler(project, path.stem)
    reports = extract(project, groups=group, datasets=dataset, keys=key, limit=limit, workers=workers,
                      dry_run=dry_run)
    _emit({"dry_run": dry_run, "groups": [asdict(r) for r in reports], "coverage": coverage(project)})


@app.command("table")
def cmd_table(
    dataset: Datasets = None,
    column: Annotated[list[str] | None, typer.Option(help="Column to print; repeatable.")] = None,
) -> None:
    """Print the wide feature table as CSV, one row per trajectory."""
    table = _frame(Project.find(), dataset).query
    if column:
        missing = sorted(set(column) - set(table.columns))
        if missing:
            raise ConfigError(f"Unknown columns: {missing}")
        table = table[column]
    table.to_csv(sys.stdout)


@app.command("sample")
def cmd_sample(sampler: Annotated[str, typer.Option(help="Sampler name under samplers/.")] = "default",
               budget: Annotated[int | None, typer.Option(help="Override the sampler budget.")] = None) -> None:
    """Select trajectories for review."""
    project = Project.find()
    spec = load_sampler(project, sampler)
    if budget:
        spec = spec.model_copy(update={"budget": budget})
    frame = _frame(project, spec.datasets)
    reference = None
    if needs_reference(spec):
        reference = Reference(keys=reference_keys(project, load_index(project)), evaluation=load_evaluation(project))
    result = sample(spec, frame, load_groups(project), reference)
    selection = {
        "population": result.population,
        "strategies": [report.model_dump() for report in result.strategies],
        "picks": [enrich(project, p, frame) for p in result.picks],
    }
    path = save(project, spec, selection)
    _emit({"selection": str(path), **selection})


@app.command("tree")
def cmd_tree(
    feature: Annotated[str, typer.Argument(help="A paths feature.")],
    dataset: Datasets = None,
    support: Annotated[float | None, typer.Option(help="Share of trajectories a node needs; default from "
                                                       "sampling.path_support.")] = None,
    by: Annotated[str | None, typer.Option(help="Column whose groups are compared at every node.")] = None,
    within: Annotated[str | None, typer.Option(help="Column held fixed while comparing --by.")] = None,
    top: Annotated[int, typer.Option(min=1, help="Nodes to print, most used first.")] = 40,
) -> None:
    """Show a paths feature as a tree: the size of every level, where events are cut, and the most used nodes."""
    project = Project.find()
    frame = _frame(project, dataset)
    if feature not in frame.paths:
        raise ConfigError(f"{feature} is not a paths feature with extracted values")
    unknown = sorted({c for c in (by, within) if c} - set(frame.query.columns))
    if unknown:
        raise ConfigError(f"Unknown columns: {unknown}")
    if within and not by:
        raise ConfigError("--within needs --by")
    values = frame.paths[feature]
    tree = PathTree.build(frame.levels[feature], values, support or project.config.sampling.path_support)
    _emit({"feature": feature, **describe(tree, values, frame.query[by] if by else None,
                                          frame.query[within] if within else None, top)})


Workers = Annotated[int | None, typer.Option(help="Worker processes for LLM calls.")]


@app.command("evaluate")
def cmd_evaluate(workers: Workers = None) -> None:
    """Measure noise, variation and dependence of every enabled feature, and assign each its role."""
    _emit(evaluate(Project.find(), workers=workers).model_dump(mode="json"))


@app.command("mine")
def cmd_mine(rounds: Annotated[int, typer.Option(min=1, help="At most this many rounds.")] = 1,
             workers: Workers = None) -> None:
    """Propose features from the trajectories, check them, and enable the accepted ones."""
    reports = mine(Project.find(), rounds=rounds, workers=workers)
    _emit({"rounds": [report.model_dump(mode="json") for report in reports]})


@app.command("view")
def cmd_view(host: Annotated[str, typer.Option(help="Address to listen on.")] = "127.0.0.1",
             port: Annotated[int, typer.Option(help="Port to listen on.")] = 8000) -> None:
    """Browse the feature table in a web page: page through, search and filter every extracted group."""
    uvicorn.run(create_app(Project.find()), host=host, port=port)


def main(argv: list[str] | None = None) -> int:
    try:
        app(args=argv, prog_name="traj")
    except SystemExit as exit:
        return exit.code if isinstance(exit.code, int) else 0
    except (ConfigError, InitError, ValidationError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
