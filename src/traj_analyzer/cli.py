from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from dataclasses import asdict
from enum import StrEnum
from importlib.resources import as_file, files
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import ValidationError
from ruamel.yaml import YAML

from traj_analyzer.adapters import discover_adapters
from traj_analyzer.agreement import agreement
from traj_analyzer.extract import extract
from traj_analyzer.features.spec import output_model, render_instruction
from traj_analyzer.features.table import read_group
from traj_analyzer.files import parse_yaml
from traj_analyzer.ingest import ingest, load_index
from traj_analyzer.operators.catalog import discover, load_groups
from traj_analyzer.project import CONFIG_FILE, Config, ConfigError, Project
from traj_analyzer.sampling import SamplerSpec, StrategySpec, enrich, load_sampler, sample, save
from traj_analyzer.studies import contrast_pairs, list_studies, load_study, outcome_features, population, screen
from traj_analyzer.vectorize import Frame, build_frame

DESCRIPTION = """\
Batch analysis of LLM trajectories.
Every command prints one JSON document to stdout, and logs go to stderr.
Exit codes: 0 success, 1 runtime error, 2 usage or configuration error.
"""

# Packaging drops files whose names start with a dot, so templates store these without it.
_DOTTED = {"gitignore": ".gitignore", "claude": ".claude"}

app = typer.Typer(help=DESCRIPTION, no_args_is_help=True, add_completion=False, pretty_exceptions_enable=False)
adapters_app = typer.Typer(help="Data sources.", no_args_is_help=True)
operators_app = typer.Typer(help="Browse, enable and disable operators.", no_args_is_help=True)
study_app = typer.Typer(help="Questions about which trajectories succeed and which features tell why.",
                        no_args_is_help=True)
app.add_typer(adapters_app, name="adapters")
app.add_typer(operators_app, name="operators")
app.add_typer(study_app, name="study")

Datasets = Annotated[list[str] | None, typer.Option("--dataset", help="Restrict to this dataset; repeatable.")]


class OperatorKind(StrEnum):
    code = "code"
    llm = "llm"


class DiscoverMethod(StrEnum):
    diversity = "diversity"
    random = "random"


class TableFormat(StrEnum):
    json = "json"
    csv = "csv"
    describe = "describe"


def _emit(data: Any) -> None:
    json.dump(data, sys.stdout, ensure_ascii=False, indent=1)
    sys.stdout.write("\n")


@app.command("init")
def cmd_init(directory: Annotated[str, typer.Argument(help="Directory of the new project.")],
             force: Annotated[bool, typer.Option(help="Overwrite an existing traj.yaml.")] = False) -> None:
    """Create an analysis project skeleton."""
    target = Path(directory).expanduser().resolve()
    if (target / CONFIG_FILE).exists() and not force:
        raise ConfigError(f"{target / CONFIG_FILE} already exists; use --force to overwrite")
    written = []
    with as_file(files("traj_analyzer") / "templates" / "project") as root:
        for path in sorted(root.rglob("*")):
            if path.is_dir() or "__pycache__" in path.parts:
                continue
            relative = Path(*(_DOTTED.get(p, p) for p in path.relative_to(root).parts))
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            written.append(str(relative))
    _emit({"project": str(target), "files": written})


@app.command("ingest")
def cmd_ingest(datasets: Annotated[list[str] | None, typer.Argument(help="Datasets; default all.")] = None) -> None:
    """Read raw data into the unified form."""
    _emit({"ingested": ingest(Project.find(), datasets or None)})


@app.command("validate")
def cmd_validate(instructions: Annotated[bool, typer.Option(help="Include rendered instructions.")] = False) -> None:
    """Check traj.yaml, enabled operators, samplers and studies."""
    project = Project.find()
    groups = load_groups(project)
    report: dict[str, Any] = {"datasets": list(project.config.datasets), "groups": []}
    for group in groups:
        entry: dict[str, Any] = {
            "group": group.name, "kind": group.kind, "operators": [i.id for i in group.instances],
            "features": [f.name for f in group.features],
        }
        if not group.is_code:
            entry["output_schema"] = output_model(group).model_json_schema()
            if instructions:
                entry["instruction"] = render_instruction(group)
        report["groups"].append(entry)
    samplers = sorted(p.stem for p in project.samplers_dir.glob("*.yaml"))
    for name in samplers:
        load_sampler(project, name)
    report["samplers"] = samplers
    features = {f.name for g in groups for f in g.features}
    for name in list_studies(project):
        study = load_study(project, name)
        named = [*study.by, *([study.pairs.within] if study.pairs else [])]
        unknown = sorted(set(named) - features)
        if unknown:
            raise ConfigError(f"Study {name} names features that are not enabled: {unknown}")
    report["studies"] = list_studies(project)
    _emit(report)


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
        if kind and kind.value != operator.kind:
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
    """Apply `change` to the operators list of traj.yaml, validate the result, then write it back."""
    path = project.root / CONFIG_FILE
    editor = YAML()
    editor.preserve_quotes = True
    editor.indent(mapping=2, sequence=4, offset=2)
    editor.representer.add_representer(
        type(None), lambda representer, _: representer.represent_scalar("tag:yaml.org,2002:null", "null"))
    document = editor.load(path)
    if "operators" not in document:
        document["operators"] = []
    change(document["operators"])
    config = Config.model_validate(json.loads(json.dumps(document)))
    load_groups(Project(project.root, config))
    editor.dump(document, path)
    return [use.model_dump(by_alias=True, exclude_none=True) for use in config.operators]


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


def _frame(project: Project, datasets: list[str] | None, groups: list[str] | None = None) -> Frame:
    return build_frame(project, load_groups(project, groups), load_index(project, datasets),
                       project.config.sampling.vector_length)


@app.command("discover")
def cmd_discover(
    n: Annotated[int, typer.Option(help="Trajectories to pick.")] = 20,
    seed: int = 0,
    dataset: Datasets = None,
    method: Annotated[DiscoverMethod, typer.Option(help="How to pick.")] = DiscoverMethod.diversity,
    group: Annotated[list[str] | None, typer.Option(help="Groups whose features drive diversity; default all code "
                                                         "groups; repeatable.")] = None,
) -> None:
    """Pick trajectories to read when proposing features, sampling over the features of code groups."""
    project = Project.find()
    groups = group or [g.name for g in load_groups(project) if g.is_code]
    frame = _frame(project, dataset, groups)
    if not frame.columns:
        raise ConfigError(f"Groups {groups} have no extracted values; run `traj extract` for them first")
    spec = SamplerSpec(name="discover", budget=n, seed=seed, datasets=dataset,
                       strategies=[StrategySpec(kind=method.value, quota="rest")])
    result = sample(spec, frame, load_groups(project, groups))
    rows = {row.key: row for row in load_index(project, dataset)}
    _emit({"method": method.value, "population": result.population, "trajectories": [
        {"key": pick.key, "markdown": str(project.trajectory_path(pick.key, ".md")),
         "n_steps": rows[pick.key].n_steps, "chars": rows[pick.key].chars, "reason": pick.reason,
         "metadata": rows[pick.key].metadata}
        for pick in result.picks
    ]})


@app.command("extract")
def cmd_extract(
    group: Annotated[list[str] | None, typer.Option(help="Execution group; repeatable.")] = None,
    dataset: Datasets = None,
    key: Annotated[list[str] | None, typer.Option(help="Trajectory key; repeatable.")] = None,
    limit: Annotated[int | None, typer.Option(help="At most this many trajectories.")] = None,
    workers: Annotated[int | None, typer.Option(help="Worker processes for LLM groups.")] = None,
    dry_run: Annotated[bool, typer.Option(help="Only look up the cache.")] = False,
) -> None:
    """Compute features."""
    reports = extract(Project.find(), groups=group, datasets=dataset, keys=key, limit=limit, workers=workers,
                      dry_run=dry_run)
    _emit({"dry_run": dry_run, "groups": [asdict(r) for r in reports]})


@app.command("table")
def cmd_table(
    dataset: Datasets = None,
    column: Annotated[list[str] | None, typer.Option(help="Column to print; repeatable.")] = None,
    format: Annotated[TableFormat, typer.Option(help="Output format.")] = TableFormat.json,
) -> None:
    """Print the wide feature table."""
    table = _frame(Project.find(), dataset).query
    if column:
        missing = sorted(set(column) - set(table.columns))
        if missing:
            raise ConfigError(f"Unknown columns: {missing}")
        table = table[column]
    if format is TableFormat.csv:
        table.to_csv(sys.stdout)
    elif format is TableFormat.describe:
        _emit(json.loads(table.describe(include="all").to_json()))
    else:
        _emit(json.loads(table.reset_index().to_json(orient="records", force_ascii=False)))


@app.command("sample")
def cmd_sample(sampler: Annotated[str, typer.Option(help="Sampler name under samplers/.")] = "default",
               budget: Annotated[int | None, typer.Option(help="Override the sampler budget.")] = None) -> None:
    """Select trajectories for review."""
    project = Project.find()
    spec = load_sampler(project, sampler)
    if budget:
        spec = spec.model_copy(update={"budget": budget})
    frame = _frame(project, spec.datasets)
    result = sample(spec, frame, load_groups(project))
    selection = {
        "population": result.population,
        "strategies": [report.model_dump() for report in result.strategies],
        "picks": [enrich(project, p, frame) for p in result.picks],
    }
    path = save(project, spec, selection)
    _emit({"selection": str(path), **selection})


@app.command("show")
def cmd_show(key: str, cat: Annotated[bool, typer.Option(help="Print the rendered Markdown.")] = False) -> None:
    """One trajectory's rendered file and features."""
    project = Project.find()
    path = project.trajectory_path(key, ".md")
    if not path.is_file():
        raise ConfigError(f"No trajectory {key}")
    if cat:
        sys.stdout.write(path.read_text(encoding="utf-8"))
        return
    features = {}
    for group in load_groups(project):
        for row in read_group(project, group.name):
            if row.key == key:
                features[row.feature] = row.model_dump(exclude={"key", "feature"})
    _emit({"key": key, "markdown": str(path), "features": features})


@app.command("status")
def cmd_status() -> None:
    """Trajectories per dataset, and per group how many have rows in the feature table, by status.

    The table holds what the last `traj extract` wrote; after changing data or operators, run it again.
    """
    project = Project.find()
    index = load_index(project)
    keys = {row.key for row in index}
    groups = {}
    for group in load_groups(project):
        statuses: dict[str, str] = {}
        for row in read_group(project, group.name):
            if row.key in keys and statuses.get(row.key, "ok") == "ok":
                statuses[row.key] = row.status
        groups[group.name] = {"operators": [i.id for i in group.instances], "extracted": len(statuses),
                              "missing": len(keys) - len(statuses), **Counter(statuses.values())}
    _emit({"project": str(project.root), "datasets": dict(Counter(row.dataset for row in index)),
           "groups": groups})


@study_app.command("list")
def cmd_study_list() -> None:
    """Studies under studies/."""
    project = Project.find()
    _emit({"studies": [load_study(project, name).model_dump() for name in list_studies(project)]})


@study_app.command("pairs")
def cmd_study_pairs(name: str, n: Annotated[int | None, typer.Option(help="Override pairs.n.")] = None,
                    seed: int = 0) -> None:
    """Contrast pairs: trajectories sharing the `within` feature, one that succeeded and one that did not."""
    project = Project.find()
    study = load_study(project, name)
    if study.pairs is None:
        raise ConfigError(f"Study {name} has no pairs section")
    spec = study.pairs.model_copy(update={"n": n}) if n else study.pairs
    groups = load_groups(project)
    data = population(study, _frame(project, study.datasets), groups)
    pairs = contrast_pairs(spec, data, seed)
    for pair in pairs:
        for side in ["succeeded", "failed"]:
            pair[side] = {"key": pair[side], "markdown": str(project.trajectory_path(pair[side], ".md"))}
    _emit({"study": name, "population": len(data.success), "success_rate": _rate(data.success), "pairs": pairs})


def _rate(success: Any) -> float | None:
    return None if success.notna().sum() == 0 else round(float(success.dropna().astype(float).mean()), 3)


@study_app.command("screen")
def cmd_study_screen(
    name: str,
    feature: Annotated[list[str] | None, typer.Option(help="Feature to screen; default every enabled feature "
                                                           "except outcomes and by; repeatable.")] = None,
) -> None:
    """Check features on the study population: constant, redundant, varying by a `by` feature, or explaining
    success once the `by` features are held fixed."""
    project = Project.find()
    study = load_study(project, name)
    groups = load_groups(project)
    data = population(study, _frame(project, study.datasets), groups)
    outcomes = outcome_features(study, groups)
    features = feature or [f for f in data.frame.columns if f not in outcomes and f not in study.by]
    report = screen(study, data, features, outcomes)
    verdicts: dict[str, list[str]] = {}
    for feature_name, entry in report.items():
        verdicts.setdefault(entry["verdict"], []).append(feature_name)
    _emit({"study": name, "population": len(data.success), "success_rate": _rate(data.success),
           "rules": study.rules.model_dump(), "verdicts": verdicts, "features": report})


@app.command("agreement")
def cmd_agreement(reference: Annotated[str, typer.Argument(
        help="JSON {key: {feature: value}} written by a careful reader.")]) -> None:
    """Compare reference answers with the extracted features."""
    project = Project.find()
    answers = json.loads(Path(reference).read_text(encoding="utf-8"))
    specs = {}
    rows = {}
    for group in load_groups(project):
        specs.update({f.name: f for f in group.features})
        for row in read_group(project, group.name):
            if row.key in answers:
                rows[(row.key, row.feature)] = row
    _emit({"reference": reference, "features": agreement(answers, specs, rows)})


def main(argv: list[str] | None = None) -> int:
    try:
        app(args=argv, prog_name="traj")
    except SystemExit as exit:
        return exit.code if isinstance(exit.code, int) else 0
    except (ConfigError, ValidationError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
