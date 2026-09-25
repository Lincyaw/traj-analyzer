from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from dataclasses import asdict
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from ruamel.yaml import YAML

from traj_analyzer.adapters import discover_adapters
from traj_analyzer.extract import extract
from traj_analyzer.features.spec import output_model, render_instruction
from traj_analyzer.features.table import read_group
from traj_analyzer.files import parse_yaml
from traj_analyzer.ingest import ingest, load_index
from traj_analyzer.operators.catalog import discover, load_groups
from traj_analyzer.project import CONFIG_FILE, Config, ConfigError, Project
from traj_analyzer.sampling import SamplerSpec, StrategySpec, enrich, load_sampler, sample, save
from traj_analyzer.vectorize import Frame, build_frame

DESCRIPTION = """\
Batch analysis of LLM trajectories.
Every command prints one JSON document to stdout, and logs go to stderr.
Exit codes: 0 success, 1 runtime error, 2 usage or configuration error.
"""

# Packaging drops files whose names start with a dot, so templates store these without it.
_DOTTED = {"gitignore": ".gitignore", "claude": ".claude"}


def _emit(data: Any) -> None:
    json.dump(data, sys.stdout, ensure_ascii=False, indent=1)
    sys.stdout.write("\n")


def cmd_init(args: argparse.Namespace) -> None:
    target = Path(args.dir).expanduser().resolve()
    if (target / CONFIG_FILE).exists() and not args.force:
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


def cmd_ingest(args: argparse.Namespace) -> None:
    _emit({"ingested": ingest(Project.find(), args.datasets or None)})


def cmd_validate(args: argparse.Namespace) -> None:
    project = Project.find()
    report: dict[str, Any] = {"datasets": list(project.config.datasets), "groups": []}
    for group in load_groups(project):
        entry: dict[str, Any] = {
            "group": group.name, "kind": group.kind, "operators": [i.id for i in group.instances],
            "features": [f.name for f in group.features],
        }
        if not group.is_code:
            entry["output_schema"] = output_model(group).model_json_schema()
            if args.instructions:
                entry["instruction"] = render_instruction(group)
        report["groups"].append(entry)
    samplers = sorted(p.stem for p in project.samplers_dir.glob("*.yaml"))
    for name in samplers:
        load_sampler(project, name)
    report["samplers"] = samplers
    _emit(report)


def _enabled(project: Project) -> dict[str, list[str]]:
    enabled: dict[str, list[str]] = {}
    for use in project.config.operators:
        enabled.setdefault(use.use, []).append(use.alias or use.use)
    return enabled


def cmd_adapters_list(args: argparse.Namespace) -> None:
    project = Project.find()
    used = {name: config.adapter for name, config in project.config.datasets.items()}
    _emit({"adapters": [
        {"name": ref.name, "origin": ref.origin, "path": str(ref.path),
         "description": (ref.adapter.__doc__ or "").strip().splitlines()[0] if ref.adapter.__doc__ else "",
         "used_by": sorted(d for d, adapter in used.items() if adapter == ref.name)}
        for ref in discover_adapters(project.root).values()
    ]})


def cmd_operators_list(args: argparse.Namespace) -> None:
    project = Project.find()
    enabled = _enabled(project)
    rows = []
    for name, ref in discover(project).items():
        operator = ref.operator
        if args.tag and args.tag not in operator.tags:
            continue
        if args.kind and args.kind != operator.kind:
            continue
        rows.append({
            "name": name, "kind": operator.kind, "origin": ref.origin, "tags": list(operator.tags),
            "description": operator.description,
            "outputs": [f.name for f in operator.outputs(operator.params())],
            "enabled_as": enabled.get(name, []),
        })
    _emit({"operators": rows})


def cmd_operators_show(args: argparse.Namespace) -> None:
    project = Project.find()
    refs = discover(project)
    if args.name not in refs:
        raise ConfigError(f"Unknown operator {args.name}")
    ref = refs[args.name]
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


def cmd_operators_enable(args: argparse.Namespace) -> None:
    project = Project.find()
    entry: dict[str, Any] = {"use": args.name}
    for option, value in (("as", args.alias), ("prefix", args.prefix), ("call", args.call)):
        if value:
            entry[option] = value
    params = {}
    for item in args.param or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise ConfigError(f"--param must be key=value: {item}")
        params[key] = parse_yaml(value)
    if params:
        entry["params"] = params
    _emit({"operators": _edit_operators(project, lambda ops: ops.append(entry))})


def cmd_operators_disable(args: argparse.Namespace) -> None:
    project = Project.find()

    def remove(ops: list[Any]) -> None:
        keep = [op for op in ops if args.name not in (op["use"], op.get("as"))]
        if len(keep) == len(ops):
            raise ConfigError(f"No enabled operator is named {args.name}")
        ops[:] = keep

    _emit({"operators": _edit_operators(project, remove)})


def _frame(project: Project, datasets: list[str] | None, groups: list[str] | None = None) -> Frame:
    return build_frame(project, load_groups(project, groups), load_index(project, datasets),
                       project.config.sampling.vector_length)


def cmd_discover(args: argparse.Namespace) -> None:
    """Sample with one strategy over the features of code groups, which are cheap to extract for everything."""
    project = Project.find()
    groups = args.group or [g.name for g in load_groups(project) if g.is_code]
    frame = _frame(project, args.dataset, groups)
    if not frame.columns:
        raise ConfigError(f"Groups {groups} have no extracted values; run `traj extract` for them first")
    spec = SamplerSpec(name="discover", budget=args.n, seed=args.seed, datasets=args.dataset,
                       strategies=[StrategySpec(kind=args.method, quota="rest")])
    result = sample(spec, frame, load_groups(project, groups))
    rows = {row.key: row for row in load_index(project, args.dataset)}
    _emit({"method": args.method, "population": result.population, "trajectories": [
        {"key": pick.key, "markdown": str(project.trajectory_path(pick.key, ".md")),
         "n_steps": rows[pick.key].n_steps, "chars": rows[pick.key].chars, "reason": pick.reason,
         "metadata": rows[pick.key].metadata}
        for pick in result.picks
    ]})


def cmd_extract(args: argparse.Namespace) -> None:
    reports = extract(
        Project.find(), groups=args.group, datasets=args.dataset, keys=args.key,
        limit=args.limit, workers=args.workers, dry_run=args.dry_run,
    )
    _emit({"dry_run": args.dry_run, "groups": [asdict(r) for r in reports]})


def cmd_table(args: argparse.Namespace) -> None:
    table = _frame(Project.find(), args.dataset).query
    if args.columns:
        missing = sorted(set(args.columns) - set(table.columns))
        if missing:
            raise ConfigError(f"Unknown columns: {missing}")
        table = table[args.columns]
    if args.format == "csv":
        table.to_csv(sys.stdout)
    elif args.format == "describe":
        _emit(json.loads(table.describe(include="all").to_json()))
    else:
        _emit(json.loads(table.reset_index().to_json(orient="records", force_ascii=False)))


def cmd_sample(args: argparse.Namespace) -> None:
    project = Project.find()
    spec = load_sampler(project, args.sampler)
    if args.budget:
        spec = spec.model_copy(update={"budget": args.budget})
    frame = _frame(project, spec.datasets)
    result = sample(spec, frame, load_groups(project))
    selection = {
        "population": result.population,
        "strategies": [report.model_dump() for report in result.strategies],
        "picks": [enrich(project, p, frame) for p in result.picks],
    }
    path = save(project, spec, selection)
    _emit({"selection": str(path), **selection})


def cmd_show(args: argparse.Namespace) -> None:
    project = Project.find()
    path = project.trajectory_path(args.key, ".md")
    if not path.is_file():
        raise ConfigError(f"No trajectory {args.key}")
    if args.cat:
        sys.stdout.write(path.read_text(encoding="utf-8"))
        return
    features = {}
    for group in load_groups(project):
        for row in read_group(project, group.name):
            if row.key == args.key:
                features[row.feature] = row.model_dump(exclude={"key", "feature"})
    _emit({"key": args.key, "markdown": str(path), "features": features})


def cmd_status(args: argparse.Namespace) -> None:
    """Trajectories per dataset, and per group how many trajectories have rows in the feature table, by status.

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


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="traj", description=DESCRIPTION,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = root.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="create an analysis project skeleton")
    p.add_argument("dir")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_init)

    p = sub.add_parser("ingest", help="read raw data into the unified form")
    p.add_argument("datasets", nargs="*")
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("validate", help="check traj.yaml, enabled operators and samplers")
    p.add_argument("--instructions", action="store_true", help="include rendered instructions")
    p.set_defaults(fn=cmd_validate)

    p = sub.add_parser("adapters", help="data sources")
    ads = p.add_subparsers(dest="adapters_command", required=True)
    q = ads.add_parser("list", help="adapters of the package and the project")
    q.set_defaults(fn=cmd_adapters_list)

    p = sub.add_parser("operators", help="browse, enable and disable operators")
    ops = p.add_subparsers(dest="operators_command", required=True)
    q = ops.add_parser("list", help="operators in the built-in library and the project")
    q.add_argument("--tag")
    q.add_argument("--kind", choices=["code", "llm"])
    q.set_defaults(fn=cmd_operators_list)
    q = ops.add_parser("show", help="one operator's params, outputs and guidance")
    q.add_argument("name")
    q.set_defaults(fn=cmd_operators_show)
    q = ops.add_parser("enable", help="add an operator to traj.yaml")
    q.add_argument("name")
    q.add_argument("--as", dest="alias", help="instance name, needed to enable one operator twice")
    q.add_argument("--prefix", help="prepended as <prefix>_ to every feature name")
    q.add_argument("--call", help="call group for llm operators")
    q.add_argument("--param", action="append", help="key=value, value parsed as YAML")
    q.set_defaults(fn=cmd_operators_enable)
    q = ops.add_parser("disable", help="remove operators with this name or alias from traj.yaml")
    q.add_argument("name")
    q.set_defaults(fn=cmd_operators_disable)

    p = sub.add_parser("discover", help="pick trajectories to read when proposing features")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset", action="append")
    p.add_argument("--method", choices=["diversity", "random"], default="diversity")
    p.add_argument("--group", action="append", help="groups whose features drive diversity; default all code groups")
    p.set_defaults(fn=cmd_discover)

    p = sub.add_parser("extract", help="compute features")
    p.add_argument("--group", action="append")
    p.add_argument("--dataset", action="append")
    p.add_argument("--key", action="append")
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_extract)

    p = sub.add_parser("table", help="print the wide feature table")
    p.add_argument("--dataset", action="append")
    p.add_argument("--columns", nargs="*")
    p.add_argument("--format", choices=["json", "csv", "describe"], default="json")
    p.set_defaults(fn=cmd_table)

    p = sub.add_parser("sample", help="select trajectories for review")
    p.add_argument("--sampler", default="default")
    p.add_argument("--budget", type=int)
    p.set_defaults(fn=cmd_sample)

    p = sub.add_parser("show", help="one trajectory's file and features")
    p.add_argument("key")
    p.add_argument("--cat", action="store_true", help="print the rendered Markdown")
    p.set_defaults(fn=cmd_show)

    p = sub.add_parser("status", help="dataset sizes and extraction coverage")
    p.set_defaults(fn=cmd_status)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        args.fn(args)
    except (ConfigError, ValidationError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
