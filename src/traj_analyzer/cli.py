from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ValidationError

from traj_analyzer.extract import extract, select_rows
from traj_analyzer.features.spec import load_groups, output_model, render_instruction
from traj_analyzer.features.table import read_group
from traj_analyzer.ingest import ingest, load_index
from traj_analyzer.project import CONFIG_FILE, ConfigError, Project
from traj_analyzer.sampling import diverse, enrich, load_sampler, sample, save
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
            "group": group.group, "engine": group.engine,
            "features": [f.name for f in group.features], "spec_hash": group.spec_hash(),
        }
        if not group.is_builtin:
            entry["output_schema"] = output_model(group).model_json_schema()
            if args.instructions:
                entry["instruction"] = render_instruction(group)
        report["groups"].append(entry)
    samplers = sorted(p.stem for p in project.samplers_dir.glob("*.yaml"))
    for name in samplers:
        load_sampler(project, name)
    report["samplers"] = samplers
    _emit(report)


def _frame(project: Project, datasets: list[str] | None, groups: list[str] | None = None) -> Frame:
    return build_frame(project, load_groups(project, groups), load_index(project, datasets),
                       project.config.sampling.vector_length)


def cmd_discover(args: argparse.Namespace) -> None:
    project = Project.find()
    rows = select_rows(project, args.dataset, None, None)
    if args.method == "random":
        order = [int(i) for i in np.random.default_rng(args.seed).permutation(len(rows))]
        chosen = [(rows[i], {}) for i in order[: args.n]]
    else:
        frame = _frame(project, args.dataset, [args.group])
        if not frame.columns:
            raise ConfigError(f"Group {args.group} has no extracted values; "
                              f"run `traj extract --group {args.group}` first")
        frame = frame.complete(list(frame.columns))
        x = frame.matrix(dict.fromkeys(frame.columns, 1.0))
        position = {row.key: row for row in rows}
        keys = list(frame.query.index)
        picks = diverse("discover", args.n, list(range(len(keys))), keys, x, args.seed,
                         population=list(range(len(keys))))
        chosen = [(position[p.key], p.reason) for p in picks]
    _emit({"method": args.method, "population": len(rows), "trajectories": [
        {"key": r.key, "markdown": str(project.dataset_dir(r.dataset) / r.markdown),
         "n_steps": r.n_steps, "chars": r.chars, "reason": reason, "metadata": r.metadata}
        for r, reason in chosen
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
    dataset, _, tid = args.key.partition("/")
    path = project.dataset_dir(dataset) / f"{tid}.md"
    if not path.is_file():
        raise ConfigError(f"No trajectory {args.key}")
    if args.cat:
        sys.stdout.write(path.read_text(encoding="utf-8"))
        return
    features = {}
    for group in load_groups(project):
        for row in read_group(project, group.group):
            if row.key == args.key:
                features[row.feature] = row.model_dump(exclude={"key", "feature"})
    _emit({"key": args.key, "markdown": str(path), "features": features})


def cmd_status(args: argparse.Namespace) -> None:
    project = Project.find()
    index = load_index(project)
    shas = {row.key: row.sha256 for row in index}
    datasets: dict[str, int] = {}
    for row in index:
        datasets[row.dataset] = datasets.get(row.dataset, 0) + 1
    groups = {}
    for group in load_groups(project):
        digest = group.spec_hash()
        stored = read_group(project, group.group)
        fresh = [r for r in stored if r.spec_hash == digest and shas.get(r.key) == r.sha256]
        current = {r.key for r in fresh if r.status == "ok"}
        problems = {r.key for r in fresh if r.status != "ok"}
        stale = {r.key for r in stored if r.key in shas} - current - problems
        groups[group.group] = {"ok": len(current), "not_ok": len(problems), "stale": len(stale),
                               "missing": len(shas) - len(current | problems | stale)}
    _emit({"project": str(project.root), "datasets": datasets, "groups": groups})


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

    p = sub.add_parser("validate", help="check traj.yaml, feature groups and samplers")
    p.add_argument("--instructions", action="store_true", help="include rendered instructions")
    p.set_defaults(fn=cmd_validate)

    p = sub.add_parser("discover", help="pick trajectories to read when proposing features")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset", action="append")
    p.add_argument("--method", choices=["diversity", "random"], default="diversity")
    p.add_argument("--group", default="stats", help="features used by the diversity method")
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
