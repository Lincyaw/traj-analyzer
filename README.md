# traj-analyzer

traj-analyzer analyses batches of LLM trajectories in any format.
Operators you choose to enable turn each trajectory into features, a sampler picks the few trajectories most worth reading, and Claude Code reads them and writes a report.
An operator is one Python file; the built-in library lives in `src/traj_analyzer/operators/library/`, and an analysis project adds its own in `operators/`.
The design is in [docs/design.md](docs/design.md).

## Install

```sh
uv sync
uv run python -m aifn install --dsh-home ~/.dsh
export DEEPSEEK_API_KEY=...
```

The second command installs the aifn harness bundle into the dsh home, which runs LLM feature extraction.

## Usage

```sh
traj init ~/analysis/cc-sessions
cd ~/analysis/cc-sessions && git init
traj adapters list
traj ingest
traj operators list
traj extract --group stats-basic --group stats-tool_usage
traj discover --n 20
traj validate
traj extract --group dialogue --limit 5
traj study screen corrections
traj study pairs corrections
traj sample
```

The `traj.yaml` created by `traj init` enables every general operator of the built-in library, and the seven LLM operators share the `dialogue` call group.
New data sources and new features are added as files: adapters in the project's `adapters/`, operators in the project's `operators/`, as the "Extension points" section of the design describes.
A study in `studies/` states which trajectories succeed; `traj study pairs` draws contrast pairs to read, and `traj study screen` checks whether each feature is constant, redundant, differs between groups such as models, or explains success.
The project also carries three skills: `traj-features` explains how features are defined and extracted, `traj-discover` proposes features from contrast pairs and verifies them, and `traj-report` writes reports.

Every command prints JSON to stdout.
Exit codes: 0 success, 1 runtime error, 2 usage or configuration error.

## Development

```sh
uv run pytest
uv run ruff check src tests
```
