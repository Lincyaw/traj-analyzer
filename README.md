# traj-analyzer

traj-analyzer analyses batches of LLM trajectories in any format, at low cost in people and model calls.
Parsing layers turn every trajectory into tables, operators turn the tables and the text into features, the features go on a dashboard, `traj anomalies` finds the batches and trajectories that stand apart, and every alert leads to a trajectory and a step.
Features come from code, which runs on every trajectory, or from judgments, one model call per trajectory or per window; `traj mine` finds new ones with multi-step agents and checks their tails.
An operator is one Python file; the built-in library lives in `src/traj_analyzer/operators/library/`, and an analysis project adds its own in `operators/`.
The design is in [docs/design.md](docs/design.md).

## Install

```sh
uv sync
uv run python -m aifn install --dsh-home ~/.dsh
export DEEPSEEK_API_KEY=...
```

The second command installs the aifn harness bundle into the dsh home, which runs the judgment calls and the mining functions.

## Usage

```sh
traj init ~/analysis/cc-sessions
cd ~/analysis/cc-sessions && git init
traj adapters list
traj ingest
traj operators list
traj layers list
traj extract --group stats-basic --group stats-actions --group events-generic
traj extract --group reactions --limit 5
traj query "SELECT act, outcome, count(*) FROM 'actions.parquet' GROUP BY ALL ORDER BY 3 DESC"
traj query "SELECT ended_by, count(*) FROM 'turns.parquet' GROUP BY ended_by"
traj anomalies
traj review swe-chat/<id> --feature failure_run --step 32 --worth --note "three failed builds, then the user gave up"
traj table > .traj/table.csv
traj view
traj sample
traj evaluate
traj mine
```

The `traj.yaml` created by `traj init` enables every general operator of the built-in library: the code operators over the steps and the actions, the generic event rules, the signals of the user's reaction and the end state, and the seven judgment operators of the `dialogue` call group.
`batches`, `strata`, `alerts` and `labels` in `traj.yaml` say how batches are compared, which event rules count as consequences, and which labels, if any, check the method.
The turn, from one message the user typed to the next, is the unit of oversight: the `turns` table says how each ended, `traj anomalies` measures per batch how often the agent asks and the user interrupts or pushes back, and a person's verdicts recorded with `traj review` take the place of the model's tail check once there are enough of them.
New data sources and new features are added as files: adapters in the project's `adapters/`, harness profiles in the project's `harness/`, operators in the project's `operators/`, as the "Extension points" section of the design describes.
The library knows the tools and result texts of no harness itself; a profile names the tools, argument keys and result openings of one harness, and the built-in `claude_code` profile is the only one it ships.
`traj extract` checks the configuration before it runs and reports, per execution group, how many trajectories have features.
This repository provides two shared skills for Codex and Claude Code: `traj-features` explains how features are defined and extracted, and `traj-report` writes reports.
`traj init` installs them into `.claude/skills/` and creates relative directory links in `.agents/skills/` for Codex.
Both agents read the same skill files, including any later project edits or supporting resources.
The tool repository uses the same directory arrangement; maintain skill content in `.claude/skills/`.
Launch either agent in the analysis project to define features or write reports.

`traj evaluate` measures every enabled feature for noise, variation and dependence and gives it a role: distinguishing, invariant or dependent; with labels declared it also checks the signals and the anomalies against them.
`traj mine` finds new features: agents describe the scenario of the data, reason from it to measures and expectations that may matter, test those hypotheses against the trajectories and their tables, write the findings as features or event rules and review them; each candidate is then checked for noise, for how its values spread over the batches, and for whether its tail is worth a look, and the accepted ones are enabled.
Both need `evaluation.model` in `traj.yaml`, a second model that measures judgment features again; the "Feature evaluation and mining" section of the design describes them.

`traj view` serves an interactive analysis page at http://127.0.0.1:8000.
Select features and grouping columns, then configure charts, aggregates, filters, expressions and multiple panels in the embedded Perspective viewer.
The Rows mode retains the searchable feature table and every table of `.traj/tables/`: the actions, the events, the evidence, the judgments, the shifts and the anomalies.
See [Interactive analysis](docs/viewer.md) for label analysis, saved views and data limits.
`traj table` and `traj query` print CSV to stdout, and every other command except `traj view` prints JSON.
Exit codes: 0 success, 1 runtime error, 2 usage or configuration error.

## Development

```sh
uv run pytest
uv run ruff check src tests
```
