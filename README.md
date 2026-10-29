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
uv run traj check
```

The second command installs the aifn harness bundle into the dsh home, which runs the judgment calls and the functions of mining and hunts.
`traj check`, run in an analysis project, checks that workers can run model calls: the variables and patches of `engine`, the dsh runtime and the aifn bundle, and one call that runs a shell command in the sandbox of a call; every command that starts workers runs the same check first, once per project.

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
traj hunt run reward-hacking
traj hunt report reward-hacking
```

The `traj.yaml` created by `traj init` enables every general operator of the built-in library: the code operators over the steps and the actions, the generic event rules, the signals of the user's reaction and the end state, and the seven judgment operators of the `dialogue` call group.
`batches`, `strata`, `alerts` and `labels` in `traj.yaml` say how batches are compared, which event rules count as consequences, and which labels, if any, check the method.
The turn, from one message the user typed to the next, is the unit of oversight: the `turns` table says how each ended, `traj anomalies` measures per batch how often the agent asks and the user interrupts or pushes back, and a person's verdicts recorded with `traj review` take the place of the model's tail check once there are enough of them.
New data sources and new features are added as files: adapters in the project's `adapters/`, harness profiles in the project's `harness/`, operators in the project's `operators/`, as the "Extension points" section of the design describes.
Three adapters ship with the library: `atif` for a trajectory in the Agent Trajectory Interchange Format, `claude_code` for Claude Code session files, and `messages` for records holding a list of messages.
An adapter whose source also records the calls of the agent, such as a proxy that logs every response, gives their tools as `recorded_calls`, and `traj ingest` stops at the first trajectory whose parsed calls lack one of them.
The library knows the tools and result texts of no harness itself; a profile names the tools, argument keys and result openings of one harness, and the built-in `claude_code` profile is the only one it ships.
`traj extract` checks the configuration before it runs and reports, per execution group, how many trajectories have features.
This repository provides three shared skills for Codex and Claude Code: `traj-features` explains how features are defined and extracted, `traj-report` writes reports, and `traj-hunt` runs a hunt from the agent session.
`traj init` installs them into `.claude/skills/` and creates relative directory links in `.agents/skills/` for Codex.
Both agents read the same skill files, including any later project edits or supporting resources.
The tool repository uses the same directory arrangement; maintain skill content in `.claude/skills/`.
Launch either agent in the analysis project to define features, write reports or run hunts.

`traj evaluate` measures every enabled feature for noise, variation and dependence and gives it a role: distinguishing, invariant or dependent; with labels declared it also checks the signals and the anomalies against them.
`traj mine` finds new features: agents describe the scenario of the data, reason from it to measures and expectations that may matter, test those hypotheses against the trajectories and their tables, write the findings as features or event rules and review them; each candidate is then checked for noise, for how its values spread over the batches, and for whether its tail is worth a look, and the accepted ones are enabled.
Both need `evaluation.model` in `traj.yaml`, a second model that measures judgment features again; the "Feature evaluation and mining" section of the design describes them.

A hunt looks for every trajectory that shows what `hunts/<name>.yaml` asks for in its `need`, such as reward hacking or a destructive command nobody asked for.
Each round, an agent proposes probes, SQL queries or operator files that find positions mechanically, and tries each with `traj hunt try`; the probes run over every trajectory; agents read a sample of what they hit and of what they miss, and other readers read some of the hit trajectories again without being shown the hits; and the verdicts tell the next proposal which probes to tighten and which misses to catch.
The search ends when `patience` rounds in a row, 3 by default, confirm no new kind of behaviour and no trajectory the probes miss.
`traj hunt run <name>` runs the whole search unattended with aifn functions.
`traj hunt run <name> --new-run` starts a separately numbered run and preserves any unfinished run, including its model calls and state, when the input conversations change.
An agent session following the skill `traj-hunt` runs the same search step by step with `traj hunt start`, `propose`, `verdicts` and `close`: it proposes the probes itself, and subagents read the cases; the commands check what it hands in against the same output models and keep the same state.
Every verdict goes to the table `hunt_verdicts`, and the probes to `hunts/<name>/`, where the next run starts from them.
The kept probes then mark every trajectory: the run enables the operator `hunt.marks` as `hunt-<name>`, one event feature per facet at the steps its probes hit, so every `traj extract` marks new trajectories too, `traj anomalies` compares the batches, and `traj view` shows the marks in the course of a trajectory.
A run whose marks differ from the positions its last round found names the probes that read the batch rather than one trajectory, and enables nothing.
`traj hunt report <name>` prints what a run found, and the Hunts page of `traj view` lists every trajectory a hunt confirmed, rejected or only marked, opening each at the step worth a look; the "Hunts" section of the design describes the rounds, the draws, the marks and the bound on what the probes may still miss.

`traj view` serves the review pages at http://127.0.0.1:8000: the batches with their oversight metrics and the features that moved, the trajectories most anomalous first with one line saying why, one trajectory's course turn by turn with its steps and the reasons it stands out, the features with how they hold up, and the hunts with every trajectory they confirmed, rejected or only marked; verdicts recorded there go to `reviews/verdicts.jsonl`.
The `Data` page keeps the searchable tables and the Perspective charts for other analyses.
See [Interactive analysis](docs/viewer.md) for the pages, saved views and data limits.
`traj table` and `traj query` print CSV to stdout, and every other command except `traj view` prints JSON.
Exit codes: 0 success, 1 runtime error, 2 usage or configuration error.

## Development

```sh
uv run pytest
uv run ruff check src tests
```
