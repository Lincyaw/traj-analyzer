---
name: traj-features
description: How to define trajectory features as traj-analyzer operators and how to extract them. Use when writing or revising an operator, deciding whether a feature should be computed by code or judged by a model, wording a judgment feature, or running and checking an extraction. Triggers include "定义特征", "写个算子", "特征怎么提取", "这个特征准不准", "define a feature", "write an operator", "extract features".
---

# Defining and extracting features

A feature is one value per trajectory.
Features come from operators; one operator is one Python file in the project's `operators/<namespace>/<name>.py`, or in the built-in library.
The feature table holds what the last `traj extract` wrote, and samplers and reports read it.

## 1. Every feature is atomic

A feature answers one question that can be read off the text directly, without multi-step reasoning.
Its value is a yes or no, or a set of items copied from the text.
A hard question gives an inaccurate feature, because the model has to reason across many steps before it answers.

Good judgment features, each answerable by finding one statement:

| Feature | Type | Question |
|---|---|---|
| `user_complains` | boolean | Does the user write that they are unhappy with the assistant? |
| `user_follow_up` | boolean | Does the user ask a further question after the assistant answers? |
| `user_corrects` | boolean | Does the user point out that the assistant got something wrong? |
| `suspected_services` | set | Which services does the agent write may be the root cause? |
| `ruled_out_services` | set | Which services does the agent write are not the root cause? |
| `agent_notes_missing_data` | boolean | Does the agent write that some expected data is missing? |
| `evidence_signals` | set | Which kinds of signal do the submitted evidence claims describe? |

Questions that must not be features, because they need judgement across the whole trajectory:

- Was the agent anchored on its first hypothesis?
- Did the agent handle contradicting evidence well?
- Is the evidence sufficient for the conclusion?
- Why did the agent miss the root cause?
- How frustrated was the user, on a scale from 0 to 1?

Build such a judgement from atomic features instead, in a sampler condition or in a report.
For example, "anchored" is `first_suspect` equal to the submitted service, with a single entry in `suspected_services`.
"Dismissed a true root cause as a victim" is a service found both in `ruled_out_services` and in the ground truth.

Name a feature with a noun, such as `tool_calls` or `suspected_services`, or with a noun and a verb, such as `user_corrects` or `agent_notes_missing_data`.
A name never starts with a verb.

## 2. Feature shapes

Every feature takes one of four shapes, so that values from many trajectories can be put side by side and compared.

| Shape | Types | Value | Example | Compared across trajectories as |
|---|---|---|---|---|
| boolean | `boolean` | yes or no | `user_corrects` | the share of trajectories where it is true, per group |
| set | `set`, and `category` for exactly one item | distinct labels | `request_kinds`, `suspected_services` | how often each label occurs; overlaps and differences between groups |
| vector | `vector`, and `scalar` for one number | ordered numbers | `n_steps`, errors per chunk | distributions, curve shapes, maxima and means |
| map | `map` | label to number | `tool_calls` = calls per tool | per-label values, aligned by label |

Judgment features are `boolean`, `category` or `set`.
Numbers, vectors, maps, paths and events come from code operators, which count and measure the tables.
Give `labels` when the possible labels are known; the model then chooses among them, and columns stay stable across runs.

## 3. Code or judgment

| The value comes from | Operator kind | Examples |
|---|---|---|
| The tables: steps, the actions of the tool calls with their objects, effects and outcomes, the relations between actions, the SQL queries, metadata | `code` | number of queries, error rate, share of edits verified in the same turn, a failed action repeated unchanged |
| Reading what someone wrote: messages, reasoning, claims | `judgment` | whether the user corrects the assistant, which services the agent says it suspects |

Code operators read the tables and the structured fields only; `traj layers list` shows the tables and their columns, and `traj query` runs SQL over them.
Do not use regular expressions to interpret text written by a user or a model; ask a judgment operator that question.
When the same question can be answered from the tables, use a code operator: it costs nothing and runs on every trajectory.
A judgment costs one model call per trajectory, or per turn list or window; bound it with `sample` per batch or a `scope` over code features when a panel needs only an estimate or the question arises in few trajectories.

## 4. Writing a judgment feature

- Phrase the description as a literal question about what is written: "The user writes that ...", "The agent writes that ...".
- Say where to look when it is narrow, such as "look only at the step named final_answer".
- Say what does not count, such as "a tool result showing the same thing does not count".
- For sets, ask the model to copy names exactly as written, and give `labels` when the possible values are known.
- Tell the model not to judge whether the statement is correct.
- Keep `evidence: true` on the call, so every value cites the step it comes from.
- Keep ground truth out of judgment features: hide it with `render.hide_metadata`, and compare extracted values with it downstream, in samplers or reports.
- Put features that read the same part of the trajectory in one call group, since each call reads the file once.
- A question about every message the user typed is an operator with `unit="turn"` and aggregates, like `signals.user_reaction`; a question about a stretch of work against the request is one with `unit="window"`, like `signals.request_fit`.

A judgment operator file:

```python
from traj_analyzer.operators.base import FeatureSpec, NoParams, Operator, has_user_messages


def outputs(params: NoParams) -> list[FeatureSpec]:
    return [
        FeatureSpec(name="user_corrects", type="boolean",
                    description="The user writes that the assistant got something wrong, such as a wrong fact, a "
                                "misread request or an unwanted change."),
        FeatureSpec(name="user_follow_up", type="boolean",
                    description="After an assistant answer, the user asks a further question on the same task."),
    ]


OPERATOR = Operator(
    kind="judgment",
    description="Whether the user corrects the assistant or asks follow-up questions.",
    outputs=outputs,
    guidance=lambda params: "Count only what the user writes; cite the user's step.",
    requires=has_user_messages,
)
```

## 5. Writing a code feature

- `compute(trajectory, tables, params)` returns a value for every output name, or None when the feature does not apply to this trajectory.
- Declare the tables it reads in `tables=(...)`; `tables["actions"]` is a pandas frame of the trajectory's rows, `tables.records("actions")` the rows as dicts.
- Read `step.kind`, `step.name`, `step.is_error`, JSON tool arguments and `trajectory.metadata` for what the tables do not hold.
- Return `Evidenced(value, steps, note)` when the value comes from particular steps, so a person can open the trajectory there.
- An expectation that should hold at some positions is an event feature: a `Rule` with `precondition`, `expectation` and `recovery` over the actions, returned through `events_of`; `events.generic` in the library shows the form.
- Values are checked against the feature type; a wrong type stops the extraction.
- Put code shared by several operators in a file whose name starts with an underscore, and import it from the operator root, e.g. `from rca._parse import calls`.
- Set `requires` when the operator only makes sense for some trajectories; the others get an empty value.

## 6. Enabling

Look in `traj operators list` before writing a new operator.
The library has code operators for size, tool usage, actions and the dialogue (`stats.*`), for the generic event rules (`events.generic`) and for metadata (`meta.fields`); the signals of quality a trajectory itself shows (`signals.user_reaction`, `signals.end_state`, `signals.request_fit`); and atomic judgment operators for users (`user.complains`, `user.corrects`, `user.follow_up`, `user.approves`), assistants (`assistant.claims_done`, `assistant.asks_user`) and requests (`task.request_kinds`).

A feature that needs data the adapter does not provide calls for a change to the adapter, `adapters/<name>.py` defining `ADAPTER`; `traj adapters list` shows the adapters.
A dataset from a harness the library has no profile for needs `harness/<name>.py` defining `HARNESS`, the tool names, argument keys and result openings of that harness, which the adapter names in `Trajectory.harness`; without a profile every tool is `other` and a result is a failure only when the adapter flagged it.

Enable an operator with `traj operators enable <name>`:

- `--call <group>` puts judgment operators into one call group.
- `--param key=value` changes a parameter.
- `--as <instance>` together with `--prefix <prefix>` enables the same operator twice without clashing feature names.

Describe the dataset in `datasets.<name>.description` of `traj.yaml`: what the trajectories are, special step names, and known defects of the data.
It is written at the top of every rendered trajectory, so every judgment operator reads it.
Hide metadata a model must not see, such as evaluation results, with `render.hide_metadata`.
Declare `batches` and `strata` in `traj.yaml`, so that `traj anomalies` compares batches inside strata, and `labels` where the dataset has them.

## 7. Extracting and checking

1. `traj extract --group <group> --dry-run` writes the instruction a call group sends to `.traj/instructions/<group>.md` and reports in `calls` how many model calls a full run needs; answers already in the mailbox are reused.
2. `traj extract --group <group> --limit 5`, or `--key <key>` for chosen trajectories, runs a small trial.
3. `.traj/features/<group>.parquet` holds one row per trajectory, with every feature's value and its cited evidence in the column `<feature>__evidence`.
   Open the trajectory's Markdown, `.traj/datasets/<dataset>/<id>.md`, and check every value against the cited step.
4. When a value is wrong, first ask whether the question is atomic.
   A feature that is often wrong usually needs to be split into simpler features; rewording the same hard question rarely helps.
5. `traj extract` runs everything.
   It checks `traj.yaml`, the enabled operators and every sampler first, and its `coverage` shows per group how many trajectories have rows, by status, and how many are `missing`.
6. `traj anomalies` compares every batch with the baseline inside every stratum, measures how often a consequence follows the turns the events of every rule fall in, and scores every trajectory; `traj query` reads the resulting `shifts`, `lifts`, `oversight` and `anomalies` tables, and the `events` and `evidence` tables name the step to open a trajectory at.
7. After opening a trajectory a feature pointed to, record what you found with `traj review <key> --feature <name> --step <n> --worth|--no-worth --note "..."`; once a feature has enough verdicts, they decide whether it raises alerts instead of the model's tail check.
