---
name: traj-features
description: How to define trajectory features as traj-analyzer operators and how to extract them. Use when writing or revising an operator, deciding whether a feature should be computed by code or asked of an LLM, wording an LLM feature, or running and checking an extraction. Triggers include "定义特征", "写个算子", "特征怎么提取", "这个特征准不准", "define a feature", "write an operator", "extract features".
---

# Defining and extracting features

A feature is one value per trajectory.
Features come from operators; one operator is one Python file in the project's `operators/<namespace>/<name>.py`, or in the built-in library.
The feature table holds what the last `traj extract` wrote, and samplers and reports read it.

## 1. Every feature is atomic

A feature answers one question that can be read off the text directly, without multi-step reasoning.
Its value is a yes or no, or a set of items copied from the text.
A hard question gives an inaccurate feature, because the model has to reason across many steps before it answers.

Good LLM features, each answerable by finding one statement:

| Feature | Type | Question |
|---|---|---|
| `user_dissatisfied` | boolean | Does the user write that they are unhappy with the assistant? |
| `user_follow_up` | boolean | Does the user ask a further question after the assistant answers? |
| `user_corrects` | boolean | Does the user point out that the assistant got something wrong? |
| `suspected_services` | set | Which services does the agent write may be the root cause? |
| `ruled_out_services` | set | Which services does the agent write are not the root cause? |
| `says_data_missing` | boolean | Does the agent write that some expected data is missing? |
| `evidence_signals` | set | Which kinds of signal do the submitted evidence claims describe? |

Questions that must not be features, because they need judgement across the whole trajectory:

- Was the agent anchored on its first hypothesis?
- Did the agent handle contradicting evidence well?
- Is the evidence sufficient for the conclusion?
- Why did the agent miss the root cause?
- How frustrated was the user, on a scale from 0 to 1?

Build such a judgement from atomic features instead, in a sampler condition or in a report.
For example, "anchored" is `first_suspect` equal to the submitted service, with a single entry in `suspected_services`.
"Dismissed a true root cause as a victim" is a non-empty `gt_ruled_out`.

## 2. Code or LLM

| The value comes from | Operator kind | Examples |
|---|---|---|
| Structured fields: step roles and kinds, tool names, tool arguments, `is_error`, metadata, the submitted JSON | `code` | number of queries, error rate, whether a normal-period file was queried, number of submitted root causes |
| Reading what someone wrote: messages, reasoning, claims | `llm` | whether the user corrects the assistant, which services the agent says it suspects |

Code operators read fields only.
Do not use regular expressions to interpret text written by a user or a model; ask an LLM operator that question.
When the same question can be answered from fields, use a code operator: it costs nothing and runs on every trajectory.

## 3. Writing an LLM feature

- Phrase the description as a literal question about what is written: "The user writes that ...", "The agent writes that ...".
- Say where to look when it is narrow, such as "look only at the step named final_answer".
- Say what does not count, such as "a tool result showing the same thing does not count".
- For sets, ask the model to copy names exactly as written, and give `labels` when the possible values are known.
- Tell the model not to judge whether the statement is correct.
- Keep `evidence: true` on the call, so every value cites the step it comes from.
- Use the ground truth in metadata only as a lookup, such as "which of rc_services appear in suspected_services".
- Put features that read the same part of the trajectory in one call group, since each call reads the file once.

An LLM operator file:

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
    kind="llm",
    description="Whether the user corrects the assistant or asks follow-up questions.",
    outputs=outputs,
    guidance=lambda params: "Count only what the user writes; cite the user's step.",
    requires=has_user_messages,
)
```

## 4. Writing a code feature

- `compute(trajectory, params)` returns a value for every output name, or None when the feature does not apply to this trajectory.
- Read `step.kind`, `step.name`, `step.is_error`, JSON tool arguments and `trajectory.metadata`.
- Values are checked against the feature type; a wrong type stops the extraction.
- Put code shared by several operators in a file whose name starts with an underscore, and import it from the operator root, e.g. `from rca._parse import calls`.
- Set `requires` when the operator only makes sense for some trajectories; the others get an empty value.

## 5. Enabling

Enable an operator with `traj operators enable <name>`:

- `--call <group>` puts LLM operators into one call group.
- `--param key=value` changes a parameter.
- `--as <instance>` together with `--prefix <prefix>` enables the same operator twice without clashing feature names.

Describe the dataset in `datasets.<name>.description` of `traj.yaml`: what the trajectories are, special step names, and known defects of the data.
It is written at the top of every rendered trajectory, so every LLM operator reads it.
Hide metadata an LLM must not see, such as evaluation results, with `render.hide_metadata`.

## 6. Extracting and checking

1. `traj validate --instructions` shows the output schema and the instruction each call group sends.
2. `traj extract --group <group> --limit 5`, or `--key <key>` for chosen trajectories, runs a small trial.
3. `traj show <key>` prints each feature with its value and cited evidence.
   Open the Markdown with `traj show <key> --cat` and check every value against the cited step.
4. When a value is wrong, first ask whether the question is atomic.
   A feature that is often wrong usually needs to be split into simpler features; rewording the same hard question rarely helps.
5. `traj extract --dry-run` reports how many LLM calls a full run needs; answers already in the mailbox are reused.
6. `traj extract` runs everything; `traj status` shows how many trajectories each group has rows for, by status.
