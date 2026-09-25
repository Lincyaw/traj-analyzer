---
name: traj-discover
description: Find the trajectory features this traj-analyzer project needs by reading sample trajectories, then enable library operators or write new operators for them. Use when the user asks to find, propose, add or revise features or operators, or says "特征发现", "提一些特征", "加个算子", "discover features".
---

# Feature discovery

A feature belongs in the project when its value changes which trajectories someone wants to read.
Features come from operators; one operator is one Python file.

## Steps

1. Run `traj status` and `traj operators list` to see what is enabled and what the library offers.
2. Make sure every code group is extracted, then run `traj discover --n 20`.
   Add `--dataset` when the project has several datasets.
   Read every listed Markdown file.
   Note how the trajectories differ: what the user wanted, how the run went, where it went wrong, and what was unusual.
3. For each difference, look for an operator that captures it with `traj operators show <name>`.
   Enable it with `traj operators enable <name>`, adding `--call` to group LLM operators into one call, `--param key=value` to change parameters, and `--as` to enable the same operator twice.
4. For differences no operator captures, write a new operator in the project's `operators/<namespace>/<name>.py`.
   Copy the structure of a library operator shown by `traj operators show`.
   - Use a code operator when the value follows from the trajectory by rule, and an LLM operator when it needs judgement.
   - Give every output a `description` precise enough that two readers would give the same value.
   - Give category, set and distribution outputs `labels` with one line definitions, covering every case seen, plus `other` where needed.
   - Give scalars and vectors a `range`, and `thresholds` when a cut off matters for sampling.
   - Put the judging scale in `guidance`, and set `requires` when the operator only makes sense for some trajectories.
   Then enable it.
5. Run `traj validate` and fix every error.
6. Try new LLM operators on a few trajectories with `traj extract --group <call> --limit 5`.
   Run `traj show <key>` for each one and check the values and evidence against the file.
   Revise the operators where the answers were wrong or inconsistent.
7. Report the enabled and new operators, the purpose of each one, and what the trial run showed.
   Leave the changes uncommitted so the user can review the diff, unless the user asked for a commit.
