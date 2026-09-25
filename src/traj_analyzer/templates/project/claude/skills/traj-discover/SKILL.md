---
name: traj-discover
description: Propose trajectory features for this traj-analyzer project by reading sample trajectories, then write them as feature group YAML. Use when the user asks to find, propose, add or revise features, or says "特征发现", "提一些特征", "discover features".
---

# Feature discovery

A feature belongs in the project when its value changes which trajectories someone wants to read.

## Steps

1. Run `traj status` and read the existing `features/*.yaml`, so that new features extend the existing ones.
2. Make sure the `stats` group is extracted, then run `traj discover --n 20`.
   Add `--dataset` when the project has several datasets.
   Read every listed Markdown file.
   Note how the trajectories differ: what the user wanted, how the run went, where it went wrong, and what was unusual.
3. Turn the differences into features.
   For each feature decide the following.
   - The `type`: scalar, boolean, category, set, vector or distribution.
   - A `description` precise enough that two readers would give the same value.
   - `labels` with one line definitions for category, set and distribution features, covering every case seen, plus `other` where needed.
   - `range` and `thresholds` for scalars and vectors when a cut off matters for sampling.
   Keep the set of features small and each definition precise.
   Put features judged from the same reading in one group, since one group costs one LLM call per trajectory.
4. Write or edit `features/<group>.yaml`, then run `traj validate` and fix every error.
5. Try the group on a few trajectories with `traj extract --group <group> --limit 5`.
   Run `traj show <key>` for each one and check the values and evidence against the file.
   Revise the definitions where the answers were wrong or inconsistent.
6. Report the proposed features, the purpose of each one, and what the trial run showed.
   Leave the YAML uncommitted so the user can review the diff, unless the user asked for a commit.
