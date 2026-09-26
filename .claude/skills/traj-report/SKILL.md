---
name: traj-report
description: Sample trajectories in a traj-analyzer analysis project, read them, and write an insight report, then write the findings back into feature and sampler definitions. Use when the user asks for an analysis, a report or insights, or says "看看这批轨迹", "写个报告", "analyze the trajectories".
---

# Insight report

## Steps

1. Run `traj extract` for the groups whose features the sampler uses; it refreshes them after any change to data or operators, and calls the model only for what changed.
   Check in its `coverage` that those groups have no `missing` trajectories.
2. Run `traj table > .traj/table.csv` and read the file with pandas for distributions, label counts and missing values over the whole population.
3. Run `traj sample --sampler <name>`, where the default name is `default`.
   Each pick carries `strategy`, `reason` and the full set of feature columns.
   The reason holds the matched condition, the outlier score, or the cluster size and share.
4. Read every picked Markdown file.
   For each one, note what happened, why the sampler picked it, and whether its feature values are correct.
5. Write `reports/<YYYY-MM-DD>-<topic>.md` with these sections.
   - **Summary**: the three to five most important findings.
   - **Findings**: each pattern, how common it is from cluster shares or table counts, and the trajectories that show it, cited as `dataset/id #step`.
   - **Feature quality**: wrong values, features that separate nothing, and distinctions seen that no feature captures.
   - **Proposed changes** to operators in `operators/`, to the `operators` list in `traj.yaml`, and to `samplers/*.yaml`.
6. When the user agrees, or asked for the changes up front, apply the proposed changes and run `traj extract`, which checks the configuration and every sampler before extracting.
   Commit the report together with the operator and configuration changes, and name the report in the commit message.
