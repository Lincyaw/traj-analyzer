# Interactive analysis

Run `traj view` from an analysis project after ingesting trajectories and extracting features, and open http://127.0.0.1:8000.
The server reads the project's Parquet tables into an in-memory DuckDB; the page fetches nothing from outside the server.
The review pages follow the order in which a person looks at a batch of trajectories: the batches, the trajectories worth a look, one trajectory, the features that pointed at them, and the hunts with the trajectories they found.
The data page, reached from the `Data` link, keeps the table browser and the Perspective charts for analyses the review pages do not anticipate.
Every page is addressed by the fragment of the URL, so a view can be shared as a link.

## Overview

The trajectories in groups, one row each, with the oversight metrics: turns, how often the agent asked the user, how often the user interrupted or pushed back, the median and tail of the turn durations, the tool calls per turn, and how many trajectories of the group rank among the highest anomaly scores of the project.
The grouping is chosen on the page: any metadata key, or any feature whose values form groups, with numbers cut into quartiles; the batches of `traj.yaml` are the starting choice.
Hover a column title for the definition of the metric.
Under each group with `alerts.min_batch` trajectories, the features where it differs most from the other groups inside the strata: the group's average against the others' average, ordered by the standardized difference, one feature per line; clicking one opens the trajectories of the group where the feature stands out.
Clicking a group opens its trajectories.
The metrics and the differences are computed when a grouping is first opened and kept for the life of the server; the count of unusual trajectories needs `traj anomalies` to have run.

## Trajectories

The trajectories most anomalous first, each on one line that says why: the first unrecovered event with its note and step, and the feature that puts the trajectory furthest from its neighbours.
Under the line, the label the last judged message or window received for every per-unit output, flagged when it counts as an event; the turns that ended otherwise than most turns do; the counts of events per rule.
On the right, the turns, steps, duration, failed calls and questions to the user.
Trajectories someone reviewed carry their verdicts.

The filters narrow the list by batch, stratum, an unrecovered event rule, a judged label, how some turn ended, a feature among the anomaly reasons or event rules, and whether the trajectory was reviewed; the order can follow the anomaly rank, the score, the duration, the turns or the steps.
The filter values and their counts come from the tables, so a project sees only what its data holds.
Arrow keys move along the list and Enter opens the selected trajectory at its first unrecovered event.

## Trajectory

The left column is the course of the trajectory, one block per turn: the message the user typed, the labels the judgments gave it, how the turn ended when that differs from most turns, the actions summarised by act and expanded on demand, the notices the harness inserted, and the events with their notes, each marked recovered or unrecovered.
The middle column shows the steps of the current turn with their full text; clicking a message, an action, an event or a notice selects its step.
The event buttons in the header walk through the unrecovered events in order.

The right column says why the trajectory stands out: the features named in its anomaly reasons, each with its value, what is typical of the batch, a bar for how far it lies from the neighbours and, when the operator cites steps, the steps the value rests on.
Every other feature is one click further down.
Below, the verdict form records what the person found: the feature, the step and whether the trajectory was worth their look because of what the feature found there, with a note.
Verdicts append to `reviews/verdicts.jsonl`, the file `traj review` writes, and measure the feature's precision.

## Features

One row per feature and per-unit output with how it holds up: its role from `traj evaluate` and the noise of a judgment; how many anomalies it explains; the events its rule found and how many went unrecovered; the consequence lift of its events; the tail precision and lift the mining check gave it and whether it raises alerts; what the people who followed it found; and the batch where it moved furthest from the baseline.
Columns sort on click, the name filter and the signal switch narrow the rows, and every count opens the trajectories behind it.
Hover a column title for what it measures.

## Hunts

One row per hunt of the project with its last finished run: the rounds, the confirmed trajectories, how many of the trajectories were read, the facets and the probes kept.
A hunt's page shows its need; the coverage with the bound on what the probes may still miss; the facets with their edge, their confirmed trajectories and the trajectories their marks point to; the batches and strata with their confirmed trajectories by facet; the probes kept with their precision and unread hits; how the blind reads agree with the reads that showed the hits; and how the verdicts people gave on the features of the facets agree with the hunt's.
Below, every trajectory the hunt read or its marks point to: confirmed, marked and unread, or rejected, each with its facet, batch, the steps its verdict cites, the reason and its marks, narrowed by verdict and facet.
A click opens the trajectory at the first step its verdict cites, or its first mark.
On the page of a trajectory, the verdicts of the hunts that read it stand in the right column with their steps, and the marks appear in its course like any other event; the verdict form records, on the feature of a facet, whether the trajectory shows it.

## Data

The data page lists `features`, which joins every extracted group's values by trajectory key, one table per extracted group with its status, detail and evidence columns, and every table of `.traj/tables/`.
Search and WHERE filter the input population before it is sent to the browser; the input row count is displayed above the chart.

Choose a Feature and use Distribution to count numeric intervals or category values; numeric distributions exclude null values from their bars and report the null count separately.
Choose a Group by column and use Compare to aggregate the selected feature for each group; numeric comparison uses the mean and `__count` uses the sum to count input rows.
Relationship displays two numeric columns as a scatter plot.
Open the Perspective settings to change columns, aggregates, grouping, splitting, filters, sorting and expressions, and use Add panel to copy the active analysis into another panel.

Select `Labels: <feature>` in Source for a set, vector or map to receive one record per label, vector position or map entry: `__label` holds the label, `__value` the membership or element, `__count` 1 per record.
Null and empty collections produce no label records.

Enter a View name and Save to retain the input table, source, columns, Search, WHERE and the panel workspace; saved views live in the browser, separately for each project's absolute path.
Export view downloads a JSON configuration that another browser can import with Import view; Export data downloads the active Perspective view.
An incompatible view displays an error and does not substitute another analysis.

The Rows mode shows the same table paged, searchable and filterable by column.

## Input limits and refresh

Analysis data is limited to 250,000 rows and 128 MiB per request; add filters or select fewer columns when a request exceeds them.
The server reads the tables when it starts; run `traj view` again after extracting features or running `traj anomalies` to see the new state.
