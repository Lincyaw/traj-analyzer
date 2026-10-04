# Interactive analysis

Run `traj view` from an analysis project after ingesting trajectories and extracting features.
Open http://127.0.0.1:8000 to analyse the project's Parquet feature tables.
Analysis runs locally, using DuckDB to prepare data and Perspective to query and visualise it in the browser.
The JavaScript, WebAssembly and themes are bundled with the Python package; the page does not fetch CDN assets.

## Explore features

Choose a table from the navigation bar.
`features` joins every extracted group's values by trajectory key.
Each group table also contains extraction status, detail and evidence columns.
Search and WHERE filter the input population before it is sent to the browser.
The input row count is displayed above the chart.
No sampling or page limit is applied to analysis data.

Choose a Feature and use Distribution to count numeric intervals or category values.
Numeric distributions exclude null values from their bars and report the null count separately.
Choose a Group by column and use Compare to aggregate the selected feature for each group.
Numeric comparison uses the mean; `__count` uses the sum to count input rows.
Relationship displays two numeric columns as a scatter plot.
These buttons configure the active panel and replace its filters and expressions.

Open the Perspective settings to change columns, aggregates, grouping, splitting, filters, sorting and expressions.
For example, group by `model`, split by `system` and select several numeric features to compare models by system.
An expression such as `"query_error_rate" * 100` creates a percentage column without changing the stored feature.
Use Add panel to copy the active analysis into another panel, then configure each panel independently.
Panel layouts and chart settings are part of the saved workspace.
The input row count describes the data loaded from DuckDB, before filters configured inside Perspective.

## Analyse labels and vector elements

Select `Labels: <feature>` in Source for a set, vector or map.
The browser receives one record per set label, vector position or map entry.
`__label` contains the set label, zero-based vector position or map key.
`__value` contains 1 for set membership, or the stored vector element or map value.
`__count` contains 1 for each input record.
The initial chart sums `__value` by label.
Null and empty collections produce no label records; neither is treated as zero.
Use Rows to inspect those collections in the original trajectory table.

Label sources include textual context columns such as the trajectory key, dataset and any category features by default.
Use Columns to load to include additional comparison measures or to reduce the input size.
Selecting a collection column in the trajectory source exposes its JSON representation; use a label source to analyse its members numerically.
Totals in a label source count label records, not trajectories.

## Save and share an analysis

Enter a View name and use Save to retain the input table, source, selected columns, Search, WHERE and complete panel workspace.
Saving an existing name replaces that saved view.
Saved views are stored in this browser, separately for each project's absolute path.
Choose a Saved view to restore it.
Delete removes the selected saved view from browser storage.

Export view downloads a JSON configuration that another browser can import with Import view.
Export data downloads the active Perspective view, including its filters and aggregates.
The target project must contain the table and columns referenced by an imported view.
An incompatible view displays an error and does not silently substitute another analysis.

## Input limits and refresh

An analysis may load at most 250,000 records and 128 MiB of decoded Arrow data into the browser.
Larger inputs return an error before transfer; narrow Search or WHERE, or select fewer columns.
A label source can have more records than the original trajectory table.
Limits do not silently truncate records.

The server loads Parquet data when `traj view` starts.
Restart the server after `traj extract` to include updated feature values.
Rows mode uses server-side pagination and retains per-column text filters.
Per-column Rows filters are independent of Perspective filters.
The page follows the system colour scheme when an analysis viewer is created.

## Rebuild frontend dependencies

Frontend dependency versions and their transitive dependencies are pinned in `web/package.json` and `web/package-lock.json`.
From this repository root, run:

```sh
npm ci --prefix web --ignore-scripts
npm run build --prefix web
```

The build writes the distributable assets to `src/traj_analyzer/viewer/static/vendor/`.
Include these assets when publishing the Python package so that end users do not need Node.js.
