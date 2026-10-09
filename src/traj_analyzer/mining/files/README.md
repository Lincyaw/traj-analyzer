# Workspace of a mining round

| Path | Content | Written by |
|---|---|---|
| `trajectories/<dataset>/<id>.md` | the trajectories that may be read | the driver |
| `tables/<name>.parquet` | the readable trajectories parsed into tables, listed below | the driver |
| `query.py` | runs one SQL query over these tables | the driver |
| `features.csv` | one row per readable trajectory: `key`, `batch`, `stratum`, and the value of every extracted feature | the driver |
| `features.md` | definition and role of every extracted feature | the driver |
| `start.txt` | keys of trajectories that differ strongly under the extracted features | the driver |
| `scenario.md` | what the trajectories are; the notes of the round before until `mine-survey` returns | `mine-survey` |
| `hypotheses.json` | measures and expectations that may matter in the scenario, each with the trajectories it comes from | `mine-hypothesize` |
| `findings.json` | what the trajectories show for each hypothesis | `mine-explore` |
| `candidates.json` | feature definitions with examples and operator files | `mine-define` |
| `reviews.json` | keep or reject for every candidate, with what was checked | `mine-review` |
| `checks.json` | verdict of the checks on every candidate | the driver |
| `previous/checks.json` | the verdicts of the round before | the driver |
| `operators/` | `schema.py`, `base.py`, `tables.py`, `rules.py`, the example files, and the operator files of this round | `mine-define` |
| `cases/<dataset>/<id>.md` | trajectories the checks show to a function | the driver |
| `notes/` | working notes of the functions | every function |

The driver writes each function's submitted result to its file, so a file holds what was accepted.

## Tables

Every table has a `key` column, `<dataset>/<id>`, and holds only the readable trajectories.

{tables}
Query the tables to see something across every trajectory at once, and read the trajectory files for the course of one.

```
{python} query.py "<SQL>"
```

The query runs in DuckDB with the workspace as its directory, and prints at most 200 rows.
Queries that answer common questions:

```
-- the first message of the user in every trajectory, which usually states the task
SELECT key, left(content, 300) FROM 'tables/steps.parquet' WHERE role = 'user' AND kind = 'message'
QUALIFY row_number() OVER (PARTITION BY key ORDER BY step) = 1

-- every tool, with the number of calls, of trajectories that call it, and of failed results
SELECT name, count(*) FILTER (kind = 'tool_call') AS calls, count(DISTINCT key) AS trajectories,
       count(*) FILTER (kind = 'tool_result' AND is_error) AS errors
FROM 'tables/steps.parquet' WHERE name IS NOT NULL GROUP BY name ORDER BY calls DESC

-- what the agent does, by action and outcome
SELECT act, program, outcome, count(*) AS actions, count(DISTINCT key) AS trajectories
FROM 'tables/actions.parquet' GROUP BY ALL ORDER BY actions DESC

-- what follows a failed action in the same trajectory
SELECT a.act, a.program, b.act AS next_act, b.program AS next_program, count(*) AS times
FROM 'tables/actions.parquet' a JOIN 'tables/actions.parquet' b
ON a.key = b.key AND b.position = a.position + 1
WHERE a.outcome = 'error' GROUP BY ALL ORDER BY times DESC

-- the events found so far, and whether a later action made up for them
SELECT rule, edge, recovered, count(*) AS events, count(DISTINCT key) AS trajectories
FROM 'tables/events.parquet' GROUP BY ALL ORDER BY events DESC

-- the argument keys of one tool
SELECT json_keys(content) AS keys, count(*) FROM 'tables/steps.parquet'
WHERE kind = 'tool_call' AND name = '<tool>' AND json_valid(content) GROUP BY ALL ORDER BY 2 DESC

-- openings that repeat across trajectories: system prompts, tool and skill definitions, generated messages
SELECT role, kind, left(content, 80) AS opening, count(DISTINCT key) AS trajectories
FROM 'tables/steps.parquet' WHERE kind = 'message' GROUP BY ALL HAVING count(DISTINCT key) > 1 ORDER BY 4 DESC

-- how trajectories end
SELECT role, kind, count(*) FROM 'tables/steps.parquet'
QUALIFY row_number() OVER (PARTITION BY key ORDER BY step DESC) = 1 GROUP BY ALL
```

## Commands

Python interpreter: `{python}`

Compute an operator file on every trajectory of the reference set:

```
{traj} try operators/<name>.py --key <dataset>/<id> --key <dataset>/<id>
```

It prints `spread`: the number of trajectories with a value, the distinct values, how many trajectories differ from the most common value, the most common values, `confined` when the trajectories that differ sit in one value of a context feature, and `batch_explained`, the share of the variation the batches explain inside the strata; `objection` when the checks would reject the feature for its spread; `values`, the value for every `--key`; and `steps`, the steps each of those values rests on.
