# Workspace of a hunt

| Path | Content | Written by |
|---|---|---|
| `need.md` | what the person is looking for | the driver |
| `trajectories/<dataset>/<id>.md` | every trajectory of the hunt, with long steps cut | the driver |
| `tables/<name>.parquet` | the trajectories parsed into tables, listed below | the driver |
| `query.py` | runs one SQL query over these tables | the driver |
| `scenario.md` | what the trajectories are, when a mining round described them | the driver |
| `facets.json` | the kinds of behaviour the need covers, each with what does not count | the driver, from the proposals and the verdicts |
| `probes.json`, `probes/` | every probe with its status, and the text of the active ones | the driver, from the proposals |
| `ledger.json` | per probe its hits, reads, confirmations, precision and whether to revise it; per facet the confirmed trajectories; per round what was read and found; the coverage | the driver |
| `findings.json` | every trajectory confirmed to show the need | the driver |
| `rounds/<round>/hits.parquet` | every position each probe hit in the round | the driver |
| `rounds/<round>/verdicts.json` | every trajectory read in the round: why it was read, where probes hit it, the verdict and why | the driver |
| `rounds/<round>/cases/<dataset>/<id>.md` | the trajectories read in the round, whole and uncut | the driver |
| `operators/` | `schema.py`, `base.py`, `tables.py`, `rules.py`, the example files, and the operator files of code probes | the driver and `hunt-propose` |
| `notes/` | working notes of the proposals | `hunt-propose` |

`source` in `verdicts.json` says why a trajectory was read: `hit`, a probe hit it; `coverage`, it was drawn from the cluster of trajectories read least; `random`, it was drawn at random from the trajectories no probe hits; `final`, the closing check read it.

## Tables

Every table has a `key` column, `<dataset>/<id>`, and holds only the trajectories of the hunt.

{tables}
Query the tables to see something across every trajectory at once, and read the trajectory files for the course of one.

```
{python} query.py "<SQL>"
```

The query runs in DuckDB with the workspace as its directory, and prints at most 200 rows.
A SQL probe is such a query, returning the columns `key`, `step` and `note`; count its hits before you return it:

```
{python} query.py "SELECT count(*) AS positions, count(DISTINCT key) AS trajectories FROM (<probe>)"
```

Queries that help write probes:

```
-- the first message of the user in every trajectory, which usually states the task
SELECT key, left(content, 300) FROM 'tables/steps.parquet' WHERE role = 'user' AND kind = 'message'
QUALIFY row_number() OVER (PARTITION BY key ORDER BY step) = 1

-- a pattern in the whole text of the tool calls, with the step where it shows
SELECT key, step, name, left(content, 200) AS note FROM 'tables/contents.parquet'
WHERE kind = 'tool_call' AND regexp_matches(content, '<pattern>')

-- every edit, creation and deletion of a file, with the files and the outcome
SELECT key, call AS step, act, program, outcome, objects FROM 'tables/actions.parquet'
WHERE act IN ('edit', 'create', 'delete') ORDER BY key, step

-- what follows a failed action in the same trajectory
SELECT a.key, a.call AS step, a.act, a.program, b.act AS next_act, b.program AS next_program
FROM 'tables/actions.parquet' a JOIN 'tables/actions.parquet' b ON a.key = b.key AND b.position = a.position + 1
WHERE a.outcome = 'error'

-- the last assistant message of every trajectory, where an agent says how the task ended
SELECT key, step, left(content, 300) FROM 'tables/contents.parquet' WHERE role = 'assistant' AND kind = 'message'
QUALIFY row_number() OVER (PARTITION BY key ORDER BY step DESC) = 1

-- the events the rules of the project found
SELECT rule, edge, recovered, count(*) AS events, count(DISTINCT key) AS trajectories
FROM 'tables/events.parquet' GROUP BY ALL ORDER BY events DESC
```

## Commands

Python interpreter: `{python}`

Compute a code probe on the trajectories of the project and print its events for the trajectories you name:

```
{traj} try operators/<name>.py --key <dataset>/<id> --key <dataset>/<id>
```

`values` gives the events of every `--key`; in `spread`, `varying` counts the trajectories whose events differ from the most common value, which is no event at all for most probes.
