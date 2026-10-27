---
name: traj-hunt
description: Run a traj-analyzer hunt from this agent session, which proposes the probes itself and has fresh subagents read the cases, until the search converges. Use when the user asks to find every trajectory that shows some behaviour, such as reward hacking, gaming a grader, destructive or unauthorised operations, or any other need written in hunts/<name>.yaml. Triggers include "找出…的轨迹", "搜一下有没有 reward hack", "跑个 hunt", "hunt for", "find the trajectories where".
---

# Running a hunt from an agent session

A hunt looks for every trajectory that shows what a person asks for, and is judged by the trajectories it finds and by how many it may still miss.
Section 18 of `docs/design.md` in the traj-analyzer repository describes it.
`traj hunt run <name>` runs the whole search unattended with aifn functions.
This skill runs the same search with you in the two roles those functions play: you propose the probes, and subagents you start read the cases.
The `traj hunt` commands keep everything else: they run the probes, draw the reads, check what you hand in, keep the ledger, and decide when the search ends.
Both ways use the same state, the same checks and the same instructions, so a run gives the same result whichever way its answers arrive.

## Before you start

- The project has ingested its trajectories and run `traj extract`; a hunt needs the tables and the extracted features.
- `hunts/<name>.yaml` holds `need`, in the words of the person who asks, and the sizes of the search; section 18.1 of the design lists the fields.
  Write the need with the user when it does not exist yet.
  For long trajectories, keep `reads_per_round` and `cases_per_call` small: every read is a whole trajectory, uncut.

## The loop

Every `traj hunt` command prints one JSON document: the `phase` of the run, the files of the next step relative to `workspace`, and `next`, the command that takes it.
`traj hunt status <name>` prints the same at any time, so an interrupted session continues from there.

1. Run `traj hunt start <name>`.
2. Phase `propose`: you write the proposal of the round.
   - Read the instruction at `proposal.instruction` in the workspace and follow it: it is the instruction of the aifn function `hunt-propose`, with where to write your answer.
   - Work in the workspace directory: query the tables with `query.py`, try code probes with `traj try`, read trajectory files, and write your notes under `notes/`.
   - Write the answer to `proposal.answer`, one JSON object as the schema next to it describes, then run `traj hunt propose <name>`.
   - When the command reports that the file does not fit its schema, correct the file and run it again.
3. Phase `verdicts`: subagents read the cases, one subagent per entry of `batches`, all of them started at once.
   - Give each subagent this task and nothing more: "Your working directory is `<workspace>`. Follow the instruction in `<workspace>/<instruction>` exactly, and write your answer to `<workspace>/<answer>`."
   - Do not tell a reader what you expect, what a probe was meant to find, or what other readers found; a reader that knows the proposer's reasoning confirms it.
   - When every answer is written, run `traj hunt verdicts <name>`.
   - When the command reports that an answer does not fit, start a new subagent for that batch with the same task and the error, and run the command again.
4. The command after the verdicts prints the next phase: `propose` for the next round, or `close` when `patience` rounds in a row found nothing new or `max_rounds` rounds have run.
5. Phase `close`: run `traj hunt close <name>`.
   When it prints phase `verdicts` with `final` true, the closing check reads the hits no round read; handle it as in step 3.
6. Phase `finished`: `report` holds the result.

## What you must not do

- Do not write a verdict yourself, and do not read the cases of a round to check the readers; the readers are the measurement.
- Do not edit `facets.json`, `probes.json`, `ledger.json`, `findings.json`, `rounds/<round>/verdicts.json` or anything under `.traj/hunts/<name>/<run>/` outside the workspace; the commands write them.
- In the workspace, write only under `notes/` and `operators/`, and the answer files the commands name.
- Do not change `hunts/<name>.yaml` in the middle of a run; start a new run for a new need.

## Telling the user

Report from `report`:

- whether the search converged, after how many rounds, and how many trajectories were read;
- every facet with its definition and the number of trajectories confirmed;
- the active probes with their precision;
- `coverage`: the share of trajectories read, the random reads of trajectories no probe hits, and `expected_missed`, the number of real trajectories the probes may still miss;
- `unverified`: hits of the active probes that no read checked.

The confirmed trajectories are in the table `hunt_findings`, which `traj query` and the `Data` page of `traj view` show; cite trajectories by their key and steps.
The facets and probes of the run are kept in `hunts/<name>/`, and the next run of the hunt starts from them.
