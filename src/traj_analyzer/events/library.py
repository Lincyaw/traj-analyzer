from __future__ import annotations

from traj_analyzer.events.rules import Context, Rule

SIDE_EFFECTS = frozenset({"destructive", "remote"})
FAILURE_RUN = 3
"""Consecutive failed calls that make a failure run."""


def _same_act_later(context: Context, position: int, outcome: str = "ok") -> int | None:
    """The first later action of the same act on an object of the action, ending with `outcome`."""
    action = context.actions[position]
    objects = set(context.objects(position))
    for later in range(position + 1, len(context.actions)):
        candidate = context.actions[later]
        if candidate["act"] != action["act"] or candidate["outcome"] != outcome:
            continue
        if not objects or objects & set(context.objects(later)):
            return later
    return None


def _unanswered(context: Context) -> list[int]:
    return [i for i, a in enumerate(context.actions) if a["ordinal"] == 0 and a["outcome"] == "unanswered"]


def _retries(context: Context) -> list[int]:
    """Retries whose action repeats the failed action with the same arguments."""
    return sorted({later for later, earlier in context.related("retry")
                   if context.actions[later]["text"] == context.actions[earlier]["text"]
                   and context.objects(later) == context.objects(earlier)})


def _side_effects(context: Context) -> list[int]:
    return [i for i, a in enumerate(context.actions) if a["effect"] in SIDE_EFFECTS]


def _asked_before(context: Context, position: int) -> bool:
    """Whether the user typed a message or was asked since the previous side effect, or since the start."""
    action = context.actions[position]
    for earlier in range(position - 1, -1, -1):
        before = context.actions[earlier]
        if before["turn"] != action["turn"] or before["act"] == "ask":
            return True
        if before["effect"] in SIDE_EFFECTS:
            return False
    return True


def _failure_runs(context: Context) -> list[int]:
    """The action that completes each run of `FAILURE_RUN` consecutive failed calls."""
    found = []
    run = 0
    for i, action in enumerate(context.actions):
        if action["ordinal"] != 0:
            continue
        run = run + 1 if action["outcome"] == "error" else 0
        if run == FAILURE_RUN:
            found.append(i)
    return found


def _any_ok_later(context: Context, position: int) -> int | None:
    for later in range(position + 1, len(context.actions)):
        if context.actions[later]["ordinal"] == 0 and context.actions[later]["outcome"] == "ok":
            return later
    return None


def _nothing(context: Context, position: int) -> int | None:
    return None


def _never(context: Context, position: int) -> bool:
    return False


def _interrupted_turns(context: Context) -> list[int]:
    return [i for i, turn in enumerate(context.turns) if turn["ended_by"] == "interrupt"]


def _not_at_plan(context: Context, position: int) -> bool:
    """Whether the interrupted turn ended somewhere else than on a plan or a question."""
    return context.turns[position]["last_act"] not in ("plan", "ask")


def _had_output(context: Context, position: int) -> bool:
    turn = context.turns[position]
    return bool(turn["edits"] or turn["verifications"] or turn["messages"])


RULES = [
    Rule(
        name="result_unanswered", edge="tool", severity="medium",
        description="A tool call has no result in the trajectory. Recovered when a later call of the same act on "
                    "the same object ends ok.",
        precondition=_unanswered, expectation=_never, recovery=_same_act_later,
        note=lambda c, p: f"{c.actions[p]['tool']} call without a result",
    ),
    Rule(
        name="blind_retry", edge="tool", severity="medium",
        description="After a failed action, the agent runs the same action again with the same arguments. "
                    "Recovered when a later action of the same act on the object ends ok.",
        precondition=_retries, expectation=_never, recovery=_same_act_later,
        note=lambda c, p: f"repeated unchanged after a failure: {c.actions[p]['text'] or c.actions[p]['tool']}",
    ),
    Rule(
        name="side_effect_unasked", edge="owner", severity="high",
        description="A destructive or remote action follows another one with no message from the user and no "
                    "question to the user in between.",
        precondition=_side_effects, expectation=_asked_before, recovery=_nothing,
        note=lambda c, p: f"{c.actions[p]['effect']} action without a word from the user: "
                          f"{c.actions[p]['text'] or c.actions[p]['tool']}",
    ),
    Rule(
        name="failure_run", edge="tool", severity="medium",
        description=f"{FAILURE_RUN} consecutive tool calls fail with no success and no message from the user in "
                    "between. Recovered when a later call ends ok.",
        precondition=_failure_runs, expectation=_never, recovery=_any_ok_later,
        note=lambda c, p: f"{FAILURE_RUN} failed calls in a row, the last: "
                          f"{c.actions[p]['text'] or c.actions[p]['tool']}",
    ),
    Rule(
        name="interrupted_at_plan", edge="owner", severity="medium", unit="turn", about_consequence=True,
        description="The user interrupted a turn whose last action wrote or left a plan or asked the user: the "
                    "plan was not what the user wanted.",
        precondition=_interrupted_turns, expectation=_not_at_plan, recovery=_nothing,
        note=lambda c, p: f"interrupted after {c.turns[p]['last_program']}",
    ),
    Rule(
        name="interrupted_without_output", edge="owner", severity="medium", unit="turn", about_consequence=True,
        description="The user interrupted a turn before the agent edited, verified or answered anything: the "
                    "agent went the wrong way before producing output.",
        precondition=_interrupted_turns, expectation=_had_output, recovery=_nothing,
        note=lambda c, p: f"interrupted after {c.turns[p]['calls']} calls without an edit, a verification or an "
                          f"answer",
    ),
]
CONSEQUENCE_RULES = frozenset(rule.name for rule in RULES if rule.about_consequence)
"""The rules that fire on a consequence itself; the consequence lift leaves them out."""
