from __future__ import annotations

from traj_analyzer.operators.base import Aggregate, EventMark, FeatureSpec, NoParams, Operator, has_user_messages

LABELS = {
    "failure_report": "The message says that what the assistant produced does not work: an error, a failing check, "
                      "wrong behaviour. It applies whether or not the message also says how to fix it.",
    "rejection": "The message refuses, undoes or stops the assistant's work without saying what to do instead.",
    "repeat": "The message says again, in the same or other words, an instruction the user already gave in an "
              "earlier message of the file, because the assistant did not follow it.",
    "correction": "The message changes what the assistant did or was about to do: it points out a mistake, a "
                  "misunderstanding or a constraint the assistant missed, or it replaces the assistant's plan with "
                  "the user's. Information the user volunteers to turn the assistant around counts; the tone does "
                  "not matter.",
    "clarify": "The assistant asked a question, offered options or said what it needed, and the message answers, "
               "picks or supplies that, without faulting what the assistant did.",
    "continue": "The message takes the assistant's work as it stands and goes on from it: approval, thanks, a yes "
                "or go-ahead, or the next step that builds on the result or follows the assistant's plan.",
    "new_task": "The message asks for something that neither reacts to nor builds on the assistant's last work: "
                "the first message of the trajectory, or a fresh request about something else.",
    "other": "None of the above: a greeting, small talk, or a message that asks nothing and reacts to nothing.",
}
"""What a message the user typed says about the work before it, in the order the labels are decided in."""
PUSHBACK = ["correction", "rejection", "failure_report", "repeat"]
"""The reactions in which the user resists the work instead of carrying on from it."""

GUIDANCE = """\
Judge the message against what the assistant did or said last, under `### Before the message`, and against the
user's earlier messages in the file.
Go through the labels in the order they are listed, and give the first one that applies; a message with several
parts gets the earliest label that any part earns.

Where labels meet:

- `failure_report` beats `correction`: "it crashes on start, you need to close the file first" reports a failure.
- `correction` beats `clarify`: the message is `clarify` only when the assistant asked for what it brings; information
  the user brings unasked to change the assistant's course is `correction`.
- `correction` beats `continue`: the next step is `continue` when it builds on the assistant's result or follows its
  plan, and `correction` when it replaces or amends what the assistant did or planned.
- `continue` beats `new_task`: a request that uses or extends the assistant's result is `continue`; a request about
  something else is `new_task`.
- `repeat` needs an earlier message of the user in the file with the same instruction; a direction given for the
  first time is `correction`.

Examples, each after an assistant turn:

- "Ran it, the first request fails with TypeError: x is undefined." → `failure_report`
- "No, undo that." → `rejection`
- "Again: keep the output in English." after an earlier "Reply in English." → `repeat`
- "That field is optional, not required; the check should allow it to be missing." → `correction`
- "Use the existing helper instead of writing a new one." → `correction`
- "Postgres." after the assistant asked "Postgres or SQLite?" → `clarify`
- "Looks good, now do the same for the other module." → `continue`
- "Thanks!" → `continue`
- "Unrelated: how do I set the time zone of the container?" → `new_task`
- "hi" → `other`
"""
"""Decision order, boundaries and examples, rendered before the label definitions in the instruction."""


def outputs(params: NoParams) -> list[FeatureSpec]:
    return [FeatureSpec(
        name="reaction", type="category", labels=LABELS,
        description="How the message the user typed at the start of this turn reacts to the assistant's work in the "
                    "turns before: failure_report, rejection, repeat, correction, clarify, continue, new_task or "
                    "other, decided in that order. The first message of a trajectory is new_task unless it reacts "
                    "to work shown in the metadata or the opening.",
        event=EventMark(edge="owner", when=PUSHBACK),
    )]


def guidance(params: NoParams) -> str:
    return GUIDANCE


def aggregates(params: NoParams) -> list[Aggregate]:
    return [
        Aggregate(name="pushback_share", of="reaction", kind="share", values=PUSHBACK,
                  description="Share of the messages the user typed that correct, reject, report a failure of, or "
                              "repeat an instruction about the assistant's work."),
        Aggregate(name="pushbacks", of="reaction", kind="count", values=PUSHBACK,
                  description="Messages the user typed that correct, reject, report a failure of, or repeat an "
                              "instruction about the assistant's work."),
        Aggregate(name="failure_reports", of="reaction", kind="count", values=["failure_report"],
                  description="Messages the user typed that report that the assistant's work does not work."),
        Aggregate(name="repeats", of="reaction", kind="count", values=["repeat"],
                  description="Messages the user typed that repeat or re-explain an earlier instruction."),
        Aggregate(name="ended_with", of="reaction", kind="last",
                  description="The reaction of the last message the user typed."),
    ]


OPERATOR = Operator(
    kind="judgment",
    description="What every message the user typed says about the work before it: whether the user carries on from "
                "what the assistant did, corrects it, rejects it, reports that it does not work, or repeats an "
                "instruction.",
    tags=("chat", "agent"),
    outputs=outputs,
    aggregates=aggregates,
    guidance=guidance,
    unit="turn",
    requires=has_user_messages,
)
