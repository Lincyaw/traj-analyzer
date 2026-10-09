from __future__ import annotations

from collections import Counter
from typing import Any

from traj_analyzer.layers.tables import Tables
from traj_analyzer.operators.base import Evidenced, FeatureSpec, NoParams, Operator, has_user_messages
from traj_analyzer.schema import Trajectory

SHORT_PROMPT = 20


def outputs(params: NoParams) -> list[FeatureSpec]:
    return [
        FeatureSpec(name="interjections", type="scalar",
                    description="Messages the user typed while the assistant was calling tools: user messages that "
                                "follow a tool call or tool result with no assistant message in between."),
        FeatureSpec(name="repeated_prompts", type="scalar",
                    description="Messages the user typed whose text repeats an earlier message of the user."),
        FeatureSpec(name="short_prompt_share", type="scalar", range=(0, 1),
                    description=f"Share of the messages the user typed after the first with at most {SHORT_PROMPT} "
                                "characters, such as `yes` or `continue`. Empty with one message."),
        FeatureSpec(name="prompt_chars_median", type="scalar",
                    description="Median characters of the messages the user typed."),
    ]


def _median(values: list[int]) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    middle = len(ordered) // 2
    return float(ordered[middle]) if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def compute(trajectory: Trajectory, tables: Tables, params: NoParams) -> dict[str, Any]:
    prompts = [step for step in trajectory.steps if step.is_user_message]
    interjections = []
    before = None
    for step in trajectory.steps:
        if step.is_user_message and before is not None and before.kind in ("tool_call", "tool_result"):
            interjections.append(step.index)
        if step.role in ("assistant", "tool") or step.is_user_message:
            before = step
    seen: Counter[str] = Counter()
    repeated = []
    for step in prompts:
        text = step.content.strip()
        if seen[text]:
            repeated.append(step.index)
        seen[text] += 1
    later = [len(step.content.strip()) for step in prompts[1:]]
    return {
        "interjections": Evidenced(float(len(interjections)), tuple(interjections)),
        "repeated_prompts": Evidenced(float(len(repeated)), tuple(repeated)),
        "short_prompt_share": sum(n <= SHORT_PROMPT for n in later) / len(later) if later else None,
        "prompt_chars_median": _median([len(step.content.strip()) for step in prompts]),
    }


OPERATOR = Operator(
    kind="code",
    description="How the user takes part: interjecting while the assistant works, repeating a message, and how "
                "long the messages are.",
    tags=("chat", "agent"),
    outputs=outputs,
    compute=compute,
    requires=has_user_messages,
)
