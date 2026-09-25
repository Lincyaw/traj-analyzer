from __future__ import annotations

from traj_analyzer.operators.base import FeatureSpec, NoParams, Operator, has_tool_calls


def outputs(params: NoParams) -> list[FeatureSpec]:
    return [
        FeatureSpec(name="task_type", type="category", description="Kind of request.",
                    labels={"lookup": "Answer from tools", "creative": "Write something", "other": "Else"}),
        FeatureSpec(name="frustration", type="scalar", range=(0, 1), thresholds={"high": 0.7},
                    description="User dissatisfaction."),
        FeatureSpec(name="curve", type="vector", per="chunk", range=(0, 1), description="Dissatisfaction per chunk."),
        FeatureSpec(name="time_split", type="map", range=(0, 1), description="Where effort went.",
                    labels={"tools": "Calling tools", "answer": "Writing the answer"}),
    ]


OPERATOR = Operator(
    kind="llm",
    description="The feature set of the recorded model runs in tests/data/replay.",
    outputs=outputs,
    requires=has_tool_calls,
)
