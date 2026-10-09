from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from traj_analyzer.layers.tables import Tables
from traj_analyzer.operators.base import Edge, Event, EventMark, FeatureSpec

Severity = Literal["low", "medium", "high"]
Unit = Literal["action", "turn"]


@dataclass(frozen=True)
class Context:
    """What a rule reads for one trajectory: the actions, their relations, the steps and the turns, as rows."""

    actions: list[dict[str, Any]]
    relations: list[dict[str, Any]]
    steps: list[dict[str, Any]] = field(default_factory=list)
    turns: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def of(cls, tables: Tables) -> Context:
        return cls(actions=tables.records("actions"), relations=tables.records("relations"),
                   steps=tables.records("steps") if "steps" in tables else [],
                   turns=tables.records("turns") if "turns" in tables else [])

    def related(self, kind: str) -> list[tuple[int, int]]:
        """(later, earlier) positions of the relations of this kind."""
        return [(r["later"], r["earlier"]) for r in self.relations if r["kind"] == kind]

    def objects(self, position: int) -> list[str]:
        return [o["name"] for o in self.actions[position]["objects"] if o["kind"] in ("file", "directory")]

    def turn_of(self, step: int) -> dict[str, Any] | None:
        """The turn a step lies in, by the turns table."""
        return next((turn for turn in self.turns if turn["start"] <= step <= turn["end"]), None)


@dataclass(frozen=True)
class Rule:
    """One expectation and how it is broken.

    `precondition` gives the positions where the expectation applies: positions in the actions of the context for
    a rule over actions, positions in its turns for a rule over turns. `expectation` tells whether it was met at
    one of them; an event is every position where it was not. `recovery` gives the position of the later action
    that made up for the event, or None. `note` says what was expected and what happened instead.
    """

    name: str
    description: str
    edge: Edge
    severity: Severity
    precondition: Callable[[Context], list[int]]
    expectation: Callable[[Context, int], bool]
    recovery: Callable[[Context, int], int | None]
    note: Callable[[Context, int], str] = lambda context, position: ""
    unit: Unit = "action"
    about_consequence: bool = False
    """Whether the rule fires on a consequence itself, such as the user interrupting; such a rule gets no
    consequence lift, since the consequence is what it finds."""

    def spec(self) -> FeatureSpec:
        return FeatureSpec(name=self.name, type="event", description=self.description, event=EventMark(edge=self.edge))

    def events(self, context: Context) -> list[Event]:
        found = []
        for position in self.precondition(context):
            if self.expectation(context, position):
                continue
            recovered_at = self.recovery(context, position)
            if self.unit == "turn":
                step, objects = context.turns[position]["start"], []
            else:
                step, objects = context.actions[position]["call"], context.objects(position)
            found.append(Event(
                step=step, objects=objects, recovered=recovered_at is not None,
                recovered_at=context.actions[recovered_at]["call"] if recovered_at is not None else None,
                note=self.note(context, position),
            ))
        return found


def events_of(rules: list[Rule], tables: Tables) -> dict[str, list[dict[str, Any]]]:
    """The events of every rule on one trajectory, as the values of its event features."""
    context = Context.of(tables)
    return {rule.name: [event.model_dump(mode="json") for event in rule.events(context)] for rule in rules}
