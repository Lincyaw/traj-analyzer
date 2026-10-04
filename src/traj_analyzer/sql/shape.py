from __future__ import annotations

from dataclasses import dataclass

from sqlglot import exp

from traj_analyzer.sql.atoms import AGGREGATE_ROLES, atoms, filter_values
from traj_analyzer.sql.parsing import tables

# Roles that say which rows a query looks at; the other roles say what it computes and shows of them.
SCOPE_ROLES = ("filter", "group", "join")


@dataclass(frozen=True)
class Shape:
    """One query as facets, each a sorted tuple of labels, from what it asks to how it is written.

    `tables` is what it reads, `scope` the columns that say which rows it looks at, `measures` the columns inside
    aggregates, `aggregates` those columns with their aggregate, `shown` the columns it shows or orders by, and
    `values` the string values it filters on.
    """

    tables: tuple[str, ...]
    scope: tuple[str, ...]
    measures: tuple[str, ...]
    aggregates: tuple[str, ...]
    shown: tuple[str, ...]
    values: tuple[str, ...]


def shape(tree: exp.Expression, skip: frozenset[str] = frozenset()) -> Shape:
    """The facets of a parsed query; `skip` names the columns whose filter values are left out."""
    roles = [atom.split(":", 1) for atom in sorted(atoms(tree))]
    return Shape(
        tables=tuple(sorted(tables(tree))),
        scope=tuple(f"{role}:{column}" for role, column in roles if role in SCOPE_ROLES),
        measures=tuple(sorted({column for role, column in roles if role in AGGREGATE_ROLES})),
        aggregates=tuple(f"{role}:{column}" for role, column in roles if role in AGGREGATE_ROLES),
        shown=tuple(f"{role}:{column}" for role, column in roles if role in ("show", "order")),
        values=tuple(sorted(filter_values(tree, skip))),
    )
