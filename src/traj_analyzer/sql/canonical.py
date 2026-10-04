from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sqlglot import exp

COMMUTATIVE = (exp.EQ, exp.NEQ, exp.Add, exp.Mul)
# Node kinds that count as a part of a query: conditions, function calls, joins, sort keys and table reads.
PARTS = (exp.Predicate, exp.Func, exp.Join, exp.Ordered, exp.Table, exp.Case, exp.Window)


def literal_template(tree: exp.Expression, dialect: str, name_table: Callable[[str], str] = str.lower) -> str:
    """The query as written, with every literal replaced by a placeholder and every table read through a string
    literal, such as a file path, named by `name_table`."""

    def replace(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.Table):
            paths = [lit.this for lit in node.find_all(exp.Literal) if lit.is_string]
            if paths:
                name = "+".join(sorted(name_table(path) for path in paths))
                return exp.Table(this=exp.to_identifier(name), alias=node.args.get("alias"))
        if isinstance(node, exp.Literal):
            return exp.Placeholder()
        return node

    return tree.copy().transform(replace).sql(dialect=dialect, normalize=True)


class Canonical:
    """The form of a resolved query without what its author chose freely: names and the order of commutative parts.

    Aliases are replaced by what they stand for, CTEs are written where they are read, literals are placeholders,
    and the operands of AND, OR, = and +, the select list, the joins and the GROUP BY keys are sorted. Two queries
    with the same `text` ask the same thing. `parts` holds the text of every condition, function call, join, sort
    key and table read, so two queries can be compared by the parts they share. Table qualifiers of columns are
    dropped, so the two sides of a self-join are not told apart.
    """

    def __init__(self, tree: exp.Expression, name_table: Callable[[str], str] = str.lower) -> None:
        self.name_table = name_table
        self.ctes = {cte.alias_or_name.lower(): cte.this for cte in tree.find_all(exp.CTE)}
        self.aliases: dict[str, exp.Expression] = {}
        for alias in tree.find_all(exp.Alias):
            name = alias.alias.lower()
            identity = isinstance(alias.this, exp.Column) and alias.this.name.lower() == name
            if not identity and name not in self.aliases:
                self.aliases[name] = alias.this
        self.resolving: set[str] = set()
        self.parts: set[str] = set()
        self.text = self.render(tree)

    def render(self, node: Any) -> str:
        if isinstance(node, list):
            return "[" + ",".join(self.render(item) for item in node) + "]"
        if not isinstance(node, exp.Expression):
            return str(node).lower()
        text = self._expression(node)
        if isinstance(node, PARTS):
            self.parts.add(text)
        return text

    def _sorted(self, nodes: list[Any]) -> str:
        return "{" + ",".join(sorted(self.render(node) for node in nodes)) + "}"

    def _expression(self, node: exp.Expression) -> str:
        if isinstance(node, (exp.Alias, exp.Paren, exp.Subquery)):
            return self.render(node.this)
        if isinstance(node, exp.Literal):
            return "?"
        if isinstance(node, exp.Identifier):
            return node.name.lower()
        if isinstance(node, exp.Column):
            return self._column(node)
        if isinstance(node, exp.Table):
            return self._table(node)
        if isinstance(node, exp.In) and not node.args.get("query"):
            return f"in({self.render(node.this)},?)"
        if isinstance(node, (exp.Limit, exp.Offset)):
            return type(node).__name__.lower() + "(?)"
        if isinstance(node, (exp.And, exp.Or)):
            return type(node).__name__.lower() + self._sorted(list(node.flatten()))
        if isinstance(node, COMMUTATIVE):
            return type(node).__name__.lower() + self._sorted([node.this, node.expression])
        if isinstance(node, exp.Select):
            return self._select(node)
        if isinstance(node, exp.Anonymous):
            return f"{node.name.lower()}({self.render(node.expressions)})"
        parts = [f"{key}={self.render(value)}" for key, value in sorted(node.args.items())
                 if value not in (None, False, [])]
        return f"{type(node).__name__.lower()}({','.join(parts)})"

    def _column(self, node: exp.Column) -> str:
        if node.is_star:
            return "*"
        name = node.name.lower()
        if name in self.aliases and name not in self.resolving:
            self.resolving.add(name)
            text = self.render(self.aliases[name])
            self.resolving.discard(name)
            return text
        return name

    def _table(self, node: exp.Table) -> str:
        paths = [lit.this for lit in node.find_all(exp.Literal) if lit.is_string]
        if paths:
            return "table(" + "+".join(sorted(self.name_table(path) for path in paths)) + ")"
        name = node.name.lower()
        if name in self.ctes and name not in self.resolving:
            self.resolving.add(name)
            text = self.render(self.ctes[name])
            self.resolving.discard(name)
            return text
        return f"table({self.name_table(node.name)})"

    def _select(self, node: exp.Select) -> str:
        args = node.args
        group = args.get("group")
        parts = [
            "show" + self._sorted(node.expressions),
            "from(" + self.render(args.get("from_") or "") + ")",
            "join" + self._sorted(args.get("joins") or []),
            "where(" + self.render(args["where"].this if args.get("where") else "") + ")",
            "group" + self._sorted(group.expressions if group else []),
            "having(" + self.render(args["having"].this if args.get("having") else "") + ")",
            "order(" + self.render(args["order"].expressions if args.get("order") else []) + ")",
            "limit" if args.get("limit") else "",
            "distinct" if args.get("distinct") else "",
        ]
        return "select(" + ";".join(parts) + ")"
