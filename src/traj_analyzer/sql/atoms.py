from __future__ import annotations

from sqlglot import exp

# Aggregate functions by the name an atom uses for them; percentile and deviation variants share one name.
AGGREGATES: dict[type[exp.Expression], str] = {
    exp.Count: "count", exp.Avg: "avg", exp.Sum: "sum", exp.Min: "min", exp.Max: "max",
    exp.Quantile: "quantile", exp.ApproxQuantile: "quantile", exp.PercentileCont: "quantile",
    exp.PercentileDisc: "quantile", exp.Median: "quantile",
    exp.Stddev: "stddev", exp.StddevPop: "stddev", exp.StddevSamp: "stddev", exp.Variance: "stddev",
}
AGGREGATE_ROLES = frozenset(AGGREGATES.values())
COMPARISONS: dict[type[exp.Expression], str] = {exp.EQ: "=", exp.In: "=", exp.NEQ: "!=", exp.Like: "like",
                                                exp.ILike: "ilike"}


def defined_names(tree: exp.Expression) -> set[str]:
    """Names a query defines itself, as aliases or CTEs, which are not data columns.

    An alias that only repeats a column's own name, as in a.service_name AS service_name, still names that column.
    """
    aliases = {a.alias.lower() for a in tree.find_all(exp.Alias)
               if not (isinstance(a.this, exp.Column) and a.this.name.lower() == a.alias.lower())}
    return aliases | {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}


def _select_item(select: exp.Select, key: exp.Expression) -> exp.Expression:
    """The select-list expression a GROUP BY or ORDER BY key stands for when it is an ordinal or an alias."""
    if isinstance(key, exp.Literal) and key.is_int and 0 < int(key.this) <= len(select.expressions):
        key = select.expressions[int(key.this) - 1]
        return key.this if isinstance(key, exp.Alias) else key
    if isinstance(key, exp.Column) and not key.table:
        for item in select.expressions:
            if isinstance(item, exp.Alias) and item.alias.lower() == key.name.lower():
                return item.this
    return key


def atoms(tree: exp.Expression | None) -> set[str]:
    """`<role>:<column>` for every data column of every SELECT of a query.

    Roles: filter (WHERE, HAVING, or the FILTER clause of an aggregate), group (GROUP BY), order (ORDER BY), join
    (JOIN ... ON), the aggregate name for a column inside an aggregate of the select list, and show for any other
    column of the select list. SELECT * gives show:* and COUNT(*) gives count:*. GROUP BY and ORDER BY keys written as
    an ordinal or an alias count as the select-list expression they stand for. Other names the query defines itself as
    aliases or CTEs are not data columns.
    """
    if tree is None:
        return set()
    aliases = defined_names(tree)
    found: set[str] = set()

    def mark(scope: exp.Expression | None, role: str) -> None:
        if scope is None:
            return
        for column in scope.find_all(exp.Column):
            name = column.name.lower()
            if name and name not in aliases:
                found.add(f"{role}:{name}")

    for select in tree.find_all(exp.Select):
        mark(select.args.get("where"), "filter")
        mark(select.args.get("having"), "filter")
        for clause, role in (("group", "group"), ("order", "order")):
            keys = select.args.get(clause)
            for key in keys.expressions if keys else []:
                mark(_select_item(select, key.this if isinstance(key, exp.Ordered) else key), role)
        for join in select.args.get("joins") or []:
            mark(join.args.get("on"), "join")
        for expression in select.expressions:
            if isinstance(expression, exp.Star) or isinstance(expression, exp.Column) and expression.is_star:
                found.add("show:*")
            for filtered in expression.find_all(exp.Filter):
                mark(filtered.expression, "filter")
            aggregates = list(expression.find_all(*AGGREGATES))
            for aggregate in aggregates:
                mark(aggregate, AGGREGATES[type(aggregate)])
                if isinstance(aggregate, exp.Count) and isinstance(aggregate.this, exp.Star):
                    found.add("count:*")
            if not aggregates:
                mark(expression, "show")
    return found


def _compared_column(side: exp.Expression) -> tuple[str, str] | None:
    """How one side of a comparison reads a data column, and that column's name.

    The side is a column, or a function of exactly one column, written as lower(level).
    """
    if isinstance(side, exp.Column):
        return side.name.lower(), side.name.lower()
    if isinstance(side, exp.Func):
        columns = list(side.find_all(exp.Column))
        if len(columns) == 1:
            name = columns[0].name.lower()
            return f"{type(side).__name__.lower()}({name})", name
    return None


def filter_values(tree: exp.Expression | None, skip: frozenset[str] = frozenset()) -> set[str]:
    """`<column> <op> <value>` for string literals compared with a data column, op being =, !=, like or ilike.

    IN counts as =. Values keep the query's exact text, case and LIKE wildcards included, because a database
    compares them exactly and a value the data does not hold matches nothing without an error. A column wrapped in
    one function is named with it, as lower(level). The columns in `skip`, such as identifiers whose values name
    one row of a system, and names the query defines itself are left out.
    """
    if tree is None:
        return set()
    aliases = defined_names(tree)
    found = set()
    for node in tree.find_all(*COMPARISONS):
        sides = [node.this] if isinstance(node, exp.In) else [node.this, node.expression]
        columns = [c for c in map(_compared_column, sides) if c]
        if not columns:
            continue
        column, name = columns[0]
        if name in skip or name in aliases:
            continue
        values = node.expressions if isinstance(node, exp.In) else sides
        for value in values:
            if isinstance(value, exp.Literal) and value.is_string:
                found.add(f"{column} {COMPARISONS[type(node)]} {value.this}")
    return found
