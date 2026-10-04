from __future__ import annotations

from functools import lru_cache

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.qualify import qualify


@lru_cache(maxsize=256)
def parse(sql: str, dialect: str) -> exp.Expression | None:
    """The syntax tree of a query, or None when sqlglot cannot parse it.

    Agents write invalid SQL too, which the database also rejects; such a query has no atoms and reads nothing.
    Trees are cached, since several callers parse the same queries of one trajectory; callers only read them.
    """
    try:
        return sqlglot.parse_one(sql, read=dialect)
    except sqlglot.errors.SqlglotError:
        return None


def tables(tree: exp.Expression | None) -> set[str]:
    """What a query reads: its table names, and the string literals inside its table expressions, which are the
    paths of table functions such as read_parquet."""
    if tree is None:
        return set()
    names = set()
    for table in tree.find_all(exp.Table):
        names.add(table.name)
        names.update(lit.this for lit in table.find_all(exp.Literal) if lit.is_string)
    return {n for n in names if n}


def resolve(tree: exp.Expression, dialect: str) -> exp.Expression:
    """A copy of the tree with GROUP BY and ORDER BY ordinals replaced by what they stand for and every table
    given an alias. No schema is known, so `SELECT *` stays and unqualified columns keep no table.

    sqlglot fails on some queries it parsed; the error is raised as it comes.
    """
    return qualify(tree.copy(), dialect=dialect, validate_qualify_columns=False, infer_schema=False,
                   quote_identifiers=False, identify=False)
