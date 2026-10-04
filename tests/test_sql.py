from __future__ import annotations

from traj_analyzer import sql

DIALECT = "duckdb"
# Queries agents wrote in RCABench runs, with file paths shortened to the file name.
LOGS = "SELECT service_name, level, count(*) AS n FROM abnormal_logs GROUP BY 1,2 ORDER BY n DESC LIMIT 25"
LOGS_REWRITTEN = ("select level, service_name, COUNT(*) as cnt from abnormal_logs "
                  "group by service_name, level order by cnt desc limit 40")
TRACES = ("SELECT service_name, \"attr.status_code\" as status, COUNT(*) cnt, AVG(duration) avg_dur, "
          "MAX(duration) max_dur FROM read_parquet('/data/case/normal_traces.parquet') "
          "WHERE service_name IN ('frontend', 'cart') AND lower(level) = 'error' AND \"attr.status_code\" = 'Error' "
          "GROUP BY 1,2 ORDER BY cnt DESC")
JOINED = ("SELECT t1.trace_id, t2.span_name as child_span FROM abnormal_traces t1 LEFT JOIN abnormal_traces t2 "
          "ON t1.trace_id = t2.trace_id AND t2.parent_span_id = t1.span_id "
          "WHERE t1.service_name = 'frontend' AND t2.span_name = 'user.User/CheckUser' LIMIT 20")
COMPARED = ("WITH a AS (SELECT service_name, avg(duration)/1e6 AS ms FROM abnormal_traces GROUP BY 1), "
            "n AS (SELECT service_name, avg(duration)/1e6 AS ms FROM normal_traces GROUP BY 1) "
            "SELECT a.service_name, a.ms, n.ms FROM a JOIN n ON a.service_name = n.service_name")


def tree(text: str):
    parsed = sql.parse(text, DIALECT)
    assert parsed is not None
    return parsed


def test_parse_returns_none_for_invalid_sql() -> None:
    assert sql.parse("SELECT FROM WHERE", DIALECT) is None
    assert sql.atoms(None) == set() and sql.tables(None) == set() and sql.filter_values(None) == set()


def test_tables_name_identifiers_and_file_paths() -> None:
    assert sql.tables(tree(LOGS)) == {"abnormal_logs"}
    assert sql.tables(tree(TRACES)) == {"/data/case/normal_traces.parquet"}
    assert sql.tables(tree(COMPARED)) == {"a", "n", "abnormal_traces", "normal_traces"}


def test_atoms_give_every_column_its_role() -> None:
    assert sql.atoms(tree(LOGS)) == {"show:service_name", "show:level", "count:*", "group:service_name",
                                     "group:level"}
    assert sql.atoms(tree(TRACES)) == {
        "show:service_name", "show:attr.status_code", "count:*", "avg:duration", "max:duration",
        "filter:service_name", "filter:level", "filter:attr.status_code", "group:service_name",
        "group:attr.status_code"}
    assert {"join:trace_id", "join:parent_span_id", "join:span_id"} <= sql.atoms(tree(JOINED))
    # Names the query defines itself are not data columns.
    assert not any(atom.endswith(":ms") or atom.endswith(":cnt") for text in (TRACES, COMPARED)
                   for atom in sql.atoms(tree(text)))


def test_filter_values_keep_the_exact_text_and_skip_identifiers() -> None:
    values = sql.filter_values(tree(TRACES))
    assert values == {"service_name = frontend", "service_name = cart", "lower(level) = error",
                      "attr.status_code = Error"}
    assert sql.filter_values(tree(TRACES), frozenset({"service_name"})) == {"lower(level) = error",
                                                                            "attr.status_code = Error"}


def test_shape_orders_facets_from_what_is_asked_to_how() -> None:
    shape = sql.shape(tree(TRACES), frozenset({"service_name"}))
    assert shape == sql.Shape(
        tables=("/data/case/normal_traces.parquet",),
        scope=("filter:attr.status_code", "filter:level", "filter:service_name", "group:attr.status_code",
               "group:service_name"),
        measures=("*", "duration"),
        aggregates=("avg:duration", "count:*", "max:duration"),
        shown=("show:attr.status_code", "show:service_name"),
        values=("attr.status_code = Error", "lower(level) = error"),
    )


def canonical(text: str) -> sql.Canonical:
    return sql.Canonical(sql.resolve(tree(text), DIALECT))


def test_canonical_form_ignores_names_order_and_literals() -> None:
    assert canonical(LOGS).text == canonical(LOGS_REWRITTEN).text
    assert sql.literal_template(tree(LOGS), DIALECT) != sql.literal_template(tree(LOGS_REWRITTEN), DIALECT)
    swapped = JOINED.replace("t1.trace_id = t2.trace_id AND t2.parent_span_id = t1.span_id",
                             "t1.span_id = t2.parent_span_id AND t2.trace_id = t1.trace_id")
    assert canonical(JOINED).text == canonical(swapped).text
    assert canonical(JOINED).text != canonical(JOINED.replace("LEFT JOIN", "JOIN")).text
    other_value = TRACES.replace("'Error'", "'ERROR'")
    assert canonical(TRACES).text == canonical(other_value).text


def test_canonical_form_names_tables_and_inlines_ctes() -> None:
    def kind(path: str) -> str:
        return path.rsplit("/", 1)[-1].removesuffix(".parquet").removeprefix("abnormal_").removeprefix("normal_")

    with_windows = sql.Canonical(sql.resolve(tree(COMPARED), DIALECT))
    merged = sql.Canonical(sql.resolve(tree(COMPARED), DIALECT), kind)
    assert "table(abnormal_traces)" in with_windows.parts and "table(normal_traces)" in with_windows.parts
    assert "table(traces)" in merged.parts and "table(a)" not in merged.text
    assert "avg(this=duration)" in merged.parts
    assert sql.literal_template(tree(TRACES), DIALECT, kind) == (
        'SELECT service_name, "attr.status_code" AS status, COUNT(*) AS cnt, AVG(duration) AS avg_dur, '
        'MAX(duration) AS max_dur FROM traces WHERE service_name IN (?, ?) AND LOWER(level) = ? AND '
        '"attr.status_code" = ? GROUP BY ?, ? ORDER BY cnt DESC')
