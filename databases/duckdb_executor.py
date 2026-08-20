"""Configured DuckDB read-only execution adapter.

This module owns database validation, resource limits, execution, and result
serialization. It is not an LLM tool and contains no LangGraph wiring.
"""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import date, datetime
from decimal import Decimal
from math import prod
from pathlib import Path

import duckdb
from config.project_paths import PROJECT_ROOT


FORBIDDEN_SQL = re.compile(
    r"\b(insert|update|delete|merge|create|drop|alter|truncate|attach|detach|copy|"
    r"export|import|install|load|call|pragma|vacuum)\b",
    re.IGNORECASE,
)

# These are generic execution budgets and are not tied to a benchmark query.
SQL_QUERY_TIMEOUT_SECONDS = float(os.getenv("SQL_QUERY_TIMEOUT_SECONDS", "5"))
SQL_MAX_ESTIMATED_INTERMEDIATE_ROWS = int(
    os.getenv("SQL_MAX_ESTIMATED_INTERMEDIATE_ROWS", "10000000")
)
SQL_MAX_PLAN_NODES = int(os.getenv("SQL_MAX_PLAN_NODES", "80"))

# Defense in depth: even if a future SQL policy misses an external table
# function, the database connection itself cannot read files, URLs, object
# storage, or external databases.
DUCKDB_SAFE_CONNECTION_CONFIG = {"enable_external_access": "false"}


def _database_path() -> Path:
    raw_path = os.getenv("DATA_AGENT_DATABASE_PATH", "").strip()
    path = Path(raw_path).expanduser() if raw_path else PROJECT_ROOT / "databases" / "data_agent.duckdb"
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


class SqlResourceLimitError(ValueError):
    """查询超过执行时间或执行计划预算。"""


class SqlTimeoutError(SqlResourceLimitError):
    """查询超过配置的墙钟执行时间。"""


def _json_value(value):
    if isinstance(value, (Decimal, date, datetime)):
        return str(value)
    return value


def validate_readonly_sql(sql: str) -> str:
    cleaned = sql.strip()
    if not cleaned:
        raise ValueError("SQL 不能为空。")

    without_trailing_semicolon = cleaned[:-1].rstrip() if cleaned.endswith(";") else cleaned
    if ";" in without_trailing_semicolon:
        raise ValueError("只允许执行一条 SQL。")
    if not re.match(r"^(select|with)\b", without_trailing_semicolon, re.IGNORECASE):
        raise ValueError("只允许 SELECT 或 WITH ... SELECT 查询。")
    if FORBIDDEN_SQL.search(without_trailing_semicolon):
        raise ValueError("SQL 包含禁止的写入或管理操作。")
    return without_trailing_semicolon


def _estimated_cardinality(plan_node: dict) -> int | None:
    """根据 DuckDB JSON 执行计划，估计当前节点产生的中间行数。"""

    operator_name = str(plan_node.get("name", "")).upper()
    if operator_name == "UNGROUPED_AGGREGATE":
        return 1

    extra_info = plan_node.get("extra_info") or {}
    raw_cardinality = extra_info.get("Estimated Cardinality")
    if raw_cardinality is not None:
        try:
            return int(str(raw_cardinality).replace(",", ""))
        except ValueError:
            pass

    child_estimates = [
        estimate
        for child in plan_node.get("children") or []
        if (estimate := _estimated_cardinality(child)) is not None
    ]
    if not child_estimates:
        return None

    if operator_name == "CROSS_PRODUCT":
        return prod(child_estimates)
    if operator_name == "UNION":
        return sum(child_estimates)
    return max(child_estimates)


def _walk_plan(plan_node: dict):
    yield plan_node
    for child in plan_node.get("children") or []:
        yield from _walk_plan(child)


def _load_query_plan(connection: duckdb.DuckDBPyConnection, sql: str) -> list[dict]:
    explain_row = connection.execute(f"EXPLAIN (FORMAT JSON) {sql}").fetchone()
    if not explain_row or len(explain_row) < 2:
        raise ValueError("DuckDB 未返回可检查的执行计划。")
    return json.loads(explain_row[1])


def _validate_query_plan_budget(plan_roots: list[dict]) -> None:
    plan_nodes = [node for root in plan_roots for node in _walk_plan(root)]
    if len(plan_nodes) > SQL_MAX_PLAN_NODES:
        raise SqlResourceLimitError(
            f"查询执行计划包含 {len(plan_nodes)} 个节点，超过上限 "
            f"{SQL_MAX_PLAN_NODES}。"
        )

    for node in plan_nodes:
        estimated_rows = _estimated_cardinality(node)
        if estimated_rows is None:
            continue
        if estimated_rows > SQL_MAX_ESTIMATED_INTERMEDIATE_ROWS:
            operator_name = str(node.get("name", "UNKNOWN")).upper()
            raise SqlResourceLimitError(
                f"执行计划中的 {operator_name} 预计产生 {estimated_rows:,} 行中间结果，"
                f"超过上限 {SQL_MAX_ESTIMATED_INTERMEDIATE_ROWS:,}。"
            )


def validate_sql_resource_budget(sql: str) -> str:
    """编译 SQL 并检查执行计划规模，但不真正执行查询。"""

    readonly_sql = validate_readonly_sql(sql)
    with duckdb.connect(
        str(_database_path()),
        read_only=True,
        config=DUCKDB_SAFE_CONNECTION_CONFIG,
    ) as connection:
        plan_roots = _load_query_plan(connection, readonly_sql)
    _validate_query_plan_budget(plan_roots)
    return readonly_sql


def validate_sql_against_database(sql: str) -> str:
    """让 DuckDB 编译 SQL并检查资源预算，但不真正执行查询。"""

    return validate_sql_resource_budget(sql)


def _fetch_query_rows(
    connection: duckdb.DuckDBPyConnection,
    sql: str,
) -> tuple[list[str], list[tuple]]:
    cursor = connection.execute(sql)
    columns = [description[0] for description in cursor.description]
    return columns, cursor.fetchmany(101)


def _execute_query_with_timeout(sql: str) -> tuple[list[str], list[tuple]]:
    """在独立执行线程中查询；超时后由主线程中断 DuckDB。"""

    connection = duckdb.connect(
        str(_database_path()),
        read_only=True,
        config=DUCKDB_SAFE_CONNECTION_CONFIG,
    )
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="duckdb-query")
    future = executor.submit(_fetch_query_rows, connection, sql)

    try:
        return future.result(timeout=SQL_QUERY_TIMEOUT_SECONDS)
    except FutureTimeoutError as error:
        connection.interrupt()
        try:
            future.result(timeout=1)
        except Exception:
            pass
        raise SqlTimeoutError(
            f"查询执行超过 {SQL_QUERY_TIMEOUT_SECONDS:g} 秒，已被中断。"
        ) from error
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        if future.done():
            connection.close()
        else:
            future.add_done_callback(lambda _future: connection.close())


def execute_readonly_sql_query(sql: str) -> str:
    readonly_sql = validate_sql_resource_budget(sql)
    columns, raw_rows = _execute_query_with_timeout(readonly_sql)

    truncated = len(raw_rows) > 100
    visible_rows = raw_rows[:100]
    rows = [
        {
            column: _json_value(value)
            for column, value in zip(columns, row, strict=True)
        }
        for row in visible_rows
    ]
    return json.dumps(
        {
            "columns": columns,
            "rows": rows,
            "returned_rows": len(rows),
            "truncated": truncated,
        },
        ensure_ascii=False,
        indent=2,
    )
