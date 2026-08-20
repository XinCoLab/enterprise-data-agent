"""Read-only PostgreSQL execution adapter for the configured database."""

import json
import os
import re
from datetime import date, datetime
from decimal import Decimal

import psycopg2
from psycopg2.errors import QueryCanceled

FORBIDDEN_SQL = re.compile(
    r"\b(insert|update|delete|merge|create|drop|alter|truncate|copy|grant|"
    r"revoke|comment|vacuum|analyze|refresh|reindex|cluster|call|do|set|"
    r"reset|listen|notify|unlisten|prepare|execute|deallocate|lock)\b",
    re.IGNORECASE,
)
EXTERNAL_ACCESS = re.compile(
    r"\b(pg_read_file|pg_read_binary_file|pg_ls_dir|pg_stat_file|"
    r"lo_import|lo_export|dblink|dblink_connect|postgres_fdw|"
    r"file_fdw|program)\b",
    re.IGNORECASE,
)

SQL_QUERY_TIMEOUT_SECONDS = 5


class SqlResourceLimitError(ValueError):
    """The query exceeded the configured PostgreSQL execution budget."""


class SqlTimeoutError(SqlResourceLimitError):
    """The query exceeded the configured PostgreSQL wall-clock limit."""


def validate_readonly_sql(sql: str) -> str:
    """Accept exactly one SELECT/CTE query and reject side effects."""

    cleaned = sql.strip()
    if not cleaned:
        raise ValueError("SQL must not be empty.")
    statement = cleaned[:-1].rstrip() if cleaned.endswith(";") else cleaned
    if ";" in statement:
        raise ValueError("Only one SQL statement is allowed.")
    if not re.match(r"^(select|with)\b", statement, re.IGNORECASE):
        raise ValueError("Only SELECT or WITH ... SELECT is allowed.")
    if FORBIDDEN_SQL.search(statement):
        raise ValueError("SQL contains a prohibited write or administrative operation.")
    if EXTERNAL_ACCESS.search(statement):
        raise ValueError("SQL contains prohibited external or server-file access.")
    return statement


def _json_value(value):
    if isinstance(value, (Decimal, date, datetime)):
        return str(value)
    return value


def _connect():
    database = os.getenv("DATA_AGENT_POSTGRES_DATABASE", "").strip()
    if not database:
        raise RuntimeError("DATA_AGENT_POSTGRES_DATABASE is required.")
    return psycopg2.connect(
        host=os.getenv("DATA_AGENT_POSTGRES_HOST", "127.0.0.1").strip(),
        port=int(os.getenv("DATA_AGENT_POSTGRES_PORT", "5432")),
        user=os.getenv("DATA_AGENT_POSTGRES_USER", "").strip(),
        password=os.getenv("DATA_AGENT_POSTGRES_PASSWORD", ""),
        dbname=database,
        connect_timeout=5,
        application_name="enterprise-data-agent",
    )


def execute_readonly_sql_query(sql: str) -> str:
    """Execute a validated query in a read-only transaction, returning 100 rows."""

    statement = validate_readonly_sql(sql)
    try:
        with _connect() as connection:
            connection.set_session(readonly=True)
            with connection.cursor() as cursor:
                cursor.execute(
                    "SET LOCAL statement_timeout = %s",
                    (int(SQL_QUERY_TIMEOUT_SECONDS * 1000),),
                )
                cursor.execute(statement)
                columns = [column.name for column in cursor.description]
                raw_rows = cursor.fetchmany(101)
    except QueryCanceled as error:
        raise SqlTimeoutError(
            f"Query execution exceeded {SQL_QUERY_TIMEOUT_SECONDS} seconds."
        ) from error

    truncated = len(raw_rows) > 100
    rows = [
        {
            column: _json_value(value)
            for column, value in zip(columns, row, strict=True)
        }
        for row in raw_rows[:100]
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
