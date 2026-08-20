"""Read-only MySQL execution adapter for the configured database."""

import json
import os
import re
from datetime import date, datetime, time, timedelta
from decimal import Decimal

import pymysql

from config.project_paths import (
    MYSQL_DATABASE,
    MYSQL_HOST,
    MYSQL_PASSWORD,
    MYSQL_PORT,
    MYSQL_USER,
)


FORBIDDEN_SQL = re.compile(
    r"\b(insert|update|delete|replace|merge|create|drop|alter|truncate|"
    r"rename|load|grant|revoke|call|do|handler|set|reset|prepare|execute|"
    r"deallocate|lock|unlock|start|begin|commit|rollback|savepoint|release|"
    r"into|outfile|dumpfile)\b",
    re.IGNORECASE,
)
EXTERNAL_OR_EXPENSIVE_ACCESS = re.compile(
    r"\b(load_file|sleep|benchmark|get_lock|release_lock|is_free_lock|"
    r"is_used_lock)\s*\(",
    re.IGNORECASE,
)
SYSTEM_SCHEMA = re.compile(
    r"\b(information_schema|performance_schema|mysql|sys)\s*\.",
    re.IGNORECASE,
)
SQL_COMMENT = re.compile(r"(--(?:\s|$)|#|/\*)")
TABLE_REFERENCE = re.compile(
    r"\b(?:from|join)\s+"
    r"((?:`[^`]+`|[a-z_][a-z0-9_$]*)"
    r"(?:\s*\.\s*(?:`[^`]+`|[a-z_][a-z0-9_$]*))?)",
    re.IGNORECASE,
)

SQL_QUERY_TIMEOUT_SECONDS = float(os.getenv("SQL_QUERY_TIMEOUT_SECONDS", "5"))
SQL_DISPLAY_ROW_LIMIT = int(os.getenv("SQL_DISPLAY_ROW_LIMIT", "100"))


class SqlResourceLimitError(ValueError):
    """The query exceeded the configured MySQL execution budget."""


class SqlTimeoutError(SqlResourceLimitError):
    """The query exceeded the configured MySQL wall-clock limit."""


def validate_readonly_sql(sql: str) -> str:
    """Accept exactly one SELECT/CTE query and reject side effects."""

    cleaned = sql.strip()
    if not cleaned:
        raise ValueError("SQL must not be empty.")
    statement = cleaned[:-1].rstrip() if cleaned.endswith(";") else cleaned
    if ";" in statement:
        raise ValueError("Only one SQL statement is allowed.")
    if SQL_COMMENT.search(statement):
        raise ValueError("SQL comments are not allowed.")
    if not re.match(r"^(select|with)\b", statement, re.IGNORECASE):
        raise ValueError("Only SELECT or WITH ... SELECT is allowed.")
    if FORBIDDEN_SQL.search(statement):
        raise ValueError("SQL contains a prohibited write or administrative operation.")
    if EXTERNAL_OR_EXPENSIVE_ACCESS.search(statement):
        raise ValueError("SQL contains prohibited external or resource-abuse access.")
    if SYSTEM_SCHEMA.search(statement):
        raise ValueError("MySQL system schemas are outside the approved data scope.")
    if re.search(r"\bfor\s+update\b|\block\s+in\s+share\s+mode\b", statement, re.I):
        raise ValueError("Locking reads are not allowed.")
    if "@" in statement:
        raise ValueError("MySQL user and system variables are not allowed.")
    for table_reference in TABLE_REFERENCE.findall(statement):
        parts = [
            part.strip().strip("`").lower()
            for part in re.split(r"\s*\.\s*", table_reference)
        ]
        if len(parts) == 2 and parts[0] != MYSQL_DATABASE.lower():
            raise ValueError(
                "SQL references a database outside the configured data scope."
            )
    return statement


def _json_value(value):
    if isinstance(value, (Decimal, date, datetime, time, timedelta)):
        return str(value)
    if isinstance(value, bytes):
        return value.hex()
    return value


def _connect():
    if not MYSQL_HOST or not MYSQL_PORT or not MYSQL_USER or not MYSQL_DATABASE:
        raise RuntimeError(
            "MySQL host, port, user, and database are required."
        )
    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        charset="utf8mb4",
        connect_timeout=5,
        read_timeout=int(SQL_QUERY_TIMEOUT_SECONDS) + 2,
        write_timeout=5,
        autocommit=False,
    )


def execute_readonly_sql_query(sql: str) -> str:
    """Execute a validated query in a read-only transaction."""

    statement = validate_readonly_sql(sql)
    try:
        connection = _connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute("SET TRANSACTION READ ONLY")
                cursor.execute(
                    "SET SESSION MAX_EXECUTION_TIME = %s",
                    (int(SQL_QUERY_TIMEOUT_SECONDS * 1000),),
                )
                cursor.execute(statement)
                columns = [column[0] for column in cursor.description]
                raw_rows = cursor.fetchmany(SQL_DISPLAY_ROW_LIMIT + 1)
        finally:
            connection.rollback()
            connection.close()
    except pymysql.err.OperationalError as error:
        error_code = error.args[0] if error.args else None
        if error_code in {2013, 3024}:
            raise SqlTimeoutError(
                f"Query execution exceeded {SQL_QUERY_TIMEOUT_SECONDS:g} seconds."
            ) from error
        raise

    truncated = len(raw_rows) > SQL_DISPLAY_ROW_LIMIT
    rows = [
        {
            column: _json_value(value)
            for column, value in zip(columns, row, strict=True)
        }
        for row in raw_rows[:SQL_DISPLAY_ROW_LIMIT]
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
