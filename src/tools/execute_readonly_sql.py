"""Agent-facing wrapper around the configured read-only SQL executor."""

import json
from typing import Annotated

from langchain_core.tools import tool

from config.project_paths import DATABASE_BACKEND

if DATABASE_BACKEND == "postgresql":
    from databases.postgresql_executor import (
        SqlResourceLimitError,
        SqlTimeoutError,
        execute_readonly_sql_query,
    )
elif DATABASE_BACKEND == "mysql":
    from databases.mysql_executor import (
        SqlResourceLimitError,
        SqlTimeoutError,
        execute_readonly_sql_query,
    )
elif DATABASE_BACKEND == "duckdb":
    from databases.duckdb_executor import (
        SqlResourceLimitError,
        SqlTimeoutError,
        execute_readonly_sql_query,
    )
else:
    raise RuntimeError(f"Unsupported database backend: {DATABASE_BACKEND!r}")


@tool("execute_readonly_sql")
def execute_readonly_sql(
    sql: Annotated[
        str,
        (
            "Exactly one read-only SELECT or WITH ... SELECT statement using "
            "only physical database objects verified through approved "
            "knowledge."
        ),
    ],
) -> str:
    """Execute one read-only query against the currently configured database.

    The configured adapter enforces its own read-only validation, execution
    timeout, resource limits, and display-row limit. A successful return is
    required before query results may be presented to the user.
    """

    try:
        return execute_readonly_sql_query(sql)
    except SqlTimeoutError as error:
        return json.dumps(
            {
                "status": "rejected",
                "error_type": "TIMEOUT",
                "retryable": False,
                "message": "The query exceeded the configured execution timeout.",
                "details": str(error),
            },
            ensure_ascii=False,
            indent=2,
        )
    except SqlResourceLimitError as error:
        return json.dumps(
            {
                "status": "rejected",
                "error_type": "RESOURCE_LIMIT_EXCEEDED",
                "retryable": False,
                "message": (
                    "The query was rejected or interrupted because it "
                    "exceeded the configured database resource budget."
                ),
                "details": str(error),
            },
            ensure_ascii=False,
            indent=2,
        )
