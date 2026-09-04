"""Agent-facing wrapper around the configured read-only SQL executor."""

from importlib import import_module
import json
import os
from typing import Annotated

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from agent_runtime.agent_run_context import read_agent_run_context

ALLOWED_DATABASE_BACKENDS = {"postgresql", "mysql", "duckdb"}


def _configured_executor():
    backend = os.getenv("DATA_AGENT_DATABASE_BACKEND", "postgresql").strip().lower()
    if backend not in ALLOWED_DATABASE_BACKENDS:
        raise RuntimeError(f"Unsupported database backend: {backend!r}")
    module = import_module(f"databases.{backend}_executor")
    return (
        module.execute_readonly_sql_query,
        module.SqlResourceLimitError,
        module.SqlTimeoutError,
    )


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
    config: RunnableConfig,
) -> str:
    """Execute one read-only query against the currently configured database.

    The configured adapter enforces its own read-only validation, execution
    timeout, resource limits, and display-row limit. A successful return is
    required before query results may be presented to the user.
    """

    run_context = read_agent_run_context(config)
    if run_context is not None:
        selected_ids = run_context.selected_data_source_ids
        allowed_ids = run_context.allowed_data_source_ids
        if not selected_ids:
            raise RuntimeError("No data source is selected for this Agent run.")
        if any(data_source_id not in allowed_ids for data_source_id in selected_ids):
            raise RuntimeError("This Agent run selected a forbidden data source.")

    executor, resource_limit_error, timeout_error = _configured_executor()
    try:
        return executor(sql)
    except timeout_error as error:
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
    except resource_limit_error as error:
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
