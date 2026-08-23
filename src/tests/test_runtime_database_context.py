import json

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from prompts.prompt_loader import build_model_input
from prompts.runtime_database_context import (
    build_runtime_database_environment,
    build_runtime_database_system_message,
    inject_runtime_database_context,
)


def test_postgresql_environment_declares_dialect_database_and_schema():
    runtime_environment = build_runtime_database_environment(
        {
            "DATA_AGENT_DATABASE_BACKEND": "postgresql",
            "DATA_AGENT_POSTGRES_DATABASE": "analytics",
        }
    )

    assert runtime_environment == {
        "database_engine": "PostgreSQL",
        "sql_dialect": "PostgreSQL",
        "database": "analytics",
        "default_schema": "public",
        "default_schema_basis": "application default; not database-probed",
    }


def test_postgresql_schema_can_be_overridden_by_runtime_configuration():
    runtime_environment = build_runtime_database_environment(
        {
            "DATA_AGENT_DATABASE_BACKEND": "postgresql",
            "DATA_AGENT_POSTGRES_DATABASE": "analytics",
            "DATA_AGENT_POSTGRES_SCHEMA": "reporting",
        }
    )

    assert runtime_environment["default_schema"] == "reporting"


def test_mysql_uses_the_current_database_as_its_default_schema():
    runtime_environment = build_runtime_database_environment(
        {
            "DATA_AGENT_DATABASE_BACKEND": "mysql",
            "DATA_AGENT_MYSQL_DATABASE": "warehouse",
        }
    )

    assert runtime_environment == {
        "database_engine": "MySQL",
        "sql_dialect": "MySQL",
        "database": "warehouse",
        "default_schema": "warehouse",
        "default_schema_basis": "current database",
    }


def test_duckdb_exposes_only_the_database_filename():
    runtime_environment = build_runtime_database_environment(
        {
            "DATA_AGENT_DATABASE_BACKEND": "duckdb",
            "DATA_AGENT_DATABASE_PATH": "/srv/data/analytics.duckdb",
        }
    )

    assert runtime_environment == {
        "database_engine": "DuckDB",
        "sql_dialect": "DuckDB",
        "database": "analytics.duckdb",
        "default_schema": "main",
        "default_schema_basis": "engine default",
    }


def test_database_system_message_excludes_credentials_and_network_details():
    message = build_runtime_database_system_message(
        {
            "DATA_AGENT_DATABASE_BACKEND": "postgresql",
            "DATA_AGENT_POSTGRES_DATABASE": "analytics",
            "DATA_AGENT_POSTGRES_HOST": "secret.internal",
            "DATA_AGENT_POSTGRES_USER": "private-user",
            "DATA_AGENT_POSTGRES_PASSWORD": "private-password",
        }
    )

    assert "analytics" in message.content
    assert "secret.internal" not in message.content
    assert "private-user" not in message.content
    assert "private-password" not in message.content


def test_database_context_is_inserted_before_knowledge_context():
    model_input = build_model_input(
        [HumanMessage(content="Count the relevant records.")],
        runtime_directory={"path": "/", "entries": []},
        runtime_navigation_graph="GLOBAL KNOWLEDGE GRAPH",
    )

    enriched_input = inject_runtime_database_context(
        model_input,
        {
            "DATA_AGENT_DATABASE_BACKEND": "postgresql",
            "DATA_AGENT_POSTGRES_DATABASE": "analytics",
        },
    )

    assert isinstance(enriched_input[0], SystemMessage)
    assert "[Runtime database environment]" in enriched_input[1].content
    assert "[Runtime knowledge directory]" in enriched_input[2].content
    payload = json.loads(enriched_input[1].content.split("\n\n", 1)[1])
    assert payload["sql_dialect"] == "PostgreSQL"
    assert payload["default_schema_basis"].endswith("not database-probed")


def test_database_context_rejects_unsupported_backends():
    with pytest.raises(RuntimeError, match="Unsupported database backend"):
        build_runtime_database_environment(
            {"DATA_AGENT_DATABASE_BACKEND": "sqlite"}
        )
