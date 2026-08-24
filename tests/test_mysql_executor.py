import json
from unittest.mock import patch

import pytest

from databases.mysql_executor import (
    execute_readonly_sql_query,
    validate_readonly_sql,
)


def test_mysql_accepts_one_select_or_cte():
    assert validate_readonly_sql("SELECT 1;") == "SELECT 1"
    assert (
        validate_readonly_sql("WITH x AS (SELECT 1 AS n) SELECT n FROM x")
        == "WITH x AS (SELECT 1 AS n) SELECT n FROM x"
    )


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM transfer_cmd_his",
        "SELECT 1; SELECT 2",
        "SELECT LOAD_FILE('/etc/passwd')",
        "SELECT SLEEP(10)",
        "SELECT * FROM information_schema.tables",
        "SELECT * FROM another_database.transfer_cmd_his",
        "SELECT @x := 1",
        "SELECT * FROM transfer_cmd_his FOR UPDATE",
        "SELECT 1 -- hidden text",
    ],
)
def test_mysql_rejects_unsafe_queries(sql):
    with pytest.raises(ValueError):
        validate_readonly_sql(sql)


class FakeCursor:
    description = (("total",),)

    def __init__(self):
        self.statements = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, statement, parameters=None):
        self.statements.append((statement, parameters))

    def fetchmany(self, count):
        return [(3,)]


class FakeConnection:
    def __init__(self):
        self.cursor_instance = FakeCursor()
        self.rolled_back = False
        self.closed = False

    def cursor(self):
        return self.cursor_instance

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


@patch("databases.mysql_executor._connect")
def test_mysql_executes_readonly_and_returns_standard_contract(connect):
    connection = FakeConnection()
    connect.return_value = connection

    payload = json.loads(execute_readonly_sql_query("SELECT COUNT(*) AS total"))

    assert payload == {
        "columns": ["total"],
        "rows": [{"total": 3}],
        "returned_rows": 1,
        "truncated": False,
    }
    assert connection.cursor_instance.statements[0][0] == "SET TRANSACTION READ ONLY"
    assert connection.rolled_back
    assert connection.closed
