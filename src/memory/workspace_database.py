"""Workspace users, memberships, and local development identities."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from config.project_paths import CHAT_HISTORY_DATABASE_PATH


DEFAULT_DEV_USER = "admin-a"

# Development identities only choose who is making the request. Roles are read
# back from workspace_members so the browser cannot assign its own role.
DEV_ACCOUNT_SEEDS = (
    {
        "login_id": "admin-a",
        "user_id": "user-admin-a",
        "display_name": "XinCo",
        "avatar": "X",
        "workspace_id": "workspace-a",
        "workspace_name": "XinCo 工作空间",
        "role": "admin",
        "resources_ready": True,
    },
    {
        "login_id": "analyst-b",
        "user_id": "user-analyst-b",
        "display_name": "分析员 B",
        "avatar": "B",
        "workspace_id": "workspace-b",
        "workspace_name": "分析演练空间 B",
        "role": "analyst",
        "resources_ready": False,
    },
    {
        "login_id": "viewer-c",
        "user_id": "user-viewer-c",
        "display_name": "观察员 C",
        "avatar": "C",
        "workspace_id": "workspace-c",
        "workspace_name": "只读演练空间 C",
        "role": "viewer",
        "resources_ready": False,
    },
)


def connect_workspace_database(
    database_path: Path | None = None,
) -> sqlite3.Connection:
    connection = sqlite3.connect(database_path or CHAT_HISTORY_DATABASE_PATH)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def create_workspace_tables(database_path: Path | None = None) -> None:
    """Create workspace tables and the three local simulation accounts."""

    resolved_path = database_path or CHAT_HISTORY_DATABASE_PATH
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    connection = connect_workspace_database(resolved_path)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS workspaces (
            workspace_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            resources_ready INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS workspace_members (
            workspace_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('admin', 'analyst', 'viewer')),
            PRIMARY KEY (workspace_id, user_id),
            FOREIGN KEY (workspace_id) REFERENCES workspaces(workspace_id),
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
        """
    )

    created_at = datetime.now(timezone.utc).isoformat()
    for account in DEV_ACCOUNT_SEEDS:
        connection.execute(
            """
            INSERT OR IGNORE INTO users(user_id, display_name, created_at)
            VALUES (?, ?, ?)
            """,
            (account["user_id"], account["display_name"], created_at),
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO workspaces(
                workspace_id,
                name,
                resources_ready,
                created_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                account["workspace_id"],
                account["workspace_name"],
                int(account["resources_ready"]),
                created_at,
            ),
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO workspace_members(workspace_id, user_id, role)
            VALUES (?, ?, ?)
            """,
            (account["workspace_id"], account["user_id"], account["role"]),
        )

    connection.commit()
    connection.close()


def read_dev_account(
    login_id: str,
    database_path: Path | None = None,
) -> dict | None:
    account_seed = next(
        (account for account in DEV_ACCOUNT_SEEDS if account["login_id"] == login_id),
        None,
    )
    if account_seed is None:
        return None

    connection = connect_workspace_database(database_path)
    account_row = connection.execute(
        """
        SELECT
            users.user_id,
            users.display_name,
            workspaces.workspace_id,
            workspaces.name,
            workspaces.resources_ready,
            workspace_members.role
        FROM workspace_members
        JOIN users ON users.user_id = workspace_members.user_id
        JOIN workspaces ON workspaces.workspace_id = workspace_members.workspace_id
        WHERE users.user_id = ? AND workspaces.workspace_id = ?
        """,
        (account_seed["user_id"], account_seed["workspace_id"]),
    ).fetchone()
    connection.close()
    if account_row is None:
        return None

    return {
        "login_id": login_id,
        "user_id": account_row[0],
        "display_name": account_row[1],
        "avatar": account_seed["avatar"],
        "workspace_id": account_row[2],
        "workspace_name": account_row[3],
        "resources_ready": bool(account_row[4]),
        "role": account_row[5],
    }


def list_dev_accounts(database_path: Path | None = None) -> list[dict]:
    accounts = []
    for account_seed in DEV_ACCOUNT_SEEDS:
        account = read_dev_account(account_seed["login_id"], database_path)
        if account is not None:
            accounts.append(account)
    return accounts


def user_is_workspace_member(
    user_id: str,
    workspace_id: str,
    database_path: Path | None = None,
) -> bool:
    connection = connect_workspace_database(database_path)
    membership = connection.execute(
        """
        SELECT 1
        FROM workspace_members
        WHERE user_id = ? AND workspace_id = ?
        """,
        (user_id, workspace_id),
    ).fetchone()
    connection.close()
    return membership is not None


def default_account() -> dict:
    return dict(DEV_ACCOUNT_SEEDS[0])
