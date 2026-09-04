"""Resolve the current local user and enforce workspace permissions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from typing import Annotated, Literal

from fastapi import Header, HTTPException, Request

from memory import workspace_database


Role = Literal["admin", "analyst", "viewer"]
DEMO_AUTH_ENABLED = os.getenv("DATA_AGENT_DEMO_AUTH", "1") == "1"

ROLE_LABELS: dict[Role, str] = {
    "admin": "管理员",
    "analyst": "分析员",
    "viewer": "只读用户",
}

ROLE_PERMISSIONS: dict[Role, frozenset[str]] = {
    "admin": frozenset(
        {
            "chat:run",
            "conversation:read",
            "conversation:write",
            "config:read",
            "config:write",
        }
    ),
    "analyst": frozenset(
        {
            "chat:run",
            "conversation:read",
            "conversation:write",
            "config:read",
        }
    ),
    "viewer": frozenset({"conversation:read", "config:read"}),
}


@dataclass(frozen=True)
class CurrentUser:
    login_id: str
    user_id: str
    display_name: str
    avatar: str
    workspace_id: str
    workspace_name: str
    resources_ready: bool
    role: Role

    @property
    def role_label(self) -> str:
        return ROLE_LABELS[self.role]

    @property
    def permissions(self) -> frozenset[str]:
        return ROLE_PERMISSIONS[self.role]


def resolve_current_user(login_id: str | None) -> CurrentUser:
    if not DEMO_AUTH_ENABLED:
        raise HTTPException(
            status_code=401,
            detail="模拟账号已关闭，请接入真实登录身份。",
        )
    selected_login = (login_id or workspace_database.DEFAULT_DEV_USER).strip()
    account = workspace_database.read_dev_account(selected_login)
    if account is None:
        raise HTTPException(status_code=401, detail="模拟账号不存在。")
    return CurrentUser(**account)


def read_current_user(
    x_dev_user: Annotated[str | None, Header(alias="X-Dev-User")] = None,
) -> CurrentUser:
    return resolve_current_user(x_dev_user)


def current_user_from_request(request: Request) -> CurrentUser:
    return resolve_current_user(request.headers.get("X-Dev-User"))


def require_permission(current_user: CurrentUser, permission: str) -> None:
    if permission not in current_user.permissions:
        raise HTTPException(status_code=403, detail="当前账号没有执行此操作的权限。")


def require_workspace_resources(current_user: CurrentUser) -> None:
    if not current_user.resources_ready:
        raise HTTPException(
            status_code=409,
            detail="当前模拟工作空间尚未配置独立数据库、模型和知识库。",
        )


def public_user(current_user: CurrentUser) -> dict:
    payload = asdict(current_user)
    payload["role_label"] = current_user.role_label
    payload["permissions"] = sorted(current_user.permissions)
    return payload


def list_public_dev_accounts() -> list[dict]:
    return [
        public_user(CurrentUser(**account))
        for account in workspace_database.list_dev_accounts()
    ]
