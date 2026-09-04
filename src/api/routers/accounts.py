"""Local account switcher endpoints used before real login is added."""

from typing import Annotated

from fastapi import APIRouter, Depends

from security.workspace_access import (
    DEMO_AUTH_ENABLED,
    CurrentUser,
    list_public_dev_accounts,
    public_user,
    read_current_user,
)


router = APIRouter(prefix="/api/accounts", tags=["accounts"])


@router.get("")
def list_accounts(
    current_user: Annotated[CurrentUser, Depends(read_current_user)],
):
    return {
        "demo_mode": DEMO_AUTH_ENABLED,
        "current": public_user(current_user),
        "accounts": list_public_dev_accounts(),
    }
