"""User Accounts & RBAC Administration REST Endpoints (Improvement 32)."""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from typing import List, Dict, Any

import config
from utils.database import db
from utils.audit import (
    AuditLogger, ACTION_USER_ROLE_CHANGE, ACTION_USER_STATUS_CHANGE, ACTION_USER_DELETE
)
from api.schemas import UserResponse, UserRoleUpdateRequest, UserStatusUpdateRequest, APIResponse
from api.dependencies import require_admin

router = APIRouter(prefix="/api/v1/users", tags=["User Administration (Admin Only)"])


@router.get("", response_model=List[UserResponse])
async def list_users(admin_user: Dict[str, Any] = Depends(require_admin)):
    """List all registered system users with their roles and statuses."""
    users = db.get_all_users()
    return [
        UserResponse(
            id=u["id"],
            username=u["username"],
            role=u.get("role", "editor"),
            status=u.get("status", "active"),
            created_at=str(u.get("created_at", "")),
            updated_at=str(u.get("updated_at", ""))
        )
        for u in users
    ]


@router.get("/{user_id}", response_model=UserResponse)
async def get_user_detail(user_id: int, admin_user: Dict[str, Any] = Depends(require_admin)):
    """Retrieve details for a specific user ID."""
    u = db.get_user_by_id(user_id)
    if not u:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"User #{user_id} not found.")
    return UserResponse(
        id=u["id"],
        username=u["username"],
        role=u.get("role", "editor"),
        status=u.get("status", "active"),
        created_at=str(u.get("created_at", "")),
        updated_at=str(u.get("updated_at", ""))
    )


@router.put("/{user_id}/role", response_model=APIResponse)
async def update_user_role(
    user_id: int,
    req: UserRoleUpdateRequest,
    request: Request,
    admin_user: Dict[str, Any] = Depends(require_admin)
):
    """Assign or change user authorization role (admin, editor, viewer)."""
    valid_roles = getattr(config, "VALID_ROLES", ["admin", "editor", "viewer"])
    if req.role not in valid_roles:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role '{req.role}'. Valid roles are: {valid_roles}"
        )

    target_user = db.get_user_by_id(user_id)
    if not target_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"User #{user_id} not found.")

    success = db.update_user_role(user_id, req.role)
    if not success:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update role.")

    client_ip = request.client.host if request.client else None
    AuditLogger.log(
        action=ACTION_USER_ROLE_CHANGE,
        user_id=admin_user["id"],
        username=admin_user["username"],
        resource_type="user",
        resource_id=user_id,
        status="SUCCESS",
        ip_address=client_ip,
        details={"target_username": target_user["username"], "new_role": req.role}
    )

    return APIResponse(success=True, message=f"User '{target_user['username']}' role updated to '{req.role}'.")


@router.put("/{user_id}/status", response_model=APIResponse)
async def update_user_status(
    user_id: int,
    req: UserStatusUpdateRequest,
    request: Request,
    admin_user: Dict[str, Any] = Depends(require_admin)
):
    """Activate or deactivate a user account."""
    if req.status not in ["active", "deactivated"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Status must be 'active' or 'deactivated'."
        )

    target_user = db.get_user_by_id(user_id)
    if not target_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"User #{user_id} not found.")

    if target_user["id"] == admin_user["id"] and req.status == "deactivated":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Administrators cannot deactivate their own active account."
        )

    success = db.update_user_status(user_id, req.status)
    if not success:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update status.")

    client_ip = request.client.host if request.client else None
    AuditLogger.log(
        action=ACTION_USER_STATUS_CHANGE,
        user_id=admin_user["id"],
        username=admin_user["username"],
        resource_type="user",
        resource_id=user_id,
        status="SUCCESS",
        ip_address=client_ip,
        details={"target_username": target_user["username"], "new_status": req.status}
    )

    return APIResponse(success=True, message=f"User '{target_user['username']}' status updated to '{req.status}'.")


@router.delete("/{user_id}", response_model=APIResponse)
async def delete_user(
    user_id: int,
    request: Request,
    admin_user: Dict[str, Any] = Depends(require_admin)
):
    """Permanently delete a user account."""
    if user_id == admin_user["id"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Administrators cannot delete their own account."
        )

    target_user = db.get_user_by_id(user_id)
    if not target_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"User #{user_id} not found.")

    success = db.delete_user(user_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete user.")

    client_ip = request.client.host if request.client else None
    AuditLogger.log(
        action=ACTION_USER_DELETE,
        user_id=admin_user["id"],
        username=admin_user["username"],
        resource_type="user",
        resource_id=user_id,
        status="SUCCESS",
        ip_address=client_ip,
        details={"deleted_username": target_user["username"]}
    )

    return APIResponse(success=True, message=f"User '{target_user['username']}' deleted permanently.")
