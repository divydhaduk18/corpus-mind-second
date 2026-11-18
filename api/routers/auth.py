"""Authentication & Session REST Endpoints (Improvement 32)."""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from typing import Dict, Any

import config
from utils.auth import AuthManager
from utils.database import db
from utils.audit import AuditLogger, ACTION_AUTH_LOGIN, ACTION_AUTH_LOGOUT, ACTION_AUTH_REGISTER
from api.schemas import LoginRequest, RegisterRequest, TokenResponse, UserResponse, APIResponse
from api.dependencies import get_current_user, check_rate_limit

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])


@router.post("/register", response_model=APIResponse, dependencies=[Depends(check_rate_limit)])
async def register(req: RegisterRequest, request: Request):
    """Register a new user account with role validation."""
    client_ip = request.client.host if request.client else None
    success, msg = AuthManager.register_user(
        username=req.username,
        password=req.password,
        role=req.role or "editor",
        ip_address=client_ip
    )
    if not success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)
    return APIResponse(success=True, message=msg)


@router.post("/login", response_model=TokenResponse, dependencies=[Depends(check_rate_limit)])
async def login(req: LoginRequest, request: Request):
    """Authenticate credentials and generate a signed JWT bearer access token."""
    client_ip = request.client.host if request.client else None
    user = db.get_user_by_username(req.username)
    if not user:
        AuditLogger.log(
            action=ACTION_AUTH_LOGIN,
            username=req.username,
            status="FAILURE",
            ip_address=client_ip,
            details={"reason": "User not found"}
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password."
        )

    if user.get("status") == "deactivated":
        AuditLogger.log(
            action=ACTION_AUTH_LOGIN,
            user_id=user["id"],
            username=req.username,
            status="DENIED",
            ip_address=client_ip,
            details={"reason": "Account deactivated"}
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated. Contact an administrator."
        )

    if not AuthManager.verify_password(req.password, user["password_hash"], user_id=user["id"]):
        AuditLogger.log(
            action=ACTION_AUTH_LOGIN,
            user_id=user["id"],
            username=req.username,
            status="FAILURE",
            ip_address=client_ip,
            details={"reason": "Incorrect password"}
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password."
        )

    role = user.get("role", "editor")
    if role == "user":
        role = "editor"

    token = AuthManager.generate_token(user_id=user["id"], username=user["username"], role=role)

    AuditLogger.log(
        action=ACTION_AUTH_LOGIN,
        user_id=user["id"],
        username=user["username"],
        status="SUCCESS",
        ip_address=client_ip,
        details={"role": role}
    )

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in_minutes=getattr(config, "JWT_ACCESS_TOKEN_EXPIRE_MINUTES", 1440),
        user_id=user["id"],
        username=user["username"],
        role=role
    )


@router.post("/logout", response_model=APIResponse)
async def logout(request: Request, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Logs out and records audit event."""
    client_ip = request.client.host if request.client else None
    AuditLogger.log(
        action=ACTION_AUTH_LOGOUT,
        user_id=current_user["id"],
        username=current_user["username"],
        status="SUCCESS",
        ip_address=client_ip
    )
    return APIResponse(success=True, message="Successfully logged out.")


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Returns the profile and permissions of the currently authenticated user."""
    return UserResponse(
        id=current_user["id"],
        username=current_user["username"],
        role=current_user.get("role", "editor"),
        status=current_user.get("status", "active"),
        created_at=str(current_user.get("created_at", "")),
        updated_at=str(current_user.get("updated_at", ""))
    )
