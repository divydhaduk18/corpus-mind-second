"""FastAPI Authentication, RBAC, and Security Dependencies (Improvement 32)."""

import time
import logging
from typing import Optional, List, Dict, Any
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

import config
from utils.auth import AuthManager
from utils.database import db

logger = logging.getLogger(__name__)

security_scheme = HTTPBearer(auto_error=False)


# In-memory IP-based sliding window rate limiter
class RateLimiter:
    def __init__(self, requests_per_minute: int = 120):
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, List[float]] = {}

    def is_rate_limited(self, client_ip: str) -> bool:
        now = time.time()
        window_start = now - 60.0

        if client_ip not in self.requests:
            self.requests[client_ip] = [now]
            return False

        # Clean entries outside window
        valid_timestamps = [t for t in self.requests[client_ip] if t >= window_start]
        self.requests[client_ip] = valid_timestamps

        if len(valid_timestamps) >= self.requests_per_minute:
            return True

        self.requests[client_ip].append(now)
        return False


rate_limiter = RateLimiter(requests_per_minute=getattr(config, "RATE_LIMIT_PER_MINUTE", 120))


async def check_rate_limit(request: Request):
    """Enforces client IP rate limiting."""
    client_ip = request.client.host if request.client else "127.0.0.1"
    if rate_limiter.is_rate_limited(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please wait before retrying."
        )


async def get_current_user(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme)
) -> Dict[str, Any]:
    """Validates JWT bearer token and extracts authenticated user account."""
    if not auth or not auth.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authentication token. Header must be 'Authorization: Bearer <token>'",
            headers={"WWW-Authenticate": "Bearer"}
        )

    try:
        payload = AuthManager.verify_token(auth.credentials)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"}
        )

    user_id = payload.get("user_id")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Malformed token payload.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    user = db.get_user_by_id(int(user_id))
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account no longer exists."
        )

    if user.get("status") == "deactivated":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated. Contact an administrator."
        )

    return user


def require_role(allowed_roles: List[str]):
    """Dependency factory checking if current user has an authorized role."""
    async def role_checker(current_user: Dict[str, Any] = Depends(get_current_user)):
        user_role = current_user.get("role", "editor")
        if user_role == "user":
            user_role = "editor"
        if not AuthManager.has_role(user_role, allowed_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Forbidden: Action requires one of {allowed_roles} roles. Your role is '{user_role}'."
            )
        return current_user
    return role_checker


require_admin = require_role([config.ROLE_ADMIN])
require_editor_or_admin = require_role([config.ROLE_ADMIN, config.ROLE_EDITOR])


def require_document_access(required_level: str = "viewer"):
    """Validates that current user has at least the required permission on the document."""
    async def permission_checker(
        document_id: int,
        current_user: Dict[str, Any] = Depends(get_current_user)
    ):
        has_access = db.can_user_access_document(
            user_id=current_user["id"],
            document_id=document_id,
            required_level=required_level
        )
        if not has_access:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied: You do not have '{required_level}' access to document #{document_id}."
            )
        doc = db.get_document_by_id(document_id)
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document #{document_id} not found."
            )
        return doc
    return permission_checker
