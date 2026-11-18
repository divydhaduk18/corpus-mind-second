"""Centralized Authentication & Role-Based Authorization Service for Corpus Mind.

Features:
- Enterprise password hashing with modern bcrypt (salt rounds >= 12)
- Zero-downtime automatic backward-compatible migration from legacy PBKDF2 hashes
- JWT (JSON Web Tokens) stateless token generation and verification
- Multi-tier Role-Based Access Control (Admin, Editor, Viewer)
- Defense against privilege escalation
- Deactivated user session rejection
- Integrated security audit logging
"""

import os
import sys
import secrets
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.database import db
from utils.audit import AuditLogger, ACTION_AUTH_LOGIN, ACTION_AUTH_FAILED, ACTION_AUTH_LOGOUT, ACTION_AUTH_REGISTER

logger = logging.getLogger(__name__)

# Try importing bcrypt, fallback to secure PBKDF2 if unavailable
try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False
    logger.warning("bcrypt module not installed. Falling back to PBKDF2.")

# Try importing pyjwt
try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    logger.warning("pyjwt module not installed.")


class AuthManager:
    """Authentication, JWT Token Lifecycle, and Role-Based Authorization Manager."""

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using modern bcrypt with automatic work factor."""
        if BCRYPT_AVAILABLE:
            salt = bcrypt.gensalt(rounds=12)
            hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
            return hashed.decode("utf-8")
        else:
            # Fallback to PBKDF2
            salt = os.urandom(16)
            iterations = getattr(config, "PASSWORD_SALT_ROUNDS", 100000)
            key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
            return f"{iterations}${salt.hex()}${key.hex()}"

    @classmethod
    def verify_password(cls, password: str, stored_hash: str, user_id: Optional[int] = None) -> bool:
        """Verify password against stored bcrypt or legacy PBKDF2 hash.
        
        If a legacy PBKDF2 hash matches, automatically migrates the user to bcrypt.
        """
        if not stored_hash or not password:
            return False

        try:
            # Case 1: Stored hash is bcrypt ($2b$, $2a$, $2y$)
            if stored_hash.startswith(("$2b$", "$2a$", "$2y$")):
                if BCRYPT_AVAILABLE:
                    return bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8"))
                return False

            # Case 2: Stored hash is legacy PBKDF2 (iterations$salt$key)
            parts = stored_hash.split("$")
            if len(parts) == 3:
                iterations = int(parts[0])
                salt = bytes.fromhex(parts[1])
                original_key = bytes.fromhex(parts[2])
                key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
                matches = secrets.compare_digest(key, original_key)

                # Automatic hash migration: upgrade to bcrypt upon successful authentication
                if matches and user_id and BCRYPT_AVAILABLE:
                    new_hash = cls.hash_password(password)
                    db.update_user_password(user_id, new_hash)
                    logger.info(f"User {user_id} password hash successfully migrated from PBKDF2 to bcrypt.")

                return matches

            return False
        except Exception as e:
            logger.error(f"Error during password verification: {e}")
            return False

    # ==========================================
    # JWT Token Operations
    # ==========================================

    @classmethod
    def generate_token(
        cls,
        user_id: int,
        username: str,
        role: str,
        expires_minutes: Optional[int] = None
    ) -> str:
        """Generates a signed JWT access token."""
        if not JWT_AVAILABLE:
            raise RuntimeError("JWT library not available. Please install pyjwt.")

        secret = getattr(config, "JWT_SECRET_KEY", "corpus-mind-jwt-secret-key-production-change-me-32chars")
        algorithm = getattr(config, "JWT_ALGORITHM", "HS256")
        expiry = datetime.utcnow() + timedelta(
            minutes=expires_minutes or getattr(config, "JWT_ACCESS_TOKEN_EXPIRE_MINUTES", 1440)
        )

        payload = {
            "sub": str(user_id),
            "user_id": user_id,
            "username": username,
            "role": role,
            "exp": expiry,
            "iat": datetime.utcnow()
        }
        return jwt.encode(payload, secret, algorithm=algorithm)

    @classmethod
    def verify_token(cls, token: str) -> Dict[str, Any]:
        """Decodes and validates a JWT token. Raises exceptions on expiration or signature errors."""
        if not JWT_AVAILABLE:
            raise RuntimeError("JWT library not available. Please install pyjwt.")

        secret = getattr(config, "JWT_SECRET_KEY", "corpus-mind-jwt-secret-key-production-change-me-32chars")
        algorithm = getattr(config, "JWT_ALGORITHM", "HS256")
        return jwt.decode(token, secret, algorithms=[algorithm])

    # ==========================================
    # Role-Based Authorization Checks (RBAC)
    # ==========================================

    @staticmethod
    def has_role(user_role: Optional[str], allowed_roles: List[str]) -> bool:
        """Check if user role matches one of allowed roles."""
        if not user_role:
            return False
        # Normalize legacy 'user' to 'editor'
        if user_role == 'user':
            user_role = config.ROLE_EDITOR
        return user_role in allowed_roles

    @classmethod
    def can_upload_document(cls, user_role: Optional[str]) -> bool:
        """Admins and Editors can upload documents. Viewers cannot."""
        return cls.has_role(user_role, [config.ROLE_ADMIN, config.ROLE_EDITOR])

    @classmethod
    def can_manage_users(cls, user_role: Optional[str]) -> bool:
        """Only Admins can manage users and role assignments."""
        return cls.has_role(user_role, [config.ROLE_ADMIN])

    @classmethod
    def can_view_audit_logs(cls, user_role: Optional[str]) -> bool:
        """Only Admins can view security audit logs."""
        return cls.has_role(user_role, [config.ROLE_ADMIN])

    @classmethod
    def can_manage_settings(cls, user_role: Optional[str]) -> bool:
        """Only Admins can modify system-wide configuration settings."""
        return cls.has_role(user_role, [config.ROLE_ADMIN])

    # ==========================================
    # Registration & Login Lifecycle
    # ==========================================

    @classmethod
    def register_user(
        cls,
        username: str,
        password: str,
        role: str = "editor",
        ip_address: Optional[str] = None
    ) -> Tuple[bool, str]:
        """Registers a new user account with role validation and password hashing."""
        username = (username or "").strip()
        if not username or not password:
            return False, "Username and password cannot be empty."

        # Privilege escalation prevention: users registering via public form cannot make themselves admin
        if role == getattr(config, "ROLE_ADMIN", "admin"):
            role = getattr(config, "ROLE_EDITOR", "editor")

        valid_roles = getattr(config, "VALID_ROLES", ["admin", "editor", "viewer"])
        if role not in valid_roles:
            role = getattr(config, "ROLE_EDITOR", "editor")

        # Check if user already exists
        existing_user = db.get_user_by_username(username)
        if existing_user:
            return False, f"Username '{username}' is already taken."

        pw_hash = cls.hash_password(password)
        user_id = db.create_user(username=username, password_hash=pw_hash, role=role, status="active")
        if user_id:
            AuditLogger.log(
                action=ACTION_AUTH_REGISTER,
                user_id=user_id,
                username=username,
                resource_type="user",
                resource_id=user_id,
                status="SUCCESS",
                ip_address=ip_address,
                details={"role": role}
            )
            return True, "Registration successful! You can now log in."
        
        return False, "An error occurred during registration. Please try again."

    @classmethod
    def login_user(cls, username: str, password: str, ip_address: Optional[str] = None) -> Tuple[bool, str]:
        """Authenticates user, verifies active status, creates JWT token and Streamlit session."""
        username = (username or "").strip()
        if not username or not password:
            return False, "Username and password are required."

        user = db.get_user_by_username(username)
        if not user:
            AuditLogger.log(
                action=ACTION_AUTH_FAILED,
                username=username,
                status="FAILURE",
                ip_address=ip_address,
                details={"reason": "Username not found"}
            )
            return False, "Invalid username or password."

        # Verify active account status
        if user.get("status") == "deactivated":
            AuditLogger.log(
                action=ACTION_AUTH_FAILED,
                user_id=user["id"],
                username=username,
                status="DENIED",
                ip_address=ip_address,
                details={"reason": "Account is deactivated"}
            )
            return False, "Your account has been deactivated. Please contact an administrator."

        # Verify password
        if cls.verify_password(password, user["password_hash"], user_id=user["id"]):
            user_role = user.get("role", "editor")
            if user_role == "user":
                user_role = "editor"

            token = cls.generate_token(user["id"], user["username"], user_role) if JWT_AVAILABLE else None

            # Store in Streamlit session state
            st.session_state.logged_in = True
            st.session_state.user_id = user["id"]
            st.session_state.username = user["username"]
            st.session_state.user_role = user_role
            st.session_state.token = token
            st.session_state.last_activity = datetime.now()

            AuditLogger.log(
                action=ACTION_AUTH_LOGIN,
                user_id=user["id"],
                username=username,
                status="SUCCESS",
                ip_address=ip_address,
                details={"role": user_role}
            )
            return True, "Login successful!"

        AuditLogger.log(
            action=ACTION_AUTH_FAILED,
            user_id=user["id"],
            username=username,
            status="FAILURE",
            ip_address=ip_address,
            details={"reason": "Incorrect password"}
        )
        return False, "Invalid username or password."

    @staticmethod
    def logout_user(ip_address: Optional[str] = None):
        """Clears user session from Streamlit and logs audit event."""
        if st.session_state.get("logged_in"):
            user_id = st.session_state.get("user_id")
            username = st.session_state.get("username")
            AuditLogger.log(
                action=ACTION_AUTH_LOGOUT,
                user_id=user_id,
                username=username,
                status="SUCCESS",
                ip_address=ip_address
            )

        st.session_state.logged_in = False
        st.session_state.user_id = None
        st.session_state.username = None
        st.session_state.user_role = None
        st.session_state.token = None
        st.session_state.last_activity = None
        st.session_state.chat_session_id = None

    @staticmethod
    def check_session_timeout():
        """Check if user session has timed out."""
        if st.session_state.get("logged_in") and st.session_state.get("last_activity"):
            last_act = st.session_state.get("last_activity")
            timeout_limit = timedelta(minutes=getattr(config, "SESSION_TIMEOUT_MINUTES", 30))
            if datetime.now() - last_act > timeout_limit:
                AuthManager.logout_user()
                st.warning("Session timed out. Please log in again.")
                st.rerun()
            else:
                st.session_state.last_activity = datetime.now()

    @staticmethod
    def is_authenticated() -> bool:
        """Helper to verify current page access."""
        AuthManager.check_session_timeout()
        return st.session_state.get("logged_in", False)

    @staticmethod
    def require_auth():
        """Ensure the user is authenticated; otherwise stop execution."""
        AuthManager.check_session_timeout()
        if not st.session_state.get("logged_in", False):
            st.warning("🔒 Access denied. Please log in first.")
            st.stop()

    @classmethod
    def require_role(cls, allowed_roles: List[str]):
        """Ensure the user has one of the allowed roles; otherwise stop execution."""
        cls.require_auth()
        role = st.session_state.get("user_role")
        if not cls.has_role(role, allowed_roles):
            st.error(f"⛔ Unauthorized: Access requires one of {allowed_roles} roles. Your role is '{role}'.")
            st.stop()

    @staticmethod
    def initialize_admin_account():
        """Ensure default admin account exists (admin / admin123)."""
        admin = db.get_user_by_username("admin")
        if not admin:
            pw_hash = AuthManager.hash_password("admin123")
            db.create_user("admin", pw_hash, "admin", "active")
            logger.info("Default admin user created.")
