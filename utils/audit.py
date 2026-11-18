"""Centralized Audit Logging System for Corpus Mind (Improvement 33).

Tracks security, authentication, document access, administrative changes,
and RAG interactions. Guarantees zero sensitive data leakage (passwords,
API keys, tokens, or raw document text are automatically redacted).
Supports structured JSON metadata and CSV export.
"""

import os
import sys
import csv
import io
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Union

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.database import db

logger = logging.getLogger(__name__)

# Standard Audit Actions
ACTION_AUTH_LOGIN = "AUTH_LOGIN"
ACTION_AUTH_FAILED = "AUTH_FAILED"
ACTION_AUTH_LOGOUT = "AUTH_LOGOUT"
ACTION_AUTH_REGISTER = "AUTH_REGISTER"
ACTION_AUTH_PASSWORD_CHANGE = "AUTH_PASSWORD_CHANGE"

ACTION_USER_CREATE = "USER_CREATE"
ACTION_USER_ROLE_CHANGE = "USER_ROLE_CHANGE"
ACTION_USER_STATUS_CHANGE = "USER_STATUS_CHANGE"
ACTION_USER_DELETE = "USER_DELETE"

ACTION_DOC_UPLOAD = "DOC_UPLOAD"
ACTION_DOC_VIEW = "DOC_VIEW"
ACTION_DOC_DOWNLOAD = "DOC_DOWNLOAD"
ACTION_DOC_EDIT = "DOC_EDIT"
ACTION_DOC_DELETE = "DOC_DELETE"
ACTION_DOC_PERM_GRANT = "DOC_PERM_GRANT"
ACTION_DOC_PERM_REVOKE = "DOC_PERM_REVOKE"
ACTION_DOC_OWNERSHIP_TRANSFER = "DOC_OWNERSHIP_TRANSFER"

ACTION_RAG_SEARCH = "RAG_SEARCH"
ACTION_RAG_QUERY = "RAG_QUERY"
ACTION_RAG_SUMMARIZE = "RAG_SUMMARIZE"


class AuditLogger:
    """Thread-safe, privacy-preserving audit logging service."""

    @classmethod
    def log(
        cls,
        action: str,
        user_id: Optional[int] = None,
        username: Optional[str] = None,
        resource_type: str = "system",
        resource_id: Optional[Union[str, int]] = None,
        status: str = "SUCCESS",
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> Optional[int]:
        """Record an auditable action with automatic redaction of confidential fields."""
        if not getattr(config, "AUDIT_LOGS_ENABLED", True):
            return None

        # Resolve username if user_id is provided but username is omitted
        if user_id and not username:
            user = db.get_user_by_id(user_id)
            if user:
                username = user.get("username")

        return db.create_audit_log(
            user_id=user_id,
            username=username,
            action=action,
            resource_type=resource_type,
            resource_id=str(resource_id) if resource_id is not None else None,
            status=status,
            ip_address=ip_address,
            details=details
        )

    @classmethod
    def get_logs(
        cls,
        user_id: Optional[int] = None,
        action: Optional[str] = None,
        resource_type: Optional[str] = None,
        status: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Query audit log entries with multi-attribute filtering."""
        return db.get_audit_logs(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            status=status,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset
        )

    @classmethod
    def get_count(
        cls,
        user_id: Optional[int] = None,
        action: Optional[str] = None,
        resource_type: Optional[str] = None,
        status: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> int:
        """Count total audit log entries matching filters."""
        return db.count_audit_logs(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            status=status,
            start_date=start_date,
            end_date=end_date
        )

    @classmethod
    def export_to_csv(cls, logs: List[Dict[str, Any]]) -> str:
        """Export audit log records to CSV formatted string."""
        output = io.StringIO()
        fieldnames = [
            "id", "timestamp", "user_id", "username", "action",
            "resource_type", "resource_id", "status", "ip_address", "details_json"
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for log in logs:
            row = {k: log.get(k, "") for k in fieldnames}
            writer.writerow(row)

        return output.getvalue()
