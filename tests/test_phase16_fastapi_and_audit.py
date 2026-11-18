"""Unit tests for Phase 16: Security Audit Logging and FastAPI REST API."""

import os
import sys
import unittest
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.database import db
from utils.auth import AuthManager
from utils.audit import (
    AuditLogger, ACTION_AUTH_LOGIN, ACTION_DOC_UPLOAD, ACTION_RAG_QUERY
)
from api.main import app
from api.schemas import LoginRequest, ChatRequest, SearchRequest


class TestPhase16AuditAndAPI(unittest.TestCase):
    """Test suite for Audit Logging, sanitization, CSV export, and REST API components."""

    def test_01_audit_logging_and_sanitization(self):
        """Verify audit logger redacts passwords, tokens, and secret keys."""
        dirty_details = {
            "password": "MySecretPassword123!",
            "access_token": "eyJhbGciOiJIUzI1Ni...",
            "api_key": "sk-proj-xyz123",
            "file_name": "clean_report.pdf",
            "status_code": 200
        }

        log_id = AuditLogger.log(
            action=ACTION_AUTH_LOGIN,
            user_id=1,
            username="test_admin_rb",
            resource_type="auth",
            resource_id="session_1",
            status="SUCCESS",
            details=dirty_details
        )
        self.assertIsNotNone(log_id)

        # Retrieve log from database and inspect details
        logs = AuditLogger.get_logs(action="AUTH_LOGIN", limit=5)
        self.assertTrue(len(logs) > 0)
        target_log = next((l for l in logs if l["id"] == log_id), None)
        self.assertIsNotNone(target_log)

        details_parsed = json.loads(target_log["details_json"])
        self.assertEqual(details_parsed["password"], "[REDACTED]")
        self.assertEqual(details_parsed["access_token"], "[REDACTED]")
        self.assertEqual(details_parsed["api_key"], "[REDACTED]")
        self.assertEqual(details_parsed["file_name"], "clean_report.pdf")

    def test_02_audit_csv_export(self):
        """Verify audit logs export to valid CSV format."""
        logs = AuditLogger.get_logs(limit=10)
        csv_str = AuditLogger.export_to_csv(logs)
        self.assertIn("id,timestamp,user_id,username,action", csv_str)
        self.assertIn("AUTH_LOGIN", csv_str)

    def test_03_jwt_token_flow(self):
        """Verify JWT generation, claims, and verification."""
        token = AuthManager.generate_token(user_id=42, username="alice", role="editor")
        claims = AuthManager.verify_token(token)
        self.assertEqual(claims["user_id"], 42)
        self.assertEqual(claims["username"], "alice")
        self.assertEqual(claims["role"], "editor")

    def test_04_fastapi_app_configuration(self):
        """Verify FastAPI routes and schema declarations."""
        openapi_schema = app.openapi()
        paths = openapi_schema.get("paths", {}).keys()
        self.assertIn("/health", paths)
        self.assertIn("/api/v1/auth/login", paths)
        self.assertIn("/api/v1/auth/register", paths)
        self.assertIn("/api/v1/users", paths)
        self.assertIn("/api/v1/documents", paths)
        self.assertIn("/api/v1/documents/upload", paths)
        self.assertIn("/api/v1/search", paths)
        self.assertIn("/api/v1/chat", paths)
        self.assertIn("/api/v1/audit-logs", paths)

    def test_05_schema_validations(self):
        """Verify Pydantic request schemas validation rules."""
        # Valid login request
        req = LoginRequest(username="admin", password="password123")
        self.assertEqual(req.username, "admin")

        # Valid search request
        s_req = SearchRequest(query="financial summary", top_k=5, search_mode="hybrid")
        self.assertEqual(s_req.top_k, 5)


if __name__ == "__main__":
    unittest.main()
