"""Unit tests for Phase 15: Role-Based Access Control (RBAC) and Document Permissions."""

import os
import sys
import unittest
import tempfile
import sqlite3

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.database import db
from utils.auth import AuthManager
from utils.rag import RAGManager


class TestPhase15RBACAndDocumentPermissions(unittest.TestCase):
    """Test suite for RBAC roles, password hashing, and fine-grained document access control."""

    @classmethod
    def setUpClass(cls):
        # Create test users: 1 Admin, 1 Editor, 1 Viewer
        cls.admin_user = "test_admin_rb"
        cls.editor_user = "test_editor_rb"
        cls.viewer_user = "test_viewer_rb"

        # Register users
        AuthManager.register_user(cls.admin_user, "AdminPass123!", role="admin")
        # Ensure role is admin
        u_admin = db.get_user_by_username(cls.admin_user)
        db.update_user_role(u_admin["id"], "admin")
        cls.admin_id = u_admin["id"]

        AuthManager.register_user(cls.editor_user, "EditorPass123!", role="editor")
        u_editor = db.get_user_by_username(cls.editor_user)
        cls.editor_id = u_editor["id"]

        AuthManager.register_user(cls.viewer_user, "ViewerPass123!", role="viewer")
        u_viewer = db.get_user_by_username(cls.viewer_user)
        db.update_user_role(u_viewer["id"], "viewer")
        cls.viewer_id = u_viewer["id"]

    def test_01_password_hashing_and_verification(self):
        """Verify modern bcrypt hashing, verification, and failure on bad password."""
        raw_pw = "SecretP@ssword99"
        pw_hash = AuthManager.hash_password(raw_pw)
        self.assertTrue(pw_hash.startswith(("$2b$", "$2a$", "$2y$")))
        self.assertTrue(AuthManager.verify_password(raw_pw, pw_hash))
        self.assertFalse(AuthManager.verify_password("WrongPassword", pw_hash))

    def test_02_rbac_permissions(self):
        """Verify RBAC role checks for Admin, Editor, and Viewer."""
        self.assertTrue(AuthManager.can_manage_users("admin"))
        self.assertFalse(AuthManager.can_manage_users("editor"))
        self.assertFalse(AuthManager.can_manage_users("viewer"))

        self.assertTrue(AuthManager.can_upload_document("admin"))
        self.assertTrue(AuthManager.can_upload_document("editor"))
        self.assertFalse(AuthManager.can_upload_document("viewer"))

    def test_03_document_permissions_lifecycle(self):
        """Verify document permission grant, query, access check, and revocation."""
        # Editor creates a document
        doc_id = db.add_document(
            user_id=self.editor_id,
            file_name="quarterly_confidential.pdf",
            file_size=1024,
            file_path="/tmp/quarterly_confidential.pdf",
            category="Financial"
        )
        self.assertIsNotNone(doc_id)

        # Editor is owner
        self.assertEqual(db.get_user_permission_for_document(self.editor_id, doc_id), "owner")
        # Admin has owner access
        self.assertEqual(db.get_user_permission_for_document(self.admin_id, doc_id), "owner")
        # Viewer has no initial access
        self.assertIsNone(db.get_user_permission_for_document(self.viewer_id, doc_id))
        self.assertFalse(db.can_user_access_document(self.viewer_id, doc_id, required_level="viewer"))

        # Grant Viewer access to viewer user
        granted = db.grant_document_permission(
            document_id=doc_id,
            user_id=self.viewer_id,
            permission_level="viewer",
            granted_by=self.editor_id
        )
        self.assertTrue(granted)

        # Viewer now has access
        self.assertEqual(db.get_user_permission_for_document(self.viewer_id, doc_id), "viewer")
        self.assertTrue(db.can_user_access_document(self.viewer_id, doc_id, required_level="viewer"))
        # But Viewer cannot edit
        self.assertFalse(db.can_user_access_document(self.viewer_id, doc_id, required_level="editor"))

        # Check accessible document listing
        accessible_ids = db.get_accessible_document_ids(self.viewer_id)
        self.assertIn(doc_id, accessible_ids)

        # Revoke access
        revoked = db.revoke_document_permission(doc_id, self.viewer_id)
        self.assertTrue(revoked)
        self.assertNotIn(doc_id, db.get_accessible_document_ids(self.viewer_id))

    def test_04_document_ownership_transfer(self):
        """Verify transferring document ownership to another user."""
        doc_id = db.add_document(
            user_id=self.editor_id,
            file_name="transferable_report.pdf",
            file_size=2048,
            file_path="/tmp/transferable_report.pdf",
            category="General"
        )

        success = db.transfer_document_ownership(
            document_id=doc_id,
            new_owner_id=self.viewer_id,
            transferred_by=self.editor_id
        )
        self.assertTrue(success)

        # Viewer is now the primary owner
        new_doc = db.get_document_by_id(doc_id)
        self.assertEqual(new_doc["user_id"], self.viewer_id)
        self.assertEqual(db.get_user_permission_for_document(self.viewer_id, doc_id), "owner")

    def test_05_rag_permission_security_filter(self):
        """Verify that RAG chunk filtering drops chunks belonging to unauthorized documents."""
        permitted_chunk = {
            "text": "This is accessible public context.",
            "doc_id": 101,
            "file_name": "allowed.pdf"
        }
        unauthorized_chunk = {
            "text": "Top secret confidential leak.",
            "doc_id": 999,
            "file_name": "classified.pdf"
        }

        accessible_doc_ids = {101, 102}

        # Permitted chunk passes
        self.assertTrue(RAGManager._matches_filters(
            item=permitted_chunk,
            accessible_doc_ids=accessible_doc_ids
        ))

        # Unauthorized chunk is strictly rejected
        self.assertFalse(RAGManager._matches_filters(
            item=unauthorized_chunk,
            accessible_doc_ids=accessible_doc_ids
        ))


if __name__ == "__main__":
    unittest.main()
