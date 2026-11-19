"""Database management layer for Corpus Mind.

Supports dual relational database engines:
- SQLite (Local development, tests, and zero-configuration setups)
- PostgreSQL (High-concurrency production deployments via SQLAlchemy connection pooling)

Provides schema management, Role-Based Access Control (RBAC), fine-grained
document permissions, chat histories, asynchronous jobs, and security audit logs.
"""

import os
import sys
import json
import sqlite3
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Union

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.models import Base

logger = logging.getLogger(__name__)

# SQLAlchemy support for connection pooling and PostgreSQL
try:
    from sqlalchemy import create_engine, text, inspect
    from sqlalchemy.pool import QueuePool, NullPool
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False


class DatabaseManager:
    """Unified database manager supporting both SQLite and PostgreSQL."""

    def __init__(self, db_url: Optional[str] = None, db_path: Optional[str] = None):
        self.db_url = db_url or getattr(config, "DATABASE_URL", f"sqlite:///{config.DB_PATH}")
        self.is_postgres = self.db_url.startswith("postgresql")
        self.db_path = str(db_path or config.DB_PATH)
        
        self.engine = None
        if SQLALCHEMY_AVAILABLE:
            try:
                if self.is_postgres:
                    self.engine = create_engine(
                        self.db_url,
                        poolclass=QueuePool,
                        pool_size=getattr(config, "DB_POOL_SIZE", 10),
                        max_overflow=getattr(config, "DB_MAX_OVERFLOW", 20),
                        pool_timeout=getattr(config, "DB_POOL_TIMEOUT", 30),
                        pool_pre_ping=True
                    )
                else:
                    self.engine = create_engine(
                        self.db_url,
                        poolclass=NullPool
                    )
            except Exception as e:
                logger.warning(f"Could not initialize SQLAlchemy engine: {e}")

        self.initialize_tables()

    def get_connection(self):
        """Returns a connection context suitable for the active database engine."""
        if self.is_postgres and self.engine:
            return self.engine.connect()
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _execute(self, query: str, params: Union[tuple, list, dict] = None, fetch_one: bool = False, fetch_all: bool = False, commit: bool = True):
        """Execute a parameterized query safely on either SQLite or PostgreSQL."""
        if params is None:
            params = ()
        elif isinstance(params, list):
            params = tuple(params)

        if self.is_postgres and self.engine:
            # PostgreSQL execution via SQLAlchemy text
            # Replace positional '?' with numbered ':param_i'
            param_dict = {}
            if isinstance(params, (tuple, list)):
                parts = query.split("?")
                new_q = []
                for i, part in enumerate(parts):
                    new_q.append(part)
                    if i < len(params):
                        pname = f"p_{i}"
                        new_q.append(f":{pname}")
                        param_dict[pname] = params[i]
                stmt = text("".join(new_q))
            else:
                stmt = text(query)
                param_dict = params

            with self.engine.connect() as conn:
                result = conn.execute(stmt, param_dict)
                if commit:
                    conn.commit()
                if fetch_one:
                    row = result.fetchone()
                    return dict(row._mapping) if row else None
                if fetch_all:
                    return [dict(r._mapping) for r in result.fetchall()]
                return result.rowcount
        else:
            # SQLite execution
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query, params)
                if commit:
                    conn.commit()
                if fetch_one:
                    row = cursor.fetchone()
                    return dict(row) if row else None
                if fetch_all:
                    return [dict(r) for r in cursor.fetchall()]
                return cursor.lastrowid or cursor.rowcount

    def initialize_tables(self):
        """Initializes all schema tables and applies incremental migrations."""
        if self.is_postgres and self.engine:
            try:
                Base.metadata.create_all(self.engine)
                logger.info("PostgreSQL tables successfully verified/created via SQLAlchemy Base.")
                return
            except Exception as pe:
                logger.error(f"Error creating PostgreSQL tables: {pe}")

        # SQLite Native Table Creation
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # 1. Users table with RBAC and status
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'editor',
                status TEXT DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # Migration: Ensure status and role exist and default properly
            cursor.execute("PRAGMA table_info(users)")
            user_cols = [row[1] for row in cursor.fetchall()]
            if "status" not in user_cols:
                cursor.execute("ALTER TABLE users ADD COLUMN status TEXT DEFAULT 'active'")
            if "updated_at" not in user_cols:
                cursor.execute("ALTER TABLE users ADD COLUMN updated_at TIMESTAMP")
            # Normalize legacy 'user' roles to 'editor'
            cursor.execute("UPDATE users SET role = 'editor' WHERE role = 'user'")

            # 2. Chat sessions table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """)

            # 3. Messages table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                model TEXT,
                response_time_ms INTEGER,
                FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
            )
            """)

            # 4. Documents table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                file_name TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'uploaded',
                chunk_count INTEGER DEFAULT 0,
                doc_hash TEXT,
                category TEXT DEFAULT 'General',
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """)

            cursor.execute("PRAGMA table_info(documents)")
            doc_cols = [row[1] for row in cursor.fetchall()]
            if "doc_hash" not in doc_cols:
                cursor.execute("ALTER TABLE documents ADD COLUMN doc_hash TEXT")
            if "category" not in doc_cols:
                cursor.execute("ALTER TABLE documents ADD COLUMN category TEXT DEFAULT 'General'")

            # 5. Document Permissions table (Fine-grained document sharing & RBAC)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_permissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                permission_level TEXT NOT NULL,
                granted_by INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (granted_by) REFERENCES users(id) ON DELETE SET NULL,
                UNIQUE(document_id, user_id)
            )
            """)

            # 6. Analytics logs table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS analytics_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                description TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                response_time_ms INTEGER DEFAULT 0,
                model_used TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """)

            # 7. Asynchronous Background Document Jobs Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_jobs (
                job_id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                file_name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                status TEXT DEFAULT 'queued',
                progress_pct INTEGER DEFAULT 0,
                current_stage TEXT DEFAULT 'queued',
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """)

            # 8. RAGAS Evaluation Results Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS ragas_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                dataset_name TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                avg_faithfulness REAL DEFAULT 0.0,
                avg_answer_relevancy REAL DEFAULT 0.0,
                avg_context_precision REAL DEFAULT 0.0,
                avg_context_recall REAL DEFAULT 0.0,
                avg_answer_correctness REAL DEFAULT 0.0,
                avg_context_utilization REAL DEFAULT 0.0,
                metrics_json TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """)

            # 9. Security & Compliance Audit Logs Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id INTEGER,
                username TEXT,
                action TEXT NOT NULL,
                resource_type TEXT DEFAULT 'system',
                resource_id TEXT,
                status TEXT DEFAULT 'SUCCESS',
                ip_address TEXT,
                details_json TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
            )
            """)

            # Indexes for high-performance querying
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_logs(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_logs(action)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_user_id ON audit_logs(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_perm_user ON document_permissions(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_perm_doc ON document_permissions(document_id)")

            conn.commit()
            logger.info("Database tables and indexes initialized successfully.")

    # ==========================================
    # 28. RBAC & User Management Operations
    # ==========================================

    def create_user(self, username: str, password_hash: str, role: str = 'editor', status: str = 'active') -> Optional[int]:
        """Creates a new user with designated role and status."""
        try:
            # Normalize legacy 'user' role to 'editor'
            if role == 'user':
                role = 'editor'
            return self._execute(
                "INSERT INTO users (username, password_hash, role, status) VALUES (?, ?, ?, ?)",
                (username, password_hash, role, status)
            )
        except Exception as e:
            logger.warning(f"Failed to create user '{username}': {e}")
            return None

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Retrieves user profile by username."""
        return self._execute("SELECT * FROM users WHERE username = ?", (username,), fetch_one=True)

    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves user profile by user ID."""
        return self._execute("SELECT * FROM users WHERE id = ?", (user_id,), fetch_one=True)

    def get_all_users(self) -> List[Dict[str, Any]]:
        """Retrieves list of all users for admin management."""
        return self._execute(
            "SELECT id, username, role, status, created_at, updated_at FROM users ORDER BY id ASC",
            fetch_all=True
        ) or []

    def update_user_role(self, user_id: int, new_role: str) -> bool:
        """Updates user authorization role (admin, editor, viewer)."""
        valid_roles = getattr(config, "VALID_ROLES", ["admin", "editor", "viewer"])
        if new_role not in valid_roles:
            return False
        res = self._execute(
            "UPDATE users SET role = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (new_role, user_id)
        )
        return res is not None and res > 0

    def update_user_status(self, user_id: int, new_status: str) -> bool:
        """Updates user account status ('active' or 'deactivated')."""
        if new_status not in ["active", "deactivated"]:
            return False
        res = self._execute(
            "UPDATE users SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (new_status, user_id)
        )
        return res is not None and res > 0

    def update_user_password(self, user_id: int, password_hash: str) -> bool:
        """Updates user's password hash."""
        res = self._execute(
            "UPDATE users SET password_hash = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (password_hash, user_id)
        )
        return res is not None and res > 0

    def delete_user(self, user_id: int) -> bool:
        """Permanently deletes a user account."""
        res = self._execute("DELETE FROM users WHERE id = ?", (user_id,))
        return res is not None and res > 0

    # ==========================================
    # 29. Document-Level Permissions Operations
    # ==========================================

    def grant_document_permission(
        self,
        document_id: int,
        user_id: int,
        permission_level: str,
        granted_by: Optional[int] = None
    ) -> bool:
        """Grants or updates document-level access permission ('owner', 'editor', 'viewer')."""
        valid_perms = getattr(config, "VALID_PERMISSIONS", ["owner", "editor", "viewer", "none"])
        if permission_level not in valid_perms:
            return False

        if permission_level == "none":
            return self.revoke_document_permission(document_id, user_id)

        # Upsert permission
        existing = self._execute(
            "SELECT id FROM document_permissions WHERE document_id = ? AND user_id = ?",
            (document_id, user_id),
            fetch_one=True
        )
        if existing:
            res = self._execute(
                "UPDATE document_permissions SET permission_level = ?, granted_by = ?, created_at = CURRENT_TIMESTAMP WHERE document_id = ? AND user_id = ?",
                (permission_level, granted_by, document_id, user_id)
            )
        else:
            res = self._execute(
                "INSERT INTO document_permissions (document_id, user_id, permission_level, granted_by) VALUES (?, ?, ?, ?)",
                (document_id, user_id, permission_level, granted_by)
            )
        return res is not None and res > 0

    def revoke_document_permission(self, document_id: int, user_id: int) -> bool:
        """Revokes a user's explicit permission on a document."""
        res = self._execute(
            "DELETE FROM document_permissions WHERE document_id = ? AND user_id = ?",
            (document_id, user_id)
        )
        return res is not None and res > 0

    def get_document_permissions(self, document_id: int) -> List[Dict[str, Any]]:
        """Returns all permission entries for a document with user details."""
        q = """
        SELECT dp.id, dp.document_id, dp.user_id, dp.permission_level, dp.granted_by, dp.created_at,
               u.username as user_name, gb.username as granted_by_name
        FROM document_permissions dp
        JOIN users u ON dp.user_id = u.id
        LEFT JOIN users gb ON dp.granted_by = gb.id
        WHERE dp.document_id = ?
        ORDER BY dp.created_at ASC
        """
        return self._execute(q, (document_id,), fetch_all=True) or []

    def get_user_permission_for_document(self, user_id: int, document_id: int) -> Optional[str]:
        """Resolves the effective permission level of a user for a document."""
        user = self.get_user_by_id(user_id)
        if not user or user.get("status") == "deactivated":
            return None

        # Admins have full access
        if user.get("role") == getattr(config, "ROLE_ADMIN", "admin"):
            return "owner"

        # Check direct document ownership
        doc = self._execute("SELECT user_id FROM documents WHERE id = ?", (document_id,), fetch_one=True)
        if not doc:
            return None
        if doc["user_id"] == user_id:
            return "owner"

        # Check explicit shared permissions
        perm = self._execute(
            "SELECT permission_level FROM document_permissions WHERE document_id = ? AND user_id = ?",
            (document_id, user_id),
            fetch_one=True
        )
        if perm:
            return perm["permission_level"]

        return None

    def transfer_document_ownership(self, document_id: int, new_owner_id: int, transferred_by: Optional[int] = None) -> bool:
        """Transfers primary document ownership to another user."""
        doc = self.get_document_by_id(document_id)
        if not doc:
            return False
        old_owner_id = doc["user_id"]

        # Update primary user_id
        res = self._execute(
            "UPDATE documents SET user_id = ? WHERE id = ?",
            (new_owner_id, document_id)
        )
        if res:
            # Grant previous owner editor access so they do not lose context unless explicitly revoked
            self.grant_document_permission(document_id, old_owner_id, "editor", granted_by=transferred_by)
            # Remove any existing permission entry for the new owner since they now own it
            self.revoke_document_permission(document_id, new_owner_id)
            return True
        return False

    def get_accessible_documents(self, user_id: int) -> List[Dict[str, Any]]:
        """Returns all documents accessible to the user (owned or shared)."""
        user = self.get_user_by_id(user_id)
        if not user or user.get("status") == "deactivated":
            return []

        # Admin can access all system documents
        if user.get("role") == getattr(config, "ROLE_ADMIN", "admin"):
            return self.get_all_documents()

        # Regular user: owned documents + explicitly shared documents
        q = """
        SELECT d.*, 
               CASE 
                   WHEN d.user_id = ? THEN 'owner'
                   ELSE COALESCE(dp.permission_level, 'none')
               END as effective_permission,
               u.username as owner_username
        FROM documents d
        JOIN users u ON d.user_id = u.id
        LEFT JOIN document_permissions dp ON d.id = dp.document_id AND dp.user_id = ?
        WHERE d.user_id = ? OR dp.permission_level IN ('owner', 'editor', 'viewer')
        ORDER BY d.upload_time DESC
        """
        return self._execute(q, (user_id, user_id, user_id), fetch_all=True) or []

    def get_accessible_document_ids(self, user_id: int) -> List[int]:
        """Returns list of document IDs accessible to the user."""
        docs = self.get_accessible_documents(user_id)
        return [d["id"] for d in docs]

    def can_user_access_document(self, user_id: int, document_id: int, required_level: str = "viewer") -> bool:
        """Validates if user has at least the required permission level on a document."""
        perm = self.get_user_permission_for_document(user_id, document_id)
        if not perm:
            return False

        hierarchy = {"viewer": 1, "editor": 2, "owner": 3}
        return hierarchy.get(perm, 0) >= hierarchy.get(required_level, 1)

    # ==========================================
    # 33. Audit Logging Operations
    # ==========================================

    def create_audit_log(
        self,
        user_id: Optional[int],
        username: Optional[str],
        action: str,
        resource_type: str = "system",
        resource_id: Optional[str] = None,
        status: str = "SUCCESS",
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> Optional[int]:
        """Records an auditable security or system event."""
        if not getattr(config, "AUDIT_LOGS_ENABLED", True):
            return None

        # Sanitize details payload
        details_json = None
        if details:
            safe_details = self._sanitize_audit_payload(details)
            details_json = json.dumps(safe_details)

        try:
            return self._execute(
                """
                INSERT INTO audit_logs (user_id, username, action, resource_type, resource_id, status, ip_address, details_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (user_id, username, action, resource_type, str(resource_id) if resource_id is not None else None, status, ip_address, details_json)
            )
        except Exception as e:
            logger.error(f"Failed to record audit log: {e}")
            return None

    @staticmethod
    def _sanitize_audit_payload(data: Any) -> Any:
        """Removes passwords, API keys, and sensitive tokens from audit records."""
        sensitive_keys = {
            "password", "password_hash", "token", "access_token", "jwt",
            "api_key", "secret", "authorization", "raw_content", "credentials"
        }
        if isinstance(data, dict):
            clean = {}
            for k, v in data.items():
                if any(sk in k.lower() for sk in sensitive_keys):
                    clean[k] = "[REDACTED]"
                else:
                    clean[k] = DatabaseManager._sanitize_audit_payload(v)
            return clean
        elif isinstance(data, list):
            return [DatabaseManager._sanitize_audit_payload(item) for item in data]
        return data

    def get_audit_logs(
        self,
        user_id: Optional[int] = None,
        action: Optional[str] = None,
        resource_type: Optional[str] = None,
        status: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Queries audit logs with dynamic multi-field filtering."""
        conditions = []
        params = []

        if user_id is not None:
            conditions.append("user_id = ?")
            params.append(user_id)
        if action:
            conditions.append("action LIKE ?")
            params.append(f"%{action}%")
        if resource_type:
            conditions.append("resource_type = ?")
            params.append(resource_type)
        if status:
            conditions.append("status = ?")
            params.append(status)
        if start_date:
            conditions.append("timestamp >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("timestamp <= ?")
            params.append(end_date)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT * FROM audit_logs {where_clause} ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        return self._execute(query, tuple(params), fetch_all=True) or []

    def count_audit_logs(
        self,
        user_id: Optional[int] = None,
        action: Optional[str] = None,
        resource_type: Optional[str] = None,
        status: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> int:
        """Counts total audit log entries matching filter criteria."""
        conditions = []
        params = []

        if user_id is not None:
            conditions.append("user_id = ?")
            params.append(user_id)
        if action:
            conditions.append("action LIKE ?")
            params.append(f"%{action}%")
        if resource_type:
            conditions.append("resource_type = ?")
            params.append(resource_type)
        if status:
            conditions.append("status = ?")
            params.append(status)
        if start_date:
            conditions.append("timestamp >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("timestamp <= ?")
            params.append(end_date)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT COUNT(*) as cnt FROM audit_logs {where_clause}"
        res = self._execute(query, tuple(params) if params else None, fetch_one=True)
        return res["cnt"] if res else 0

    # ==========================================
    # Chat Sessions operations
    # ==========================================

    def create_chat_session(self, user_id: int, title: str) -> Optional[int]:
        """Creates a new chat session."""
        return self._execute(
            "INSERT INTO chat_sessions (user_id, title) VALUES (?, ?)",
            (user_id, title)
        )

    def get_chat_sessions_by_user(self, user_id: int) -> List[Dict[str, Any]]:
        """Retrieves all chat sessions for a user."""
        return self._execute(
            "SELECT * FROM chat_sessions WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
            fetch_all=True
        ) or []

    def update_chat_session_title(self, session_id: int, new_title: str, user_id: Optional[int] = None) -> bool:
        """Updates the title of a chat session."""
        if user_id:
            res = self._execute(
                "UPDATE chat_sessions SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ? AND user_id = ?",
                (new_title, session_id, user_id)
            )
        else:
            res = self._execute(
                "UPDATE chat_sessions SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (new_title, session_id)
            )
        return res is not None and res > 0

    def delete_chat_session(self, session_id: int, user_id: Optional[int] = None) -> bool:
        """Deletes a chat session and its associated messages."""
        if user_id:
            res = self._execute("DELETE FROM chat_sessions WHERE id = ? AND user_id = ?", (session_id, user_id))
        else:
            res = self._execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
        return res is not None and res > 0

    # ==========================================
    # Messages operations
    # ==========================================

    def add_message(self, session_id: int, role: str, content: str, model: Optional[str] = None, response_time_ms: Optional[int] = None) -> Optional[int]:
        """Adds a message to a chat session."""
        msg_id = self._execute(
            """
            INSERT INTO messages (session_id, role, content, model, response_time_ms)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, role, content, model, response_time_ms)
        )
        self._execute("UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (session_id,))
        return msg_id

    def get_messages_by_session(self, session_id: int) -> List[Dict[str, Any]]:
        """Retrieves all messages for a session."""
        return self._execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp ASC",
            (session_id,),
            fetch_all=True
        ) or []

    def delete_last_message(self, session_id: int) -> bool:
        """Deletes the most recent message in a session."""
        last_msg = self._execute(
            "SELECT id FROM messages WHERE session_id = ? ORDER BY timestamp DESC LIMIT 1",
            (session_id,),
            fetch_one=True
        )
        if last_msg:
            res = self._execute("DELETE FROM messages WHERE id = ?", (last_msg['id'],))
            return res is not None and res > 0
        return False

    # ==========================================
    # Document operations
    # ==========================================

    def add_document(
        self,
        user_id: int,
        file_name: str,
        file_size: int,
        file_path: str,
        chunk_count: int = 0,
        doc_hash: Optional[str] = None,
        category: Optional[str] = "General"
    ) -> Optional[int]:
        """Adds a document record to the database."""
        return self._execute(
            """
            INSERT INTO documents (user_id, file_name, file_size, file_path, chunk_count, doc_hash, category)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, file_name, file_size, file_path, chunk_count, doc_hash, category or "General")
        )

    def get_documents_by_user(self, user_id: int) -> List[Dict[str, Any]]:
        """Retrieves documents directly owned by a user."""
        return self._execute(
            "SELECT * FROM documents WHERE user_id = ? ORDER BY upload_time DESC",
            (user_id,),
            fetch_all=True
        ) or []

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Retrieves all documents across all users for admin view."""
        return self._execute(
            """
            SELECT d.*, u.username as owner_username, 'owner' as effective_permission
            FROM documents d
            JOIN users u ON d.user_id = u.id
            ORDER BY d.upload_time DESC
            """,
            fetch_all=True
        ) or []

    def get_document_by_id(self, document_id: int, user_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Retrieves a document by ID."""
        if user_id:
            return self._execute(
                "SELECT * FROM documents WHERE id = ? AND user_id = ?",
                (document_id, user_id),
                fetch_one=True
            )
        return self._execute("SELECT * FROM documents WHERE id = ?", (document_id,), fetch_one=True)

    def get_document_by_hash(self, user_id: int, doc_hash: str) -> Optional[Dict[str, Any]]:
        """Finds an existing document for user by content hash."""
        return self._execute(
            "SELECT * FROM documents WHERE user_id = ? AND doc_hash = ?",
            (user_id, doc_hash),
            fetch_one=True
        )

    def delete_document(self, document_id: int, user_id: Optional[int] = None) -> bool:
        """Deletes a document from the database."""
        if user_id:
            # Check if user is owner or admin
            user = self.get_user_by_id(user_id)
            is_admin = user and user.get("role") == getattr(config, "ROLE_ADMIN", "admin")
            if is_admin:
                res = self._execute("DELETE FROM documents WHERE id = ?", (document_id,))
            else:
                res = self._execute("DELETE FROM documents WHERE id = ? AND user_id = ?", (document_id, user_id))
        else:
            res = self._execute("DELETE FROM documents WHERE id = ?", (document_id,))
        return res is not None and res > 0

    # ==========================================
    # Analytics logs operations
    # ==========================================

    def log_analytics(self, user_id: int, event_type: str, description: Optional[str] = None, response_time_ms: int = 0, model_used: Optional[str] = None):
        """Logs an event for usage analytics."""
        self._execute(
            """
            INSERT INTO analytics_logs (user_id, event_type, description, response_time_ms, model_used)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, event_type, description, response_time_ms, model_used)
        )

    def get_analytics_summary(self) -> Dict[str, Any]:
        """Calculates system-wide analytics including model usage and message history."""
        total_queries = self._execute("SELECT COUNT(*) as count FROM analytics_logs WHERE event_type = 'query'", fetch_one=True)
        total_docs = self._execute("SELECT COUNT(*) as count FROM documents", fetch_one=True)
        total_users = self._execute("SELECT COUNT(*) as count FROM users", fetch_one=True)
        avg_resp = self._execute("SELECT AVG(response_time_ms) as avg_time FROM analytics_logs WHERE event_type = 'query'", fetch_one=True)

        # Count user messages from messages table (falls back to query count if table missing)
        try:
            total_msgs = self._execute("SELECT COUNT(*) as count FROM messages WHERE role = 'user'", fetch_one=True)
            user_messages = total_msgs["count"] if total_msgs else 0
        except Exception:
            user_messages = total_queries["count"] if total_queries else 0

        # Model usage distribution: list of (model_name, count) tuples
        try:
            model_rows = self._execute(
                "SELECT model_used, COUNT(*) as cnt FROM analytics_logs WHERE model_used IS NOT NULL GROUP BY model_used ORDER BY cnt DESC",
                fetch_all=True
            )
            model_usage = [(r["model_used"], r["cnt"]) for r in model_rows] if model_rows else []
        except Exception:
            model_usage = []

        # Message volume per day: list of (date_str, count) tuples — tries messages table first
        try:
            hist_rows = self._execute(
                "SELECT DATE(timestamp) as day, COUNT(*) as cnt FROM messages WHERE role = 'user' GROUP BY day ORDER BY day ASC",
                fetch_all=True
            )
            msg_history = [(r["day"], r["cnt"]) for r in hist_rows] if hist_rows else []
        except Exception:
            try:
                hist_rows = self._execute(
                    "SELECT DATE(timestamp) as day, COUNT(*) as cnt FROM analytics_logs WHERE event_type = 'query' GROUP BY day ORDER BY day ASC",
                    fetch_all=True
                )
                msg_history = [(r["day"], r["cnt"]) for r in hist_rows] if hist_rows else []
            except Exception:
                msg_history = []

        total_q = total_queries["count"] if total_queries else 0

        return {
            "total_queries": total_q,
            "total_user_messages": user_messages,
            "total_documents": total_docs["count"] if total_docs else 0,
            "total_users": total_users["count"] if total_users else 0,
            "avg_response_time_ms": round(avg_resp["avg_time"] or 0, 2) if avg_resp else 0,
            "model_usage": model_usage,
            "msg_history": msg_history,
            "total_size_bytes": 0,  # legacy key kept for backward compatibility
        }

    # ==========================================
    # Asynchronous Background Document Jobs
    # ==========================================

    def create_document_job(self, job_id: str, user_id: int, file_name: str, file_path: str) -> bool:
        """Creates a background processing job entry."""
        res = self._execute(
            """
            INSERT INTO document_jobs (job_id, user_id, file_name, file_path, status, progress_pct, current_stage)
            VALUES (?, ?, ?, ?, 'queued', 0, 'queued')
            """,
            (job_id, user_id, file_name, file_path)
        )
        return res is not None and res > 0

    def update_document_job(
        self,
        job_id: str,
        status: str,
        progress_pct: int,
        current_stage: str,
        error_message: Optional[str] = None
    ) -> bool:
        """Updates the status and progress of a background job."""
        res = self._execute(
            """
            UPDATE document_jobs 
            SET status = ?, progress_pct = ?, current_stage = ?, error_message = ?, updated_at = CURRENT_TIMESTAMP
            WHERE job_id = ?
            """,
            (status, progress_pct, current_stage, error_message, job_id)
        )
        return res is not None and res > 0

    def get_document_job_by_id(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a background job by ID."""
        return self._execute("SELECT * FROM document_jobs WHERE job_id = ?", (job_id,), fetch_one=True)

    def get_document_jobs_by_user(self, user_id: int) -> List[Dict[str, Any]]:
        """Retrieves all background jobs for a specific user."""
        return self._execute(
            "SELECT * FROM document_jobs WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
            fetch_all=True
        ) or []

    # ==========================================
    # RAGAS Evaluation Results Persistence
    # ==========================================

    def save_ragas_evaluation(
        self,
        user_id: int,
        dataset_name: str,
        provider: str,
        model: str,
        embedding_model: str,
        metrics: Dict[str, Any]
    ) -> int:
        """Saves a RAGAS evaluation run to the database."""
        metrics_json = json.dumps(metrics)
        return self._execute(
            """
            INSERT INTO ragas_evaluations (
                user_id, dataset_name, provider, model, embedding_model,
                avg_faithfulness, avg_answer_relevancy, avg_context_precision,
                avg_context_recall, avg_answer_correctness, avg_context_utilization,
                metrics_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                dataset_name,
                provider,
                model,
                embedding_model,
                metrics.get("faithfulness", 0.0),
                metrics.get("answer_relevancy", 0.0),
                metrics.get("context_precision", 0.0),
                metrics.get("context_recall", 0.0),
                metrics.get("answer_correctness", 0.0),
                metrics.get("context_utilization", 0.0),
                metrics_json
            )
        )

    def get_ragas_evaluations_by_user(self, user_id: int) -> List[Dict[str, Any]]:
        """Retrieves all historical RAGAS evaluations for a user."""
        return self._execute(
            "SELECT * FROM ragas_evaluations WHERE user_id = ? ORDER BY timestamp DESC",
            (user_id,),
            fetch_all=True
        ) or []


# Global Singleton Database Instance
db = DatabaseManager()
