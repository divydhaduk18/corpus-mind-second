"""SQLAlchemy ORM Models for Corpus Mind (SQLite and PostgreSQL)."""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column, Integer, String, Text, Float, DateTime, ForeignKey, UniqueConstraint, Index
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class User(Base):
    """User account model with Role-Based Access Control and status."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False, default="editor")  # admin, editor, viewer
    status = Column(String(20), nullable=False, default="active")  # active, deactivated
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    documents = relationship("Document", back_populates="owner", cascade="all, delete-orphan")
    chat_sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")
    permissions = relationship("DocumentPermission", foreign_keys="DocumentPermission.user_id", back_populates="user", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user")


class Document(Base):
    """Uploaded document model with hash deduplication and category metadata."""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    file_name = Column(String(255), nullable=False)
    file_size = Column(Integer, nullable=False)
    file_path = Column(String(1024), nullable=False)
    upload_time = Column(DateTime, default=datetime.utcnow)
    status = Column(String(50), default="uploaded")
    chunk_count = Column(Integer, default=0)
    doc_hash = Column(String(64), index=True)
    category = Column(String(100), default="General")

    # Relationships
    owner = relationship("User", back_populates="documents")
    permissions = relationship("DocumentPermission", back_populates="document", cascade="all, delete-orphan")


class DocumentPermission(Base):
    """Document-level permissions for sharing and fine-grained access control."""
    __tablename__ = "document_permissions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    permission_level = Column(String(20), nullable=False, default="viewer")  # owner, editor, viewer
    granted_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("document_id", "user_id", name="uq_doc_user_perm"),
    )

    # Relationships
    document = relationship("Document", back_populates="permissions")
    user = relationship("User", foreign_keys=[user_id], back_populates="permissions")


class ChatSession(Base):
    """Conversation session model for multi-turn conversational RAG."""
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="chat_sessions")
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")


class Message(Base):
    """Individual chat messages with generation metadata."""
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    model = Column(String(100), nullable=True)
    response_time_ms = Column(Integer, default=0)

    # Relationships
    session = relationship("ChatSession", back_populates="messages")


class DocumentJob(Base):
    """Asynchronous background document processing job status."""
    __tablename__ = "document_jobs"

    job_id = Column(String(100), primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    file_name = Column(String(255), nullable=False)
    file_path = Column(String(1024), nullable=False)
    status = Column(String(50), default="queued")
    progress_pct = Column(Integer, default=0)
    current_stage = Column(String(100), default="queued")
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class RagasEvaluation(Base):
    """Persistent storage for RAGAS evaluation results."""
    __tablename__ = "ragas_evaluations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    dataset_name = Column(String(255), nullable=False)
    provider = Column(String(100), nullable=False)
    model = Column(String(100), nullable=False)
    embedding_model = Column(String(100), nullable=False)
    avg_faithfulness = Column(Float, default=0.0)
    avg_answer_relevancy = Column(Float, default=0.0)
    avg_context_precision = Column(Float, default=0.0)
    avg_context_recall = Column(Float, default=0.0)
    avg_answer_correctness = Column(Float, default=0.0)
    avg_context_utilization = Column(Float, default=0.0)
    metrics_json = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)


class AnalyticsLog(Base):
    """System analytics and telemetry event logs."""
    __tablename__ = "analytics_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    event_type = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    response_time_ms = Column(Integer, default=0)
    model_used = Column(String(100), nullable=True)


class AuditLog(Base):
    """Security audit log tracking all auth, access, document, and admin actions."""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    username = Column(String(100), nullable=True)
    action = Column(String(100), nullable=False, index=True)
    resource_type = Column(String(50), default="system", index=True)
    resource_id = Column(String(100), nullable=True)
    status = Column(String(20), default="SUCCESS", index=True)  # SUCCESS, FAILURE, DENIED
    ip_address = Column(String(45), nullable=True)
    details_json = Column(Text, nullable=True)

    # Relationship
    user = relationship("User", back_populates="audit_logs")
