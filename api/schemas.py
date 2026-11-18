"""Pydantic V2 Schemas for Corpus Mind REST API (Improvement 32)."""

from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field


# ==========================================
# Authentication Schemas
# ==========================================

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int
    user_id: int
    username: str
    role: str


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=1)


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=100)
    password: str = Field(..., min_length=6)
    role: Optional[str] = Field("editor", description="editor or viewer (cannot self-register as admin)")


class UserResponse(BaseModel):
    id: int
    username: str
    role: str
    status: str
    created_at: Optional[Union[str, datetime]] = None
    updated_at: Optional[Union[str, datetime]] = None


class UserRoleUpdateRequest(BaseModel):
    role: str = Field(..., description="admin, editor, or viewer")


class UserStatusUpdateRequest(BaseModel):
    status: str = Field(..., description="active or deactivated")


# ==========================================
# Document & Permission Schemas
# ==========================================

class DocumentResponse(BaseModel):
    id: int
    user_id: int
    file_name: str
    file_size: int
    upload_time: Optional[Union[str, datetime]] = None
    status: Optional[str] = "uploaded"
    chunk_count: int = 0
    category: Optional[str] = "General"
    effective_permission: Optional[str] = "owner"
    owner_username: Optional[str] = None


class PermissionGrantRequest(BaseModel):
    target_user_id: int
    permission_level: str = Field(..., description="owner, editor, or viewer")


class OwnershipTransferRequest(BaseModel):
    new_owner_id: int


class PermissionResponse(BaseModel):
    id: int
    document_id: int
    user_id: int
    user_name: Optional[str] = None
    permission_level: str
    granted_by: Optional[int] = None
    granted_by_name: Optional[str] = None
    created_at: Optional[Union[str, datetime]] = None


# ==========================================
# RAG, Search & Chat Schemas
# ==========================================

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: Optional[int] = Field(4, ge=1, le=20)
    search_mode: Optional[str] = Field("hybrid", description="hybrid, dense, or bm25")
    doc_filter: Optional[List[str]] = None
    metadata_filters: Optional[Dict[str, Any]] = None
    enable_reranking: Optional[bool] = None


class SearchChunkResponse(BaseModel):
    text: str
    doc_id: Optional[int] = None
    file_name: Optional[str] = None
    page_number: Optional[int] = 1
    score: Optional[float] = 0.0
    search_mode: Optional[str] = None
    file_type: Optional[str] = None
    category: Optional[str] = None


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: Optional[int] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    top_k: Optional[int] = Field(4, ge=1, le=20)
    search_mode: Optional[str] = "hybrid"
    doc_filter: Optional[List[str]] = None
    metadata_filters: Optional[Dict[str, Any]] = None
    temperature: Optional[float] = Field(0.2, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(1000, ge=50, le=4096)


class ChatResponse(BaseModel):
    session_id: int
    answer: str
    model: str
    provider: str
    sources: List[Dict[str, Any]] = []
    faithfulness_score: Optional[float] = None
    latency_ms: Optional[float] = None


class DocumentSummarizeRequest(BaseModel):
    summary_type: Optional[str] = Field("executive", description="executive, key_insights, bullet_points, comprehensive")
    provider: Optional[str] = None
    model: Optional[str] = None


class DocumentSummarizeResponse(BaseModel):
    document_id: int
    file_name: str
    summary_type: str
    summary: str
    latency_ms: float


# ==========================================
# Audit Log Schemas
# ==========================================

class AuditLogResponse(BaseModel):
    id: int
    timestamp: Optional[Union[str, datetime]] = None
    user_id: Optional[int] = None
    username: Optional[str] = None
    action: str
    resource_type: Optional[str] = "system"
    resource_id: Optional[str] = None
    status: Optional[str] = "SUCCESS"
    ip_address: Optional[str] = None
    details_json: Optional[str] = None


class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None
