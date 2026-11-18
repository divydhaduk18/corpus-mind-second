"""Document Management & Document-Level Permissions REST Endpoints (Improvements 29 & 32)."""

import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Request
from fastapi.responses import FileResponse

import config
from utils.database import db
from utils.rag import RAGManager
from utils.document_loader import DocumentLoader
from utils.text_splitter import RecursiveTextSplitter
from utils.audit import (
    AuditLogger, ACTION_DOC_UPLOAD, ACTION_DOC_VIEW, ACTION_DOC_DOWNLOAD,
    ACTION_DOC_DELETE, ACTION_DOC_PERM_GRANT, ACTION_DOC_PERM_REVOKE, ACTION_DOC_OWNERSHIP_TRANSFER
)
from api.schemas import (
    DocumentResponse, PermissionResponse, PermissionGrantRequest,
    OwnershipTransferRequest, APIResponse
)
from api.dependencies import get_current_user, require_editor_or_admin, check_rate_limit

router = APIRouter(prefix="/api/v1/documents", tags=["Documents & Permissions"])


@router.post("/upload", response_model=DocumentResponse, dependencies=[Depends(check_rate_limit)])
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    category: str = Form("General"),
    current_user: Dict[str, Any] = Depends(require_editor_or_admin)
):
    """Upload and ingest a document or image into the private RAG index."""
    user_id = current_user["id"]
    filename = file.filename
    ext = Path(filename).suffix.lower()

    if ext not in config.SUPPORTED_DOC_EXTENSIONS and ext not in config.SUPPORTED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Supported formats: {list(config.SUPPORTED_DOC_EXTENSIONS | config.SUPPORTED_IMAGE_EXTENSIONS)}"
        )

    file_bytes = await file.read()
    file_size = len(file_bytes)

    user_upload_dir = config.get_user_uploads_dir(user_id)
    safe_storage_name = f"{int(time.time() * 1000)}_{Path(filename).name}"
    save_path = user_upload_dir / safe_storage_name

    with open(save_path, "wb") as f:
        f.write(file_bytes)

    doc_hash = DocumentLoader.compute_file_hash(str(save_path))
    pages = DocumentLoader.load_document_pages(str(save_path))
    if not pages or not any(p.get("text", "").strip() for p in pages):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not extract readable text from '{filename}'. File may be blank or corrupted."
        )

    doc_id = db.add_document(
        user_id=user_id,
        file_name=filename,
        file_size=file_size,
        file_path=str(save_path),
        chunk_count=0,
        doc_hash=doc_hash,
        category=category
    )

    if not doc_id:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to save document record.")

    splitter = RecursiveTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    doc_type_label = "Image OCR" if ext in config.SUPPORTED_IMAGE_EXTENSIONS else "Document"
    chunks = splitter.split_pages(pages=pages, doc_id=doc_id, user_id=user_id, file_name=filename, file_type=doc_type_label)
    for c in chunks:
        c["category"] = category

    with db.get_connection() as conn:
        conn.cursor().execute("UPDATE documents SET chunk_count = ? WHERE id = ?", (len(chunks), doc_id))
        conn.commit()

    # Rebuild user vector index immediately
    RAGManager.rebuild_user_index(user_id)

    client_ip = request.client.host if request.client else None
    AuditLogger.log(
        action=ACTION_DOC_UPLOAD,
        user_id=user_id,
        username=current_user["username"],
        resource_type="document",
        resource_id=doc_id,
        status="SUCCESS",
        ip_address=client_ip,
        details={"file_name": filename, "file_size": file_size, "category": category, "chunks": len(chunks)}
    )

    return DocumentResponse(
        id=doc_id,
        user_id=user_id,
        file_name=filename,
        file_size=file_size,
        chunk_count=len(chunks),
        category=category,
        effective_permission="owner",
        owner_username=current_user["username"]
    )


@router.get("", response_model=List[DocumentResponse])
async def list_accessible_documents(current_user: Dict[str, Any] = Depends(get_current_user)):
    """List all documents accessible to the authenticated user (owned + shared)."""
    docs = db.get_accessible_documents(current_user["id"])
    return [
        DocumentResponse(
            id=d["id"],
            user_id=d["user_id"],
            file_name=d["file_name"],
            file_size=d["file_size"],
            upload_time=str(d.get("upload_time", "")),
            status=d.get("status", "uploaded"),
            chunk_count=d.get("chunk_count", 0),
            category=d.get("category", "General"),
            effective_permission=d.get("effective_permission", "viewer"),
            owner_username=d.get("owner_username")
        )
        for d in docs
    ]


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document_metadata(
    document_id: int,
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Retrieve metadata of a specific accessible document."""
    has_access = db.can_user_access_document(current_user["id"], document_id, required_level="viewer")
    if not has_access:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied to this document.")

    doc = db.get_document_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document #{document_id} not found.")

    perm = db.get_user_permission_for_document(current_user["id"], document_id)
    client_ip = request.client.host if request.client else None
    AuditLogger.log(
        action=ACTION_DOC_VIEW,
        user_id=current_user["id"],
        username=current_user["username"],
        resource_type="document",
        resource_id=document_id,
        status="SUCCESS",
        ip_address=client_ip
    )

    owner = db.get_user_by_id(doc["user_id"])
    return DocumentResponse(
        id=doc["id"],
        user_id=doc["user_id"],
        file_name=doc["file_name"],
        file_size=doc["file_size"],
        upload_time=str(doc.get("upload_time", "")),
        status=doc.get("status", "uploaded"),
        chunk_count=doc.get("chunk_count", 0),
        category=doc.get("category", "General"),
        effective_permission=perm,
        owner_username=owner.get("username") if owner else None
    )


@router.get("/{document_id}/download")
async def download_document(
    document_id: int,
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Download the original document binary."""
    has_access = db.can_user_access_document(current_user["id"], document_id, required_level="viewer")
    if not has_access:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied to download this document.")

    doc = db.get_document_by_id(document_id)
    if not doc or not os.path.exists(doc["file_path"]):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File binary not found on server.")

    client_ip = request.client.host if request.client else None
    AuditLogger.log(
        action=ACTION_DOC_DOWNLOAD,
        user_id=current_user["id"],
        username=current_user["username"],
        resource_type="document",
        resource_id=document_id,
        status="SUCCESS",
        ip_address=client_ip
    )

    return FileResponse(path=doc["file_path"], filename=doc["file_name"])


@router.delete("/{document_id}", response_model=APIResponse)
async def delete_document(
    document_id: int,
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Delete a document. Restricted to document owner or system administrator."""
    perm = db.get_user_permission_for_document(current_user["id"], document_id)
    if perm != "owner" and current_user.get("role") != getattr(config, "ROLE_ADMIN", "admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Only the document owner or system administrator can delete this document."
        )

    doc = db.get_document_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document #{document_id} not found.")

    # Remove file on disk if exists
    if os.path.exists(doc["file_path"]):
        try:
            os.remove(doc["file_path"])
        except Exception:
            pass

    success = db.delete_document(document_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete document.")

    # Rebuild index for owner
    RAGManager.rebuild_user_index(doc["user_id"])

    client_ip = request.client.host if request.client else None
    AuditLogger.log(
        action=ACTION_DOC_DELETE,
        user_id=current_user["id"],
        username=current_user["username"],
        resource_type="document",
        resource_id=document_id,
        status="SUCCESS",
        ip_address=client_ip,
        details={"file_name": doc["file_name"]}
    )

    return APIResponse(success=True, message=f"Document '{doc['file_name']}' deleted successfully.")


# ==========================================
# Document Permissions Endpoints
# ==========================================

@router.get("/{document_id}/permissions", response_model=List[PermissionResponse])
async def get_document_permissions(
    document_id: int,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """List all explicit user permissions granted for a document."""
    has_access = db.can_user_access_document(current_user["id"], document_id, required_level="viewer")
    if not has_access:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")

    perms = db.get_document_permissions(document_id)
    return [
        PermissionResponse(
            id=p["id"],
            document_id=p["document_id"],
            user_id=p["user_id"],
            user_name=p.get("user_name"),
            permission_level=p["permission_level"],
            granted_by=p.get("granted_by"),
            granted_by_name=p.get("granted_by_name"),
            created_at=str(p.get("created_at", ""))
        )
        for p in perms
    ]


@router.post("/{document_id}/permissions", response_model=APIResponse)
async def grant_permission(
    document_id: int,
    req: PermissionGrantRequest,
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Grant or update document permissions. Owner or Admin only."""
    perm = db.get_user_permission_for_document(current_user["id"], document_id)
    if perm != "owner" and current_user.get("role") != getattr(config, "ROLE_ADMIN", "admin"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only document owner or admin can grant permissions.")

    target_user = db.get_user_by_id(req.target_user_id)
    if not target_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Target user #{req.target_user_id} not found.")

    success = db.grant_document_permission(
        document_id=document_id,
        user_id=req.target_user_id,
        permission_level=req.permission_level,
        granted_by=current_user["id"]
    )
    if not success:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to grant permission.")

    # Rebuild vector index for target user so newly shared document is searchable
    RAGManager.rebuild_user_index(req.target_user_id)

    client_ip = request.client.host if request.client else None
    AuditLogger.log(
        action=ACTION_DOC_PERM_GRANT,
        user_id=current_user["id"],
        username=current_user["username"],
        resource_type="document",
        resource_id=document_id,
        status="SUCCESS",
        ip_address=client_ip,
        details={"grantee_id": req.target_user_id, "level": req.permission_level}
    )

    return APIResponse(success=True, message=f"Permission '{req.permission_level}' granted to user '{target_user['username']}'.")


@router.delete("/{document_id}/permissions/{user_id}", response_model=APIResponse)
async def revoke_permission(
    document_id: int,
    user_id: int,
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Revoke a user's permission on a document. Owner or Admin only."""
    perm = db.get_user_permission_for_document(current_user["id"], document_id)
    if perm != "owner" and current_user.get("role") != getattr(config, "ROLE_ADMIN", "admin"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only document owner or admin can revoke permissions.")

    success = db.revoke_document_permission(document_id, user_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Permission entry not found.")

    # Invalidate index for the revoked user
    RAGManager.rebuild_user_index(user_id)

    client_ip = request.client.host if request.client else None
    AuditLogger.log(
        action=ACTION_DOC_PERM_REVOKE,
        user_id=current_user["id"],
        username=current_user["username"],
        resource_type="document",
        resource_id=document_id,
        status="SUCCESS",
        ip_address=client_ip,
        details={"revoked_user_id": user_id}
    )

    return APIResponse(success=True, message="Document permission successfully revoked.")


@router.post("/{document_id}/transfer-ownership", response_model=APIResponse)
async def transfer_ownership(
    document_id: int,
    req: OwnershipTransferRequest,
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Transfer document primary ownership to another user. Owner or Admin only."""
    perm = db.get_user_permission_for_document(current_user["id"], document_id)
    if perm != "owner" and current_user.get("role") != getattr(config, "ROLE_ADMIN", "admin"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only document owner or admin can transfer ownership.")

    new_owner = db.get_user_by_id(req.new_owner_id)
    if not new_owner:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"New owner #{req.new_owner_id} not found.")

    success = db.transfer_document_ownership(
        document_id=document_id,
        new_owner_id=req.new_owner_id,
        transferred_by=current_user["id"]
    )
    if not success:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to transfer ownership.")

    # Reindex both users
    RAGManager.rebuild_user_index(current_user["id"])
    RAGManager.rebuild_user_index(req.new_owner_id)

    client_ip = request.client.host if request.client else None
    AuditLogger.log(
        action=ACTION_DOC_OWNERSHIP_TRANSFER,
        user_id=current_user["id"],
        username=current_user["username"],
        resource_type="document",
        resource_id=document_id,
        status="SUCCESS",
        ip_address=client_ip,
        details={"new_owner_id": req.new_owner_id, "new_owner": new_owner["username"]}
    )

    return APIResponse(success=True, message=f"Ownership transferred to '{new_owner['username']}'.")
