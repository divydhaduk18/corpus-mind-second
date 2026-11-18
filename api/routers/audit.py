"""Audit Logs REST Endpoints (Improvements 32 & 33)."""

from fastapi import APIRouter, Depends, Query, Response
from typing import List, Dict, Any, Optional
from datetime import datetime

from utils.audit import AuditLogger
from api.schemas import AuditLogResponse
from api.dependencies import require_admin

router = APIRouter(prefix="/api/v1/audit-logs", tags=["Audit Logs (Admin Only)"])


@router.get("", response_model=List[AuditLogResponse])
async def get_audit_logs(
    user_id: Optional[int] = Query(None, description="Filter by user ID"),
    action: Optional[str] = Query(None, description="Filter by action keyword (e.g. AUTH, DOC, USER, RAG)"),
    resource_type: Optional[str] = Query(None, description="user, document, rag, system"),
    status: Optional[str] = Query(None, description="SUCCESS, FAILURE, DENIED"),
    start_date: Optional[str] = Query(None, description="ISO timestamp YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="ISO timestamp YYYY-MM-DD"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    admin_user: Dict[str, Any] = Depends(require_admin)
):
    """Retrieve filtered security audit records with confidential fields automatically redacted."""
    logs = AuditLogger.get_logs(
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        status=status,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset
    )
    return [
        AuditLogResponse(
            id=log["id"],
            timestamp=str(log.get("timestamp", "")),
            user_id=log.get("user_id"),
            username=log.get("username"),
            action=log["action"],
            resource_type=log.get("resource_type", "system"),
            resource_id=log.get("resource_id"),
            status=log.get("status", "SUCCESS"),
            ip_address=log.get("ip_address"),
            details_json=log.get("details_json")
        )
        for log in logs
    ]


@router.get("/export")
async def export_audit_logs_csv(
    user_id: Optional[int] = Query(None),
    action: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(500, ge=1, le=5000),
    admin_user: Dict[str, Any] = Depends(require_admin)
):
    """Export filtered audit trail records to a CSV file."""
    logs = AuditLogger.get_logs(
        user_id=user_id,
        action=action,
        status=status,
        limit=limit
    )
    csv_content = AuditLogger.export_to_csv(logs)
    filename = f"audit_logs_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )
