"""RAG, Search & Multi-Provider LLM REST Endpoints (Improvements 32)."""

import time
import json
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import StreamingResponse

import config
from utils.database import db
from utils.rag import RAGManager
from utils.llm_provider import LLMProviderManager
from utils.hallucination_detector import HallucinationDetector
from utils.document_summarizer import DocumentSummarizer
from utils.audit import AuditLogger, ACTION_RAG_SEARCH, ACTION_RAG_QUERY, ACTION_RAG_SUMMARIZE
from api.schemas import (
    SearchRequest, SearchChunkResponse, ChatRequest, ChatResponse,
    DocumentSummarizeRequest, DocumentSummarizeResponse
)
from api.dependencies import get_current_user, check_rate_limit

router = APIRouter(prefix="/api/v1", tags=["RAG & Intelligence"])


@router.post("/search", response_model=List[SearchChunkResponse], dependencies=[Depends(check_rate_limit)])
async def search_rag_context(
    req: SearchRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Hybrid RAG context search strictly filtered by user document permissions."""
    chunks = RAGManager.retrieve_context(
        query=req.query,
        user_id=current_user["id"],
        top_k=req.top_k or 4,
        search_mode=req.search_mode or "hybrid",
        doc_filter=req.doc_filter,
        metadata_filters=req.metadata_filters,
        enable_reranking=req.enable_reranking
    )
    return [
        SearchChunkResponse(
            text=c.get("text", ""),
            doc_id=c.get("doc_id"),
            file_name=c.get("file_name"),
            page_number=c.get("page_number", 1),
            score=round(float(c.get("score", 0.0)), 4),
            search_mode=c.get("search_mode"),
            file_type=c.get("file_type"),
            category=c.get("category")
        )
        for c in chunks
    ]


@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(check_rate_limit)])
async def chat_rag(
    req: ChatRequest,
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Conversational RAG generation with context retrieval, citations, and hallucination scoring."""
    start_t = time.perf_counter()
    user_id = current_user["id"]

    # 1. Retrieve permitted context chunks
    retrieved_chunks = RAGManager.retrieve_context(
        query=req.message,
        user_id=user_id,
        top_k=req.top_k or 4,
        search_mode=req.search_mode or "hybrid",
        doc_filter=req.doc_filter,
        metadata_filters=req.metadata_filters
    )

    # 2. Build conversation session
    session_id = req.session_id
    if not session_id:
        session_id = db.create_chat_session(user_id=user_id, title=req.message[:50])

    # Record user message in DB
    db.add_message(session_id=session_id, role="user", content=req.message)

    # 3. Fetch session history for context-aware conversation
    past_messages = db.get_messages_by_session(session_id)
    history_formatted = []
    for m in past_messages[-6:]:
        history_formatted.append({"role": m["role"], "content": m["content"]})

    # 4. Construct prompt with retrieved evidence
    context_str = "\n\n".join([
        f"[{c.get('file_name', 'Doc')}, Page {c.get('page_number', 1)}]: {c.get('text', '')}"
        for c in retrieved_chunks
    ])
    system_instruction = (
        "You are Corpus Mind, a verified enterprise document intelligence assistant. "
        "Answer the user's question accurately using ONLY the context provided below. "
        "Cite sources using [file_name, Page X] notation.\n\n"
        f"Retrieved Context:\n{context_str}"
    )

    messages = [{"role": "system", "content": system_instruction}] + history_formatted

    # 5. Model Generation
    provider = req.provider or getattr(config, "DEFAULT_LLM_PROVIDER", "ollama")
    model = req.model or LLMProviderManager.get_models_for_provider(provider)[0]

    gen_resp = LLMProviderManager.generate(
        provider=provider,
        model=model,
        messages=messages,
        settings={"temperature": req.temperature or 0.2, "max_tokens": req.max_tokens or 1000}
    )

    duration_ms = round((time.perf_counter() - start_t) * 1000.0, 2)
    answer_text = gen_resp.text if not gen_resp.error else f"Generation error: {gen_resp.error}"

    # 6. Record assistant message in DB
    db.add_message(session_id=session_id, role="assistant", content=answer_text, model=model, response_time_ms=int(duration_ms))

    # 7. Hallucination Detection & Faithfulness Scoring
    h_result = HallucinationDetector.verify_answer(
        query=req.message,
        answer=answer_text,
        retrieved_chunks=retrieved_chunks
    )

    # 8. Audit Logging
    client_ip = request.client.host if request.client else None
    AuditLogger.log(
        action=ACTION_RAG_QUERY,
        user_id=user_id,
        username=current_user["username"],
        resource_type="rag",
        resource_id=session_id,
        status="SUCCESS",
        ip_address=client_ip,
        details={
            "query": req.message[:120],
            "provider": provider,
            "model": model,
            "chunks_used": len(retrieved_chunks),
            "latency_ms": duration_ms,
            "faithfulness": h_result.get("faithfulness_score")
        }
    )

    sources = [
        {
            "file_name": c.get("file_name"),
            "page_number": c.get("page_number", 1),
            "score": round(float(c.get("score", 0.0)), 4),
            "text": c.get("text", "")[:200] + "..."
        }
        for c in retrieved_chunks
    ]

    return ChatResponse(
        session_id=session_id,
        answer=answer_text,
        model=model,
        provider=provider,
        sources=sources,
        faithfulness_score=h_result.get("faithfulness_score"),
        latency_ms=duration_ms
    )


@router.post("/chat/stream")
async def chat_rag_stream(
    req: ChatRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Streaming conversational RAG endpoint returning Server-Sent Events (SSE)."""
    user_id = current_user["id"]
    retrieved_chunks = RAGManager.retrieve_context(
        query=req.message,
        user_id=user_id,
        top_k=req.top_k or 4,
        search_mode=req.search_mode or "hybrid"
    )

    context_str = "\n\n".join([
        f"[{c.get('file_name', 'Doc')}, Page {c.get('page_number', 1)}]: {c.get('text', '')}"
        for c in retrieved_chunks
    ])
    messages = [
        {"role": "system", "content": f"Answer based on context:\n{context_str}"},
        {"role": "user", "content": req.message}
    ]

    provider = req.provider or getattr(config, "DEFAULT_LLM_PROVIDER", "ollama")
    model = req.model or LLMProviderManager.get_models_for_provider(provider)[0]

    def event_stream():
        # First send sources metadata
        sources_payload = {
            "type": "sources",
            "sources": [
                {"file_name": c.get("file_name"), "page": c.get("page_number", 1)}
                for c in retrieved_chunks
            ]
        }
        yield f"data: {json.dumps(sources_payload)}\n\n"

        # Stream tokens
        for token in LLMProviderManager.generate_stream(
            provider=provider,
            model=model,
            messages=messages,
            settings={"temperature": req.temperature or 0.2, "max_tokens": req.max_tokens or 1000}
        ):
            chunk_payload = {"type": "token", "content": token}
            yield f"data: {json.dumps(chunk_payload)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/documents/{document_id}/summarize", response_model=DocumentSummarizeResponse, dependencies=[Depends(check_rate_limit)])
async def summarize_document(
    document_id: int,
    req: DocumentSummarizeRequest,
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Generate structured multi-perspective summaries for an accessible document."""
    has_access = db.can_user_access_document(current_user["id"], document_id, required_level="viewer")
    if not has_access:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")

    doc = db.get_document_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    start_t = time.perf_counter()
    summary = DocumentSummarizer.summarize_document(
        file_path=doc["file_path"],
        summary_type=req.summary_type or "executive",
        provider_name=req.provider,
        model_name=req.model
    )
    lat_ms = round((time.perf_counter() - start_t) * 1000.0, 2)

    client_ip = request.client.host if request.client else None
    AuditLogger.log(
        action=ACTION_RAG_SUMMARIZE,
        user_id=current_user["id"],
        username=current_user["username"],
        resource_type="document",
        resource_id=document_id,
        status="SUCCESS",
        ip_address=client_ip,
        details={"summary_type": req.summary_type, "latency_ms": lat_ms}
    )

    return DocumentSummarizeResponse(
        document_id=document_id,
        file_name=doc["file_name"],
        summary_type=req.summary_type or "executive",
        summary=summary,
        latency_ms=lat_ms
    )
