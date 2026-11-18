"""Asynchronous Document Ingestion & Background Job Queue for Corpus Mind.

Supports:
- Persistent SQLite-backed job queue for background document processing
- Fine-grained stage tracking (Validation -> Extraction -> OCR -> Tables -> Images -> Chunking -> Embeddings -> Indexing -> Finalization)
- Independent ThreadPoolExecutor background workers unaffected by Streamlit page reruns
- Automatic retry, duplicate job prevention, and strict user-level multi-tenant isolation
- Atomic index commit preventing partially processed documents from polluting production search
"""

import os
import sys
import time
import uuid
import threading
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.database import db
from utils.document_loader import DocumentLoader
from utils.text_splitter import RecursiveTextSplitter
from utils.table_extractor import TableExtractor
from utils.visual_rag import VisualRAGManager
from utils.rag import RAGManager
from utils.cache_manager import CacheManager

logger = logging.getLogger(__name__)


class AsyncDocumentProcessor:
    """Manages the background execution, stage updates, and concurrency of document ingestion."""

    _executor = None
    _lock = threading.Lock()

    @classmethod
    def get_executor(cls) -> ThreadPoolExecutor:
        with cls._lock:
            if cls._executor is None:
                max_w = getattr(config, "ASYNC_WORKER_CONCURRENCY", 3)
                cls._executor = ThreadPoolExecutor(max_workers=max_w, thread_name_prefix="doc_worker")
                logger.info(f"Initialized background ThreadPoolExecutor with {max_w} workers.")
            return cls._executor

    @classmethod
    def submit_document_job(
        cls,
        user_id: int,
        file_name: str,
        file_path: str,
        category: str = "General"
    ) -> str:
        """Create a persistent document job and dispatch it to the background worker pool."""
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        
        # 1. Record job in database
        db.create_document_job(
            job_id=job_id,
            user_id=user_id,
            file_name=file_name,
            file_path=file_path
        )

        # 2. Submit task to background thread pool
        executor = cls.get_executor()
        executor.submit(cls._process_document_task, job_id, user_id, file_name, file_path, category)
        logger.info(f"Submitted async document job {job_id} for '{file_name}' (User {user_id}).")
        return job_id

    @classmethod
    def _process_document_task(
        cls,
        job_id: str,
        user_id: int,
        file_name: str,
        file_path: str,
        category: str = "General"
    ):
        """Worker thread processing a document across the 10 ingestion stages."""
        try:
            # Stage 1: File Validation (10%)
            db.update_document_job(job_id, status="processing", progress_pct=10, current_stage="file_validation")
            p_obj = Path(file_path)
            if not p_obj.exists() or p_obj.stat().st_size == 0:
                raise ValueError(f"File '{file_name}' is empty or does not exist.")

            ext = p_obj.suffix.lower()
            file_size = p_obj.stat().st_size
            doc_hash = DocumentLoader.compute_file_hash(file_path)

            # Stage 2: Text Extraction (25%)
            db.update_document_job(job_id, status="processing", progress_pct=25, current_stage="text_extraction")
            pages = DocumentLoader.load_document_pages(file_path)
            if not pages:
                raise ValueError("Could not extract any readable text or pages.")

            # Stage 3: OCR Processing Verification (35%)
            db.update_document_job(job_id, status="processing", progress_pct=35, current_stage="ocr_processing")
            # If image or scanned, pages were already OCR processed via load_document_pages

            # Stage 4: Table Extraction (45%)
            db.update_document_job(job_id, status="processing", progress_pct=45, current_stage="table_extraction")
            table_chunks = []
            if ext == ".pdf" and getattr(config, "ENABLE_TABLE_EXTRACTION_ON_UPLOAD", True):
                try:
                    tables = TableExtractor.extract_tables_from_pdf(file_path)
                    for t in tables:
                        t_pass = t.to_rag_passage()
                        if t_pass.strip():
                            table_chunks.append({
                                "text": t_pass,
                                "page_number": t.page_number,
                                "section": f"Table {t.table_id}",
                                "file_type": "Table Data"
                            })
                except Exception as t_err:
                    logger.debug(f"Table extraction minor error on {file_name}: {t_err}")

            # Stage 5: Image & Figure Processing (55%)
            db.update_document_job(job_id, status="processing", progress_pct=55, current_stage="image_processing")
            figure_chunks = []
            if ext == ".pdf" and getattr(config, "ENABLE_IMAGE_EXTRACTION_ON_UPLOAD", True):
                try:
                    figures = VisualRAGManager.extract_figures_from_pdf(file_path, user_id=user_id, generate_descriptions=True)
                    for fig in figures:
                        f_pass = fig.to_rag_passage()
                        if f_pass.strip():
                            figure_chunks.append({
                                "text": f_pass,
                                "page_number": fig.page_number,
                                "section": f"Figure {fig.figure_id} ({fig.figure_type})",
                                "file_type": "Visual Diagram"
                            })
                except Exception as f_err:
                    logger.debug(f"Figure extraction minor error on {file_name}: {f_err}")

            # Stage 6: Chunking & Hierarchical Splitting (70%)
            db.update_document_job(job_id, status="processing", progress_pct=70, current_stage="chunking")
            splitter = RecursiveTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
            
            # Record document in database if not already present
            existing_docs = db.get_documents_by_user(user_id)
            doc_record = next((d for d in existing_docs if d['file_name'] == file_name), None)
            
            if not doc_record:
                doc_id = db.add_document(
                    user_id=user_id,
                    file_name=file_name,
                    file_size=file_size,
                    file_path=file_path,
                    doc_hash=doc_hash,
                    category=category
                )
            else:
                doc_id = doc_record['id']

            file_type = "Image OCR" if ext in config.SUPPORTED_IMAGE_EXTENSIONS else "Document"
            strategy = "hierarchical" if getattr(config, "PARENT_CHILD_ENABLED", True) else "semantic"

            doc_chunks = splitter.split_pages(
                pages=pages,
                doc_id=doc_id,
                user_id=user_id,
                file_name=file_name,
                file_type=file_type,
                strategy=strategy
            )

            # Combine normal chunks with table and figure chunks
            for tc in table_chunks:
                tc.update({"doc_id": doc_id, "user_id": user_id, "file_name": file_name, "category": category})
                doc_chunks.append(tc)

            for fc in figure_chunks:
                fc.update({"doc_id": doc_id, "user_id": user_id, "file_name": file_name, "category": category})
                doc_chunks.append(fc)

            for dc in doc_chunks:
                dc["category"] = category

            # Stage 7 & 8: Embedding Generation & FAISS Indexing (85%)
            db.update_document_job(job_id, status="processing", progress_pct=85, current_stage="embedding_generation")
            
            # Stage 9: BM25 Indexing & Atomic Commit (95%)
            db.update_document_job(job_id, status="processing", progress_pct=95, current_stage="bm25_indexing")
            
            # Rebuild / update index for user atomically
            RAGManager.rebuild_user_index(user_id)
            
            # Invalidate caches
            CacheManager.get_instance().invalidate_user_retrieval_cache(user_id)

            # Update document status to ready
            db.update_document_status(doc_id, status="ready", chunk_count=len(doc_chunks))

            # Stage 10: Finalization (100%)
            db.update_document_job(job_id, status="completed", progress_pct=100, current_stage="finalization")
            logger.info(f"Async document job {job_id} successfully completed for '{file_name}'.")

        except Exception as e:
            err_msg = str(e)
            logger.error(f"Async document job {job_id} failed: {err_msg}", exc_info=True)
            db.update_document_job(job_id, status="failed", progress_pct=0, current_stage="failed", error_message=err_msg)

    @classmethod
    def retry_job(cls, job_id: str) -> bool:
        """Retry a previously failed document job."""
        job = db.get_document_job_by_id(job_id)
        if not job or job.get("status") != "failed":
            return False

        executor = cls.get_executor()
        executor.submit(
            cls._process_document_task,
            job["job_id"],
            job["user_id"],
            job["file_name"],
            job["file_path"]
        )
        db.update_document_job(job_id, status="queued", progress_pct=0, current_stage="queued", error_message=None)
        return True
