import streamlit as st
import os
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.auth import AuthManager
from utils.database import db
from utils.document_loader import DocumentLoader
from utils.text_splitter import RecursiveTextSplitter
from utils.rag import RAGManager
from utils.ocr import OCRManager
from utils.audit import (
    AuditLogger, ACTION_DOC_UPLOAD, ACTION_DOC_VIEW, ACTION_DOC_DELETE,
    ACTION_DOC_PERM_GRANT, ACTION_DOC_PERM_REVOKE, ACTION_DOC_OWNERSHIP_TRANSFER
)

# Guard Page Access
AuthManager.require_auth()

user_id = st.session_state.user_id
user_role = st.session_state.get("user_role", config.ROLE_EDITOR)
can_upload = AuthManager.can_upload_document(user_role)

# Page Styling
css_path = config.CSS_DIR / "style.css"
if css_path.exists():
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Custom State variables
if "ocr_text" not in st.session_state:
    st.session_state.ocr_text = ""
if "ocr_filename" not in st.session_state:
    st.session_state.ocr_filename = ""

is_admin = st.session_state.get("user_role") == "admin"
role_badge = "👑 ADMIN" if is_admin else "👤 USER"
role_bg = "#ef4444" if is_admin else "#3b82f6"

# Sidebar layout
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/robot-3.png", width=80)
    st.markdown(f"User: **{st.session_state.username}**")
    st.markdown(
        f'<span style="background-color:{role_bg}; color:white; padding:3px 8px; border-radius:12px; font-weight:bold; font-size:11px;">{role_badge}</span>',
        unsafe_allow_html=True
    )
    if is_admin:
        st.markdown("")
        if st.button("🛡️ Admin Panel", use_container_width=True):
            st.switch_page("pages/Admin.py")
    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        AuthManager.logout_user()
        st.rerun()

st.title("📂 Multi-Modal Document & OCR RAG Portal")

tab_upload, tab_manage, tab_ocr, tab_search = st.tabs([
    "📥 Upload Documents & Images", 
    "🗃️ Manage RAG Files", 
    "📷 Image OCR & Instant Index", 
    "🔍 Hybrid RAG Context Search"
])

# -----------------
# TAB 1: UPLOAD DOCUMENTS & IMAGES FOR RAG
# -----------------
with tab_upload:
    st.header("Upload Files for RAG Indexing")
    if not can_upload:
        st.warning(
            "🔒 **Upload Restricted:** Your account has the **Viewer** role. "
            "Viewers have read-only access and cannot upload documents. "
            "Please contact an Administrator to request **Editor** privileges."
        )
    else:
        st.markdown(
            "Upload document files (**PDF, DOCX, TXT, MD, CSV, JSON**) or image files (**PNG, JPG, JPEG, WEBP, BMP, TIFF**). "
            "Text is extracted automatically via high-speed native extractors (**PyMuPDF**) or **Tesseract OCR / RapidOCR**, "
            "split into semantic chunks with page citations, and indexed into your private vector store."
        )
    
    supported_types = ["pdf", "docx", "txt", "md", "csv", "json", "png", "jpg", "jpeg", "webp", "bmp", "tiff"]
    uploaded_files = st.file_uploader(
        "Choose files (Documents or Images)", 
        type=supported_types, 
        accept_multiple_files=True
    )
    
    doc_category = st.selectbox(
        "📁 Document Category (for metadata filtering)",
        options=["General", "Financial", "Technical", "Legal", "HR", "Medical"],
        index=0,
        help="Tag your uploaded documents with a category to enable filtered retrieval."
    )
    
    async_mode = st.checkbox(
        "⚡ Process Asynchronously in Background (Non-blocking)",
        value=False,
        help="Dispatches ingestion to background workers so you can continue using the application."
    )

    if st.button("🚀 Process & Index Files for RAG", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one document or image file first.")
        elif async_mode:
            from utils.async_processor import AsyncDocumentProcessor
            user_upload_dir = config.get_user_uploads_dir(user_id)
            queued_jobs = []
            for uploaded_file in uploaded_files:
                file_name = uploaded_file.name
                file_bytes = uploaded_file.getvalue()
                safe_storage_name = f"{int(time.time() * 1000)}_{Path(file_name).name}"
                save_path = user_upload_dir / safe_storage_name
                with open(save_path, "wb") as f:
                    f.write(file_bytes)
                
                j_id = AsyncDocumentProcessor.submit_document_job(
                    user_id=user_id,
                    file_name=file_name,
                    file_path=str(save_path),
                    category=doc_category
                )
                queued_jobs.append(j_id)
            
            st.success(f"🚀 Submitted {len(queued_jobs)} document(s) to the background worker queue!")
            st.info("You can monitor live progress and ingestion stages under **Performance & Evaluation > Async Job Queue**.")
            time.sleep(1.5)
            st.rerun()
        else:
            success_count = 0
            user_upload_dir = config.get_user_uploads_dir(user_id)
            splitter = RecursiveTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
            
            with st.status("Processing uploaded files...", expanded=True) as status:
                for uploaded_file in uploaded_files:
                    file_name = uploaded_file.name
                    file_bytes = uploaded_file.getvalue()
                    file_size = len(file_bytes)
                    ext = Path(file_name).suffix.lower()
                    
                    st.write(f"🔄 Processing `{file_name}`...")
                    
                    # Prevent file collisions using unique timestamp prefix while keeping human-readable display name
                    safe_storage_name = f"{int(time.time() * 1000)}_{Path(file_name).name}"
                    save_path = user_upload_dir / safe_storage_name

                    try:
                        with open(save_path, "wb") as f:
                            f.write(file_bytes)
                        
                        doc_hash = DocumentLoader.compute_file_hash(str(save_path))
                        
                        # Extract structured page-level text
                        pages = DocumentLoader.load_document_pages(str(save_path))
                        has_text = any(p.get("text", "").strip() for p in pages)

                        if not has_text:
                            if ext in config.SUPPORTED_IMAGE_EXTENSIONS or ext == ".pdf":
                                if not OCRManager.is_tesseract_available() and not OCRManager.is_rapidocr_available():
                                    st.warning(
                                        f"⚠️ Could not extract text from `{file_name}` because no local OCR engine was found.\n\n"
                                        "Please configure Tesseract OCR in Settings."
                                    )
                                else:
                                    st.error(f"Could not extract readable text from `{file_name}`. File may be blank or unreadable.")
                            else:
                                st.error(f"Could not extract text from `{file_name}`. Empty content or unsupported layout.")
                            continue

                        doc_type_label = "Image OCR" if ext in config.SUPPORTED_IMAGE_EXTENSIONS else "Document"

                        # Save document record to database with user_id, doc_hash, and category
                        doc_id = db.add_document(
                            user_id=user_id,
                            file_name=file_name,
                            file_size=file_size,
                            file_path=str(save_path),
                            chunk_count=0,
                            doc_hash=doc_hash,
                            category=doc_category
                        )

                        if doc_id:
                            # Split pages preserving page numbers and document ID
                            chunks = splitter.split_pages(
                                pages=pages,
                                doc_id=doc_id,
                                user_id=user_id,
                                file_name=file_name,
                                file_type=doc_type_label
                            )
                            for c in chunks:
                                c["category"] = doc_category

                            # Update chunk count
                            with db.get_connection() as conn:
                                conn.cursor().execute("UPDATE documents SET chunk_count = ? WHERE id = ?", (len(chunks), doc_id))
                                conn.commit()

                            success_count += 1
                            st.write(f"✅ Extracted `{file_name}` -> {len(pages)} pages, {len(chunks)} text chunks")
                            db.log_analytics(user_id, "document_upload", f"Uploaded and indexed {file_name}")

                    except Exception as e:
                        st.error(f"Error processing file {file_name}: {e}")
                        
                if success_count > 0:
                    status.update(label="Rebuilding your private Hybrid RAG Index...", state="running")
                    success, docs_c, chunks_c = RAGManager.rebuild_user_index(user_id)
                    if success:
                        status.update(label=f"Successfully indexed {success_count} files ({chunks_c} chunks total)!", state="complete")
                        st.success(f"Indexed {success_count} files into {chunks_c} vector chunks!")
                        st.rerun()
                    else:
                        st.error("Failed to compile vector index.")

# -----------------
# TAB 2: MANAGE RAG FILES & STATS
# -----------------
with tab_manage:
    st.header("Indexed Documents & Images")
    
    # Retrieve all documents accessible to this authenticated user (owned + shared)
    docs = db.get_accessible_documents(user_id)
    
    # Format counts
    total_size = sum(doc['file_size'] for doc in docs) / (1024.0 * 1024.0) if docs else 0.0
    total_chunks = sum(doc['chunk_count'] for doc in docs) if docs else 0
    img_count = sum(1 for doc in docs if Path(doc['file_name']).suffix.lower() in config.SUPPORTED_IMAGE_EXTENSIONS)
    doc_count = len(docs) - img_count

    st.markdown(
        f"""
        <div class="stat-container" style="margin-bottom: 20px;">
            <div class="stat-card">
                <h4>Accessible Files</h4>
                <div class="stat-value">{len(docs)}</div>
                <small style="color:#9ca3af;">{doc_count} Docs | {img_count} OCR Images</small>
            </div>
            <div class="stat-card">
                <h4>Indexed Chunks</h4>
                <div class="stat-value">{total_chunks}</div>
                <small style="color:#9ca3af;">FAISS & BM25 Tokens</small>
            </div>
            <div class="stat-card">
                <h4>Storage Footprint</h4>
                <div class="stat-value">{total_size:.2f} MB</div>
                <small style="color:#9ca3af;">Accessible Knowledge Base</small>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if st.button("🔄 Rebuild My Hybrid RAG Index", use_container_width=True):
        with st.spinner("Reindexing your accessible documents..."):
            success, docs_c, chunks_c = RAGManager.rebuild_user_index(user_id)
            if success:
                st.success(f"Index rebuilt successfully! Indexed {docs_c} files with {chunks_c} total chunks.")
                st.rerun()
            else:
                st.error("Failed to rebuild index.")
                
    st.markdown("---")
    
    if docs:
        all_sys_users = db.get_all_users()
        for doc in docs:
            ext = Path(doc['file_name']).suffix.lower()
            icon = "🖼️" if ext in config.SUPPORTED_IMAGE_EXTENSIONS else ("📊" if ext == ".csv" else "📄")
            eff_perm = doc.get("effective_permission", "owner")
            
            # Permission Badge
            if eff_perm == "owner":
                perm_badge = "👑 Owner"
                perm_color = "#10b981"
            elif eff_perm == "editor":
                perm_badge = "✏️ Editor"
                perm_color = "#3b82f6"
            else:
                perm_badge = "👁️ Viewer"
                perm_color = "#6b7280"

            with st.container():
                col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
                with col1:
                    owner_info = f" (Owner: {doc.get('owner_username', 'You')})" if doc.get('owner_username') and doc.get('user_id') != user_id else ""
                    st.write(f"{icon} **{doc['file_name']}**{owner_info}")
                with col2:
                    st.markdown(
                        f'<span style="background-color:{perm_color}; color:white; padding:2px 8px; border-radius:10px; font-size:11px; font-weight:bold;">{perm_badge}</span>',
                        unsafe_allow_html=True
                    )
                with col3:
                    st.write(f"💾 {doc['file_size']/1024.0:.1f} KB")
                with col4:
                    st.write(f"🧩 {doc['chunk_count']} chunks")
                with col5:
                    can_delete = (eff_perm == "owner" or is_admin)
                    if can_delete:
                        if st.button("🗑️ Delete", key=f"del_{doc['id']}", use_container_width=True):
                            deleted = db.delete_document(doc['id'], user_id=user_id)
                            if deleted:
                                AuditLogger.log(
                                    action=ACTION_DOC_DELETE,
                                    user_id=user_id,
                                    username=st.session_state.username,
                                    resource_type="document",
                                    resource_id=doc['id'],
                                    status="SUCCESS",
                                    details={"file_name": doc['file_name']}
                                )
                                with st.spinner("Updating Vector Store Index..."):
                                    RAGManager.rebuild_user_index(user_id)
                                st.rerun()
                            else:
                                st.error("Could not delete file.")
                    else:
                        st.caption("🔒 Read-Only")

                # Document sharing controls for owners and administrators
                if eff_perm == "owner" or is_admin:
                    with st.expander(f"🤝 Share & Permissions: {doc['file_name']}"):
                        current_p = db.get_document_permissions(doc['id'])
                        if current_p:
                            st.write("**Currently Shared With:**")
                            for p in current_p:
                                st.write(f"- **{p['user_name']}**: `{p['permission_level']}` (granted {p['created_at'][:10]})")
                        else:
                            st.caption("This document is private to you.")
                            
                        col_s1, col_s2, col_s3 = st.columns([2, 1, 1])
                        with col_s1:
                            target_share_user = st.selectbox(
                                "Share with User",
                                options=[u['username'] for u in all_sys_users if u['id'] != user_id],
                                key=f"share_u_{doc['id']}"
                            )
                        with col_s2:
                            share_lvl = st.selectbox(
                                "Access Level",
                                options=["viewer", "editor", "none"],
                                key=f"share_lvl_{doc['id']}"
                            )
                        with col_s3:
                            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                            if st.button("Update Share", key=f"btn_share_{doc['id']}", use_container_width=True):
                                tu = db.get_user_by_username(target_share_user)
                                if tu:
                                    db.grant_document_permission(doc['id'], tu['id'], share_lvl, granted_by=user_id)
                                    AuditLogger.log(
                                        action=ACTION_DOC_PERM_GRANT if share_lvl != "none" else ACTION_DOC_PERM_REVOKE,
                                        user_id=user_id,
                                        username=st.session_state.username,
                                        resource_type="document",
                                        resource_id=doc['id'],
                                        status="SUCCESS",
                                        details={"target_user": target_share_user, "permission": share_lvl}
                                    )
                                    # Trigger reindexing for the target user so shared doc is available immediately
                                    RAGManager.rebuild_user_index(tu['id'])
                                    st.success(f"Updated access for {target_share_user} to '{share_lvl}'.")
                                    st.rerun()

                with st.expander(f"👁️ View Extracted Text Preview: {doc['file_name']}"):
                    if os.path.exists(doc['file_path']):
                        preview_text = DocumentLoader.load_document(doc['file_path'])
                        st.text_area("Extracted Content", preview_text[:1500] + ("..." if len(preview_text) > 1500 else ""), height=150, key=f"prev_{doc['id']}")
                        AuditLogger.log(
                            action=ACTION_DOC_VIEW,
                            user_id=user_id,
                            username=st.session_state.username,
                            resource_type="document",
                            resource_id=doc['id'],
                            status="SUCCESS"
                        )
                    else:
                        st.warning("File binary missing on disk.")
    else:
        st.info("No documents or images uploaded yet.")

# -----------------
# TAB 3: IMAGE OCR & INSTANT RAG INDEXING
# -----------------
with tab_ocr:
    st.header("Extract & Index Text from Images")
    if not can_upload:
        st.warning(
            "🔒 **OCR Ingestion Restricted:** Your account has the **Viewer** role. "
            "Viewers have read-only access and cannot index new images into the knowledge base. "
            "Please contact an Administrator to request **Editor** privileges."
        )
    else:
        st.markdown("Upload any image (PNG, JPG, JPEG, WEBP, BMP, TIFF) to run multi-pass OCR with OpenCV CLAHE preprocessing and confidence evaluation.")
        
        ocr_file = st.file_uploader("Upload Image File for OCR", type=["png", "jpg", "jpeg", "webp", "bmp", "tiff"], key="ocr_tab_uploader")
    
    if ocr_file:
        st.image(ocr_file, caption=f"Uploaded Image: {ocr_file.name}", use_container_width=True)
        st.session_state.ocr_filename = ocr_file.name
        
        if st.button("📷 Perform OCR Extraction", type="primary", use_container_width=True):
            with st.spinner("Running Multi-Pass OCR Engine..."):
                extracted_text = OCRManager.extract_text_from_bytes(ocr_file)
                st.session_state.ocr_text = extracted_text
                db.log_analytics(user_id, "ocr_extraction", f"Extracted text from {ocr_file.name}")
                
    if st.session_state.ocr_text:
        st.subheader("Extracted OCR Text Content:")
        st.code(st.session_state.ocr_text, language="text")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📥 Save & Index to RAG Store", type="primary", use_container_width=True):
                user_upload_dir = config.get_user_uploads_dir(user_id)
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                orig_stem = Path(st.session_state.ocr_filename or "image").stem
                save_filename = f"ocr_{orig_stem}_{timestamp_str}.txt"
                save_path = user_upload_dir / save_filename
                
                header_text = f"--- OCR Extracted Text from Image: {st.session_state.ocr_filename} ---\n\n" + st.session_state.ocr_text
                
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(header_text)
                    
                file_size = len(header_text.encode('utf-8'))
                doc_hash = DocumentLoader.compute_file_hash(str(save_path))
                
                doc_id = db.add_document(
                    user_id=user_id,
                    file_name=save_filename,
                    file_size=file_size,
                    file_path=str(save_path),
                    chunk_count=0,
                    doc_hash=doc_hash
                )
                
                if doc_id:
                    splitter = RecursiveTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
                    chunks = splitter.split_text(header_text)
                    with db.get_connection() as conn:
                        conn.cursor().execute("UPDATE documents SET chunk_count = ? WHERE id = ?", (len(chunks), doc_id))
                        conn.commit()

                    RAGManager.rebuild_user_index(user_id)
                    st.success(f"Saved OCR text as `{save_filename}` and indexed into RAG store ({len(chunks)} chunks)!")
                    db.log_analytics(user_id, "ocr_rag_indexed", f"Indexed OCR image {save_filename}")

        with col2:
            if st.button("✅ Send to Chat Context", use_container_width=True):
                st.success("Extracted text is active for current Chat session!")

        with col3:
            if st.button("🧹 Clear OCR Text", use_container_width=True):
                st.session_state.ocr_text = ""
                st.session_state.ocr_filename = ""
                st.rerun()

# -----------------
# TAB 4: HYBRID RAG CONTEXT SEARCH
# -----------------
with tab_search:
    st.header("Hybrid RAG Search Tester")
    st.markdown("Test search retrieval on your indexed documents and OCR images. Compare **Hybrid (Vector + BM25 RRF)**, **Dense Vector**, and **Keyword BM25** search modes.")
    
    search_query = st.text_input("Enter search query or question")
    
    engine_col, topk_col, thresh_col = st.columns([2, 1, 1])
    with engine_col:
        search_engine = st.radio(
            "Retrieval Engine Mode",
            ["Hybrid (Dense + BM25 RRF)", "Dense Vector Search", "Keyword BM25 Search"],
            horizontal=True
        )
    with topk_col:
        top_k = st.slider("Top K Results", min_value=1, max_value=10, value=4)
    with thresh_col:
        min_score = st.slider("Min Score Threshold", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
        
    mode_map = {
        "Hybrid (Dense + BM25 RRF)": "hybrid",
        "Dense Vector Search": "dense",
        "Keyword BM25 Search": "bm25"
    }
    selected_mode = mode_map[search_engine]

    # Target Files Multiselect
    user_docs = db.get_documents_by_user(user_id)
    all_file_names = [d['file_name'] for d in user_docs] if user_docs else []
    selected_test_files = st.multiselect(
        "Filter Search to Specific Files (Optional)",
        options=all_file_names,
        default=[],
        help="Select specific files to test search retrieval. Leave blank to search all files."
    )
    
    if search_query:
        with st.spinner(f"Retrieving context using {search_engine}..."):
            results = RAGManager.retrieve_context(
                query=search_query,
                user_id=user_id,
                top_k=top_k,
                similarity_threshold=min_score,
                search_mode=selected_mode,
                doc_filter=selected_test_files if selected_test_files else None
            )

            if results:
                st.success(f"Retrieved {len(results)} matching chunks using {search_engine}:")
                for i, res in enumerate(results, 1):
                    file_type = res.get('file_type', 'Document')
                    page_num = res.get('page_number', 1)
                    score_label = f"RRF Rank Score: {res['score']:.4f}" if selected_mode == "hybrid" else f"Similarity Score: {res['score']:.4f}"
                    
                    st.markdown(
                        f"""
                        <div class="glass-card" style="margin-top:10px;">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <strong>Match #{i} | <code>{res['file_name']}</code> (p. {page_num}, {file_type})</strong>
                                <span style="background:#10b981; color:white; padding:2px 8px; border-radius:12px; font-size:12px;">{score_label}</span>
                            </div>
                            <div style="background-color:rgba(0,0,0,0.15); margin-top:8px; padding:10px; border-radius:6px; font-family:monospace; white-space:pre-wrap;">{res['text']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.warning("No context chunks match the search criteria. Try adjusting query terms or lower the score threshold.")
