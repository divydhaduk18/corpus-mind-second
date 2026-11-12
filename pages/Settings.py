import streamlit as st
import os
import sys
import requests
import dotenv
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.auth import AuthManager
from utils.ocr import OCRManager
from utils.database import db

# Guard Page Access
AuthManager.require_auth()

# Page Styling
css_path = config.CSS_DIR / "style.css"
if css_path.exists():
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

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

st.title("⚙️ Application Settings")

st.markdown("Configure local directories, network endpoints, visual styles, and security parameters.")

# Theme and style
st.subheader("🎨 User Interface Preferences")
theme_select = st.selectbox(
    "Theme Selector", 
    options=["Dark Mode (Default)", "Light Mode"],
    index=0 if st.session_state.theme == "dark" else 1
)
new_theme = "dark" if "Dark Mode" in theme_select else "light"
if new_theme != st.session_state.theme:
    st.session_state.theme = new_theme
    st.success(f"Theme updated to {new_theme}! Reloading...")
    st.rerun()

font_size = st.slider("Font Size Adjustment (px)", min_value=12, max_value=20, value=15, step=1)

# Check Admin Role
is_admin = st.session_state.get("user_role") == "admin"

if not is_admin:
    st.info("🔒 **Notice**: System binary paths, Ollama endpoints, and storage management are configured by your System Administrator.")
else:
    # ---------------- ADMIN SYSTEM CONFIGURATIONS ----------------
    st.markdown("---")
    st.subheader("🛠️ Administrator System Configurations")
    
    tesseract_path = st.text_input(
        "Tesseract Binary Executable Path",
        value=os.environ.get("TESSERACT_CMD_PATH", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    )
    
    ocr_available = OCRManager.is_tesseract_available()
    rapid_available = OCRManager.is_rapidocr_available()
    if ocr_available:
        st.success("✅ Tesseract OCR binary verified successfully.")
    else:
        st.warning("⚠️ Tesseract OCR binary not found at specified path.")

    if rapid_available:
        st.info("✅ RapidOCR (ONNXRuntime) is also installed and ready as an alternative engine.")

    # OCR Engine and Language Selection
    ocr_col1, ocr_col2 = st.columns(2)
    with ocr_col1:
        ocr_engine_opt = st.selectbox(
            "Default OCR Engine",
            options=["tesseract", "rapidocr"],
            index=0 if getattr(config, "DEFAULT_OCR_ENGINE", "tesseract") == "tesseract" else 1
        )
    with ocr_col2:
        available_langs = OCRManager.get_available_languages()
        ocr_lang_opt = st.selectbox(
            "OCR Recognition Language",
            options=available_langs if available_langs else ["eng"],
            index=0
        )

    # ---------------- ADVANCED RAG ARCHITECTURE ----------------
    st.markdown("---")
    st.subheader("🧠 Advanced RAG & Retrieval Engine (10 Enhancements)")
    
    # 1. Models & Reranker
    rag_col1, rag_col2 = st.columns(2)
    with rag_col1:
        embedding_model_options = ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]
        current_emb = getattr(config, "DEFAULT_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        emb_idx = 0 if current_emb == "all-MiniLM-L6-v2" else 1
        selected_emb_model = st.selectbox(
            "Default Embedding Model",
            options=embedding_model_options,
            index=emb_idx,
            help="all-MiniLM-L6-v2 runs 15x faster with 8x less RAM on CPU. all-mpnet-base-v2 outputs 768d vectors."
        )
    with rag_col2:
        reranker_options = [
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "cross-encoder/ms-marco-TinyBERT-L-2-v2"
        ]
        curr_reranker = getattr(config, "DEFAULT_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        rerank_idx = 0 if curr_reranker == reranker_options[0] else 1
        selected_reranker = st.selectbox(
            "Cross-Encoder Reranker Model",
            options=reranker_options,
            index=rerank_idx,
            help="Deep cross-attention model that joint-scores query and passages."
        )

    # 2. Chunking & Parent-Child
    chunk_opt_col1, chunk_opt_col2 = st.columns(2)
    with chunk_opt_col1:
        chunk_strategies = ["semantic", "hierarchical", "recursive"]
        curr_strategy = getattr(config, "DEFAULT_CHUNKING_STRATEGY", "semantic")
        strat_idx = chunk_strategies.index(curr_strategy) if curr_strategy in chunk_strategies else 0
        selected_chunk_strategy = st.selectbox(
            "Document Chunking Strategy",
            options=chunk_strategies,
            index=strat_idx,
            help="Semantic breaks on cosine distance shifts; Hierarchical links small child chunks to parent sections."
        )
    with chunk_opt_col2:
        top_k_val = st.slider(
            "Retrieval Top-K Chunks",
            min_value=1,
            max_value=15,
            value=getattr(config, "DEFAULT_TOP_K", 4),
            step=1,
            help="Number of context passages returned to the LLM."
        )

    chunk_col1, chunk_col2 = st.columns(2)
    with chunk_col1:
        chunk_sz = st.number_input("Chunk Size (Characters)", min_value=100, max_value=2000, value=config.CHUNK_SIZE, step=50)
    with chunk_col2:
        chunk_ov = st.number_input("Chunk Overlap (Characters)", min_value=0, max_value=400, value=config.CHUNK_OVERLAP, step=10)

    # 3. Hybrid Fusion & Advanced Weights
    st.markdown("##### ⚖️ Hybrid Search Optimization")
    fusion_col1, fusion_col2, fusion_col3 = st.columns(3)
    with fusion_col1:
        fusion_methods = ["rrf", "weighted"]
        curr_fusion = getattr(config, "HYBRID_FUSION_METHOD", "rrf")
        f_idx = 0 if curr_fusion == "rrf" else 1
        selected_fusion = st.selectbox(
            "Hybrid Fusion Method",
            options=fusion_methods,
            index=f_idx,
            format_func=lambda x: "Reciprocal Rank Fusion (RRF)" if x == "rrf" else "Weighted Min-Max Score Fusion"
        )
    with fusion_col2:
        dense_w = st.slider("FAISS Dense Weight", min_value=0.0, max_value=1.0, value=getattr(config, "DENSE_SEARCH_WEIGHT", 0.5), step=0.05)
    with fusion_col3:
        sparse_w = st.slider("BM25 Sparse Weight", min_value=0.0, max_value=1.0, value=getattr(config, "SPARSE_SEARCH_WEIGHT", 0.5), step=0.05)

    # 4. Pipeline Feature Toggles
    st.markdown("##### 🚀 Pipeline Feature Switches")
    sw_col1, sw_col2, sw_col3 = st.columns(3)
    with sw_col1:
        enable_rerank = st.checkbox("Cross-Encoder Reranking", value=getattr(config, "RERANKING_ENABLED", True))
        enable_parent_child = st.checkbox("Parent-Child Expansion", value=getattr(config, "PARENT_CHILD_ENABLED", True))
    with sw_col2:
        enable_multi_q = st.checkbox("Multi-Query Retrieval", value=getattr(config, "MULTI_QUERY_ENABLED", True))
        multi_q_cnt = st.number_input("Number of Alternative Queries", min_value=2, max_value=5, value=getattr(config, "MULTI_QUERY_COUNT", 3))
    with sw_col3:
        enable_compression = st.checkbox("Contextual Compression", value=getattr(config, "CONTEXTUAL_COMPRESSION_ENABLED", True))
        enable_hallucination = st.checkbox("Hallucination Detection", value=getattr(config, "HALLUCINATION_DETECTION_ENABLED", True))

    ollama_url = st.text_input(
        "Ollama API URL Endpoint",
        value=config.OLLAMA_API_URL
    )
    
    if st.button("🔌 Test Ollama Connection", use_container_width=True):
        try:
            response = requests.get(f"{ollama_url}/api/tags", timeout=3)
            if response.status_code == 200:
                st.success("✅ Connected to Ollama Service successfully!")
                models = [m['name'] for m in response.json().get('models', [])]
                st.info(f"Available local models: {', '.join(models)}")
            else:
                st.warning(f"Ollama responded with status code {response.status_code}.")
        except Exception as e:
            st.error(f"❌ Failed to reach Ollama at {ollama_url}: {e}")

    session_timeout = st.number_input(
        "Session Idle Timeout Limit (Minutes)",
        min_value=5,
        max_value=120,
        value=config.SESSION_TIMEOUT_MINUTES
    )

    st.markdown("---")
    st.subheader("💾 Server Storage Footprint")
    def get_dir_size_mb(directory: Path) -> float:
        total_size = 0
        if directory.exists():
            for fp in directory.glob('**/*'):
                if fp.is_file():
                    total_size += fp.stat().st_size
        return total_size / (1024.0 * 1024.0)

    uploads_size = get_dir_size_mb(config.UPLOADS_DIR)
    vectors_size = get_dir_size_mb(config.VECTOR_STORE_DIR)
    db_size = get_dir_size_mb(config.DATA_DIR)

    st.write(f"- **Uploads Directory:** {uploads_size:.4f} MB")
    st.write(f"- **Vector Index Directory:** {vectors_size:.4f} MB")
    st.write(f"- **System Database:** {db_size:.4f} MB")

    if st.button("🧹 Flush Uploads Cache", type="secondary", use_container_width=True):
        for item in config.UPLOADS_DIR.glob('*'):
            if item.is_file():
                try:
                    os.remove(item)
                except Exception:
                    pass
        st.success("Cleaned temporary upload folder contents!")
        db.log_analytics(st.session_state.user_id, "storage_cleanup", "Cleared temporary file uploads")
        st.rerun()

    if st.button("💾 Save System Configurations", type="primary", use_container_width=True):
        env_file = Path(config.BASE_DIR) / ".env"
        dotenv.set_key(str(env_file), "OLLAMA_API_URL", ollama_url)
        dotenv.set_key(str(env_file), "TESSERACT_CMD_PATH", tesseract_path)
        dotenv.set_key(str(env_file), "DEFAULT_EMBEDDING_MODEL", selected_emb_model)
        dotenv.set_key(str(env_file), "DEFAULT_RERANKER_MODEL", selected_reranker)
        dotenv.set_key(str(env_file), "DEFAULT_CHUNKING_STRATEGY", selected_chunk_strategy)
        dotenv.set_key(str(env_file), "DEFAULT_TOP_K", str(top_k_val))
        dotenv.set_key(str(env_file), "DEFAULT_OCR_ENGINE", ocr_engine_opt)
        dotenv.set_key(str(env_file), "OCR_LANG", ocr_lang_opt)
        dotenv.set_key(str(env_file), "CHUNK_SIZE", str(chunk_sz))
        dotenv.set_key(str(env_file), "CHUNK_OVERLAP", str(chunk_ov))
        dotenv.set_key(str(env_file), "SESSION_TIMEOUT_MINUTES", str(session_timeout))
        dotenv.set_key(str(env_file), "HYBRID_FUSION_METHOD", selected_fusion)
        dotenv.set_key(str(env_file), "DENSE_SEARCH_WEIGHT", str(dense_w))
        dotenv.set_key(str(env_file), "SPARSE_SEARCH_WEIGHT", str(sparse_w))
        dotenv.set_key(str(env_file), "RERANKING_ENABLED", str(enable_rerank).lower())
        dotenv.set_key(str(env_file), "PARENT_CHILD_ENABLED", str(enable_parent_child).lower())
        dotenv.set_key(str(env_file), "MULTI_QUERY_ENABLED", str(enable_multi_q).lower())
        dotenv.set_key(str(env_file), "MULTI_QUERY_COUNT", str(multi_q_cnt))
        dotenv.set_key(str(env_file), "CONTEXTUAL_COMPRESSION_ENABLED", str(enable_compression).lower())
        dotenv.set_key(str(env_file), "HALLUCINATION_DETECTION_ENABLED", str(enable_hallucination).lower())
        
        config.OLLAMA_API_URL = ollama_url
        config.SESSION_TIMEOUT_MINUTES = session_timeout
        config.DEFAULT_EMBEDDING_MODEL = selected_emb_model
        config.DEFAULT_RERANKER_MODEL = selected_reranker
        config.DEFAULT_CHUNKING_STRATEGY = selected_chunk_strategy
        config.DEFAULT_TOP_K = top_k_val
        config.DEFAULT_OCR_ENGINE = ocr_engine_opt
        config.OCR_LANG = ocr_lang_opt
        config.CHUNK_SIZE = chunk_sz
        config.CHUNK_OVERLAP = chunk_ov
        config.HYBRID_FUSION_METHOD = selected_fusion
        config.DENSE_SEARCH_WEIGHT = dense_w
        config.SPARSE_SEARCH_WEIGHT = sparse_w
        config.RERANKING_ENABLED = enable_rerank
        config.PARENT_CHILD_ENABLED = enable_parent_child
        config.MULTI_QUERY_ENABLED = enable_multi_q
        config.MULTI_QUERY_COUNT = multi_q_cnt
        config.CONTEXTUAL_COMPRESSION_ENABLED = enable_compression
        config.HALLUCINATION_DETECTION_ENABLED = enable_hallucination
        
        st.success("System configuration updates saved to .env!")
        db.log_analytics(st.session_state.user_id, "save_configs", "Applied system settings")
        st.rerun()

