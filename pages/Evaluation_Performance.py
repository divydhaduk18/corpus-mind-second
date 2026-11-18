"""Evaluation, Performance Optimization & Scalability Suite for Corpus Mind.

Provides dedicated interfaces for:
1. RAGAS Advanced RAG Evaluation (Faithfulness, Relevancy, Precision, Recall, Correctness)
2. Retrieval Benchmarking (Recall@K, MRR, nDCG@5/10, Latency, Throughput)
3. Multi-Tier Cache Performance & Eviction Management
4. Asynchronous Background Document Ingestion Queue & Stage Tracking
5. Scalability Benchmarking & Hardware Resource Monitoring (CPU, RAM, Backups)
"""

import streamlit as st
import os
import sys
import time
import io
import json
from pathlib import Path
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.auth import AuthManager
from utils.database import db
from utils.ragas_evaluator import RAGASEvaluator, DEFAULT_RAGAS_DATASET
from utils.retrieval_benchmark import RetrievalBenchmarkEngine, RETRIEVAL_EVALUATION_DATASET
from utils.cache_manager import CacheManager
from utils.async_processor import AsyncDocumentProcessor
from utils.scalable_index import ScalableIndexManager
from utils.llm_provider import LLMProviderManager

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
    st.markdown("---")
    if st.button("💬 Back to Chat", use_container_width=True):
        st.switch_page("pages/Chat.py")
    if st.button("🛠️ Document Tools", use_container_width=True):
        st.switch_page("pages/Document_Tools.py")

st.title("🎯 Evaluation, Performance & Scalability (Features 23–27)")
st.markdown(
    "Evaluate RAG quality with RAGAS, benchmark retrieval (Recall@K, MRR, nDCG), "
    "monitor multi-tier caching, track background ingestion jobs, and measure system scalability."
)

eval_tabs = st.tabs([
    "📊 RAGAS Evaluation",
    "🔬 Retrieval Benchmarking",
    "⚡ Cache Optimization",
    "🔄 Async Job Queue",
    "📈 Scalability & Hardware"
])


# =====================================================================
# TAB 1: RAGAS ADVANCED EVALUATION
# =====================================================================
with eval_tabs[0]:
    st.subheader("📊 RAGAS Advanced RAG Quality Evaluation (Feature #23)")
    st.markdown("Quantify Faithfulness, Answer Relevancy, Context Precision, and Context Recall.")

    ragas_col1, ragas_col2 = st.columns(2)
    with ragas_col1:
        providers = LLMProviderManager.list_providers()
        sel_eval_prov = st.selectbox("LLM Provider for Evaluation", options=providers, key="ragas_prov_sel")
    with ragas_col2:
        prov_models = LLMProviderManager.get_models_for_provider(sel_eval_prov)
        sel_eval_model = st.selectbox("LLM Model", options=prov_models, key="ragas_model_sel")

    # Dataset Selection
    dataset_opt = st.radio(
        "Evaluation Dataset",
        options=["Standardized Corpus Mind Evaluation Set (3 Verified Cases)", "Upload Custom JSON Dataset"],
        horizontal=True
    )

    custom_eval_items = None
    if "Upload" in dataset_opt:
        up_json = st.file_uploader("Upload JSON Dataset", type=["json"], key="up_ragas_json")
        if up_json:
            try:
                custom_eval_items = json.load(up_json)
                st.success(f"Loaded {len(custom_eval_items)} custom evaluation test cases!")
            except Exception as e:
                st.error(f"Invalid JSON file: {e}")

    target_eval_suite = custom_eval_items if custom_eval_items else DEFAULT_RAGAS_DATASET

    if st.button("🚀 Run RAGAS Evaluation Pipeline", type="primary", key="btn_run_ragas"):
        with st.spinner("Executing RAG retrieval, generating answers, and computing RAGAS metrics..."):
            report = RAGASEvaluator.evaluate_pipeline(
                user_id=st.session_state.user_id,
                dataset=target_eval_suite,
                provider_name=sel_eval_prov,
                model_name=sel_eval_model
            )
            st.session_state.ragas_last_report = report

    rep = st.session_state.get("ragas_last_report")
    if rep:
        metrics = rep["summary_metrics"]
        st.markdown("### 🏆 Evaluation Summary Metrics")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        with m1:
            st.metric("Faithfulness", f"{int(metrics['faithfulness'] * 100)}%")
        with m2:
            st.metric("Relevancy", f"{int(metrics['answer_relevancy'] * 100)}%")
        with m3:
            st.metric("Precision", f"{int(metrics['context_precision'] * 100)}%")
        with m4:
            st.metric("Recall", f"{int(metrics['context_recall'] * 100)}%")
        with m5:
            st.metric("Correctness", f"{int(metrics['answer_correctness'] * 100)}%")
        with m6:
            st.metric("Utilization", f"{int(metrics['context_utilization'] * 100)}%")

        # Detailed Query Table
        st.markdown("#### 🔍 Per-Question Results")
        df_queries = pd.DataFrame(rep["evaluated_queries"])
        st.dataframe(df_queries[["question", "faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"]], use_container_width=True)

        # Export Report
        st.download_button(
            label="📥 Download RAGAS Report (JSON)",
            data=json.dumps(rep, indent=2),
            file_name=f"ragas_evaluation_report_{int(time.time())}.json",
            mime="application/json"
        )


# =====================================================================
# TAB 2: RETRIEVAL BENCHMARKING
# =====================================================================
with eval_tabs[1]:
    st.subheader("🔬 Retrieval Algorithm Benchmarking (Feature #24)")
    st.markdown("Measure Recall@1/3/5/10, MRR, nDCG@5/10, and latency across retrieval strategies.")

    bench_col1, bench_col2, bench_col3 = st.columns(3)
    with bench_col1:
        bm_search_mode = st.selectbox("Retrieval Algorithm", options=["hybrid", "dense", "bm25"], format_func=lambda x: {
            "hybrid": "Hybrid (FAISS Dense + BM25 Sparse)",
            "dense": "Dense FAISS Vector Only",
            "bm25": "Keyword BM25 Only"
        }.get(x, x))
    with bench_col2:
        bm_rerank = st.checkbox("Cross-Encoder Reranking", value=True)
    with bench_col3:
        bm_top_k = st.slider("Retrieval Top-K", min_value=1, max_value=10, value=5)

    if st.button("🏁 Run Retrieval Benchmark", type="primary", key="btn_run_ret_bench"):
        with st.spinner("Benchmarking retrieval quality across evaluation suite..."):
            ret_summary = RetrievalBenchmarkEngine.evaluate_retrieval_configuration(
                user_id=st.session_state.user_id,
                search_mode=bm_search_mode,
                enable_rerank=bm_rerank,
                top_k=bm_top_k
            )
            st.session_state.retrieval_bench_summary = ret_summary

    r_sum = st.session_state.get("retrieval_bench_summary")
    if r_sum:
        st.markdown("### 🏆 Information Retrieval Metrics")
        b1, b2, b3, b4, b5 = st.columns(5)
        with b1:
            st.metric("Recall@5", f"{int(r_sum['recall_at_5'] * 100)}%")
        with b2:
            st.metric("Recall@10", f"{int(r_sum['recall_at_10'] * 100)}%")
        with b3:
            st.metric("MRR", f"{r_sum['mrr']:.4f}")
        with b4:
            st.metric("nDCG@5", f"{r_sum['ndcg_at_5']:.4f}")
        with b5:
            st.metric("Avg Latency", f"{r_sum['avg_latency_ms']:.1f} ms")

        st.caption(f"Throughput: **{r_sum['throughput_qps']} queries/sec** | Recall@1: **{int(r_sum['recall_at_1'] * 100)}%** | Precision@K: **{int(r_sum['precision_at_k'] * 100)}%**")
        
        # Per query details
        st.markdown("#### 🔍 Query Level Retrieval Ranks")
        df_ret_q = pd.DataFrame(r_sum["query_details"])
        st.dataframe(df_ret_q, use_container_width=True)


# =====================================================================
# TAB 3: CACHE OPTIMIZATION
# =====================================================================
with eval_tabs[2]:
    st.subheader("⚡ Multi-Tier Caching Optimization (Feature #25)")
    st.markdown("Embedding Cache, Query LRU, User-Isolated Retrieval Cache, and Response Caches.")

    cache_mgr = CacheManager.get_instance()
    metrics_cache = cache_mgr.get_metrics_summary()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Embedding Cache", f"{metrics_cache['embedding_cache']['size']} items", f"Hit Rate: {metrics_cache['embedding_cache']['hit_rate_pct']}%")
    with c2:
        st.metric("Query LRU Cache", f"{metrics_cache['query_cache']['size']} items", f"Hit Rate: {metrics_cache['query_cache']['hit_rate_pct']}%")
    with c3:
        st.metric("Retrieval Cache", f"{metrics_cache['retrieval_cache']['size']} items", f"Hit Rate: {metrics_cache['retrieval_cache']['hit_rate_pct']}%")
    with c4:
        st.metric("LLM Response Cache", f"{metrics_cache['llm_response_cache']['size']} items", f"Hit Rate: {metrics_cache['llm_response_cache']['hit_rate_pct']}%")

    st.markdown("---")
    st.markdown("##### ⚙️ Cache Maintenance & Invalidation Controls")
    
    col_clear, col_info = st.columns([1, 2])
    with col_clear:
        if st.button("🧹 Flush All Cache Layers", type="secondary", use_container_width=True):
            cache_mgr.clear_all_caches()
            st.success("All multi-tier caches successfully invalidated and cleared!")
            st.rerun()
    with col_info:
        st.info(
            "🔒 **Security Guarantee**: Retrieval cache keys are strictly bound to `user_id`, active index versions, and "
            "document filter scopes. Cross-user cache contamination is mathematically prevented."
        )


# =====================================================================
# TAB 4: ASYNC JOB QUEUE
# =====================================================================
with eval_tabs[3]:
    st.subheader("🔄 Asynchronous Background Document Ingestion (Feature #26)")
    st.markdown("Track background document ingestion stages without blocking the chat UI.")

    # Retrieve active user jobs
    jobs = db.get_document_jobs_by_user(st.session_state.user_id)
    
    if st.button("🔄 Refresh Job Queue", use_container_width=True):
        st.rerun()

    if not jobs:
        st.info("No active or historical background ingestion jobs found.")
    else:
        st.markdown(f"**Found {len(jobs)} background ingestion jobs:**")
        for job in jobs:
            st.markdown(f"#### 📄 `{job['file_name']}` (Job ID: `{job['job_id']}`)")
            status = job['status']
            stage = job['current_stage']
            pct = job['progress_pct']

            st.progress(pct / 100.0)
            st.caption(f"Status: **{status.upper()}** | Current Stage: **{stage}** | Progress: **{pct}%** | Updated: **{job['updated_at']}**")

            if status == "failed":
                st.error(f"Error: {job.get('error_message', 'Unknown failure')}")
                if st.button(f"🔁 Retry Job {job['job_id']}", key=f"retry_{job['job_id']}"):
                    AsyncDocumentProcessor.retry_job(job['job_id'])
                    st.success("Job resubmitted to background worker pool!")
                    st.rerun()
            elif status == "completed":
                st.success("✅ Ingestion complete. Document is indexed and ready for search.")
            st.markdown("---")


# =====================================================================
# TAB 5: SCALABILITY & HARDWARE MONITORING
# =====================================================================
with eval_tabs[4]:
    st.subheader("📈 System Scalability & Hardware Monitoring (Feature #27)")
    st.markdown("Incremental vector indexing, memory profiling, backups, and scalability benchmarks.")

    sys_metrics = ScalableIndexManager.get_system_metrics()
    h1, h2, h3 = st.columns(3)
    with h1:
        st.metric("Process RSS Memory", f"{sys_metrics['rss_ram_mb']} MB")
    with h2:
        st.metric("Virtual Memory (VMS)", f"{sys_metrics['vms_ram_mb']} MB")
    with h3:
        st.metric("Process CPU Utilization", f"{sys_metrics['cpu_percent']}%")

    st.markdown("---")
    st.markdown("##### 💾 Point-in-Time Index Backup & Disaster Recovery")
    if st.button("💾 Create Point-in-Time Index Snapshot", use_container_width=True):
        ok = ScalableIndexManager.create_index_backup(st.session_state.user_id)
        if ok:
            st.success("Snapshot created! FAISS index and metadata copied to backup store.")
        else:
            st.warning("No active index found to backup. Ingest documents first!")

    st.markdown("---")
    st.markdown("##### 🚀 Run Scalability Load Test")
    st.markdown("Tests incremental batch indexing across synthetic document collections.")

    col_bench, _ = st.columns([1, 1])
    with col_bench:
        test_doc_counts = st.multiselect("Test Document Counts", options=[10, 50, 100], default=[10, 50])

    if st.button("🏁 Execute Scalability Load Test", type="primary"):
        with st.spinner("Executing incremental batch indexing and latency test..."):
            scale_results = ScalableIndexManager.run_scalability_benchmark(
                user_id=st.session_state.user_id,
                collection_sizes=test_doc_counts
            )
            st.session_state.scale_results = scale_results

    sc_res = st.session_state.get("scale_results")
    if sc_res:
        st.markdown("### 🏆 Scalability Benchmark Results")
        df_scale = pd.DataFrame(sc_res["benchmark_results"])
        st.dataframe(df_scale, use_container_width=True)

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.caption("Ingestion Time (Seconds) vs Collection Size")
            st.bar_chart(df_scale.set_index("documents_count")["ingest_time_sec"])
        with chart_col2:
            st.caption("Query Latency (ms) vs Collection Size")
            st.bar_chart(df_scale.set_index("documents_count")["query_latency_ms"])
