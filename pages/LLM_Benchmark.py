"""LLM Evaluation & Model Benchmarking Page for Corpus Mind.

Enables side-by-side comparison of local and cloud models on identical questions & retrieved context.
Tracks TTFT, latency, TPS, faithfulness, correctness, citation precision, and memory footprint.
"""

import streamlit as st
import os
import sys
import time
import pandas as pd
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.auth import AuthManager
from utils.database import db
from utils.llm_provider import LLMProviderManager
from utils.llm_evaluator import LLMEvaluator, DEFAULT_BENCHMARK_SUITE

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
    if st.button("💬 Back to Chat", use_container_width=True):
        st.switch_page("pages/Chat.py")

st.title("📊 LLM Model Benchmarking & Fair Comparison (Improvement #12)")
st.markdown(
    "Evaluate multiple local and cloud LLMs on **identical questions**, **identical retrieved context**, "
    "and **standardized generation parameters**. Measures speed, faithfulness, and accuracy."
)

tab_run, tab_history = st.tabs(["🚀 Run New Benchmark", "📜 Benchmark History & Analytics"])

# =====================================================================
# TAB 1: RUN NEW BENCHMARK
# =====================================================================
with tab_run:
    st.subheader("1. Select Providers & Models for Comparative Evaluation")

    providers = LLMProviderManager.list_providers()
    col1, col2 = st.columns(2)
    
    with col1:
        selected_providers = st.multiselect(
            "Providers to Benchmark",
            options=providers,
            default=["ollama"],
            format_func=lambda x: {
                "ollama": "🦙 Ollama (Local)",
                "openai": "🟢 OpenAI (Cloud)",
                "gemini": "✨ Google Gemini (Cloud)",
                "anthropic": "🟣 Anthropic Claude (Cloud)"
            }.get(x, x.title())
        )

    # Let user pick specific model for each selected provider
    provider_model_pairs = []
    with col2:
        st.markdown("**Configure Target Model per Provider:**")
        for p in selected_providers:
            avail = LLMProviderManager.get_models_for_provider(p)
            sel_m = st.selectbox(f"Model for {p.title()}", options=avail, key=f"bench_m_{p}")
            provider_model_pairs.append((p, sel_m))

    st.markdown("---")
    st.subheader("2. Evaluation Dataset & Test Questions")
    
    dataset_mode = st.radio(
        "Benchmark Dataset",
        options=["Standardized Evaluation Suite (4 Verified Q&A / Contracts / Finance / Negative)", "Custom Question"],
        index=0
    )

    custom_suite = None
    if "Custom" in dataset_mode:
        custom_q = st.text_input("Enter Question", value="What is the key takeaway from the latest quarterly report?")
        custom_ctx = st.text_area("Enter Mock Retrieved Context", value="In Q3, Acme Corp recorded $142.5M in revenue and 21.4% operating margins.")
        custom_facts = st.text_input("Expected Key Facts (comma-separated)", value="$142.5M, 21.4%, Q3")
        custom_suite = [{
            "id": "custom_1",
            "category": "Custom",
            "question": custom_q,
            "ground_truth": custom_facts,
            "key_facts": [f.strip() for f in custom_facts.split(",") if f.strip()],
            "retrieved_context": [{
                "file_name": "custom_doc.txt",
                "page_number": 1,
                "section": "General",
                "text": custom_ctx
            }]
        }]

    test_suite_to_run = custom_suite if custom_suite else DEFAULT_BENCHMARK_SUITE

    # Display preview of test suite
    with st.expander("👀 View Benchmark Questions & Reference Passages", expanded=False):
        for idx, item in enumerate(test_suite_to_run, start=1):
            st.markdown(f"**Q{idx} ({item['category']}):** {item['question']}")
            st.caption(f"Ground Truth Reference: {item.get('ground_truth', '')}")

    st.markdown("---")
    st.subheader("3. Benchmark Execution")

    start_bench = st.button("🏁 Start Comparative Benchmark Run", type="primary", use_container_width=True)

    if start_bench:
        if not provider_model_pairs:
            st.error("Please select at least one provider and model to benchmark.")
        else:
            all_summaries = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            total_runs = len(provider_model_pairs)
            for idx, (prov, mdl) in enumerate(provider_model_pairs):
                status_text.markdown(f"⏳ Benchmarking **{prov.upper()}** ({mdl})...")
                try:
                    summary = LLMEvaluator.evaluate_model_on_suite(
                        provider_name=prov,
                        model_name=mdl,
                        test_suite=test_suite_to_run
                    )
                    all_summaries.append(summary)
                except Exception as ex:
                    st.warning(f"Error evaluating {prov} ({mdl}): {ex}")
                progress_bar.progress((idx + 1) / total_runs)

            status_text.success("🎉 Benchmark evaluation completed successfully!")

            # Build comparison table
            table_rows = []
            for s in all_summaries:
                table_rows.append({
                    "Provider": s["provider"].upper(),
                    "Model": s["model"],
                    "Avg Latency (ms)": s["avg_latency_ms"],
                    "Avg TTFT (ms)": s["avg_ttft_ms"],
                    "Tokens / Sec (TPS)": s["avg_tokens_per_second"],
                    "Faithfulness Score": f"{int(s['avg_faithfulness'] * 100)}%",
                    "Answer Correctness": f"{int(s['correctness_rate'] * 100)}%",
                    "Citation Rate": f"{int(s['citation_rate'] * 100)}%",
                    "Error Rate": f"{int(s['error_rate'] * 100)}%"
                })

            df_comp = pd.DataFrame(table_rows)
            st.markdown("### 🏆 Comparative Benchmark Results")
            st.dataframe(df_comp, use_container_width=True)

            # Chart Latencies and Speeds
            st.markdown("### 📈 Latency & Speed Comparison")
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.caption("Average Generation Latency (ms) - Lower is Better")
                st.bar_chart(df_comp.set_index("Model")["Avg Latency (ms)"])
            with chart_col2:
                st.caption("Tokens Per Second (TPS) - Higher is Better")
                st.bar_chart(df_comp.set_index("Model")["Tokens / Sec (TPS)"])

            # Detailed Per-Question Outputs
            st.markdown("### 🔍 Detailed Model Responses")
            for item in test_suite_to_run:
                st.markdown(f"#### Question: *{item['question']}*")
                cols = st.columns(len(all_summaries))
                for col_idx, s in enumerate(all_summaries):
                    with cols[col_idx]:
                        st.markdown(f"**{s['provider'].upper()} ({s['model']})**")
                        # Find matching result
                        matched = next((r for r in s["detailed_results"] if r.get("id") == item["id"]), None)
                        if matched:
                            if matched.get("error"):
                                st.error(f"Error: {matched['error']}")
                            else:
                                st.info(matched.get("generated_answer", ""))
                                st.caption(
                                    f"Latency: {matched.get('latency_ms', 0):.0f}ms | "
                                    f"TTFT: {matched.get('ttft_ms', 0):.0f}ms | "
                                    f"Faithfulness: {int(matched.get('faithfulness_score', 0) * 100)}%"
                                )


# =====================================================================
# TAB 2: BENCHMARK HISTORY & ANALYTICS
# =====================================================================
with tab_history:
    st.subheader("📜 Historical Benchmark Runs")
    saved = LLMEvaluator.list_saved_benchmarks()
    if not saved:
        st.info("No saved benchmark runs found yet. Run your first benchmark above!")
    else:
        history_rows = []
        for s in saved:
            history_rows.append({
                "Timestamp": s.get("timestamp", "")[:19].replace("T", " "),
                "Provider": s.get("provider", "").upper(),
                "Model": s.get("model", ""),
                "Latency (ms)": s.get("avg_latency_ms", 0),
                "TTFT (ms)": s.get("avg_ttft_ms", 0),
                "TPS": s.get("avg_tokens_per_second", 0),
                "Faithfulness": f"{int(s.get('avg_faithfulness', 0) * 100)}%",
                "Correctness": f"{int(s.get('correctness_rate', 0) * 100)}%"
            })
        st.dataframe(pd.DataFrame(history_rows), use_container_width=True)
