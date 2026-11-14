"""Document Tools & Multimodal Intelligence Suite for Corpus Mind.

Provides dedicated interfaces for:
1. Advanced PDF Table Extraction (Preview, CSV & Excel export)
2. Advanced & Multilingual OCR (English, Hindi, Gujarati with Deskewing & CLAHE)
3. Visual Document & Diagram Explorer (Figures, Charts, Vision Descriptions)
4. Excel Intelligence (Multi-sheet, deterministic calculations, charts, export)
5. Document Comparison (Side-by-side diffs, changed numbers/dates, semantic drift)
6. Document Summarization (Short, Detailed, Executive, Hierarchical Map-Reduce)
"""

import streamlit as st
import os
import sys
import io
from pathlib import Path
import pandas as pd
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.auth import AuthManager
from utils.database import db
from utils.document_loader import DocumentLoader
from utils.table_extractor import TableExtractor, ExtractedTable
from utils.multilingual_ocr import MultilingualOCR
from utils.visual_rag import VisualRAGManager
from utils.excel_intelligence import ExcelWorkbookManager, ExcelIntelligenceEngine
from utils.document_comparator import DocumentComparator
from utils.document_summarizer import DocumentSummarizer

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
    if st.button("📂 Manage Documents", use_container_width=True):
        st.switch_page("pages/Documents.py")

st.title("🛠️ Document Intelligence & Multimodal Suite (Features 16–22)")
st.markdown(
    "Perform deep table extraction, multilingual OCR, visual figure understanding, "
    "spreadsheet calculations, document diffs, and hierarchical summarization."
)

# Fetch user's uploaded documents
user_docs = db.get_documents_by_user(st.session_state.user_id)
doc_name_to_path = {d['file_name']: d['file_path'] for d in user_docs if os.path.exists(d['file_path'])}

tools_tabs = st.tabs([
    "📊 Table Extraction",
    "👁️ Multilingual OCR",
    "🖼️ Visual Diagram Chat",
    "📈 Excel Intelligence",
    "⚖️ Document Comparison",
    "📝 Document Summarizer"
])


# =====================================================================
# TAB 1: ADVANCED PDF TABLE EXTRACTION
# =====================================================================
with tools_tabs[0]:
    st.subheader("📊 Advanced PDF Table Extraction (Feature #16)")
    st.markdown("Extract structured data tables from digital and scanned PDFs with multi-engine fallback.")

    pdf_files = [name for name in doc_name_to_path.keys() if name.lower().endswith(".pdf")]
    
    if not pdf_files:
        st.info("No PDF documents found in your library. Upload a PDF in Documents first!")
    else:
        sel_pdf = st.selectbox("Select PDF Document to Extract Tables From", options=pdf_files, key="table_pdf_sel")
        pdf_path = doc_name_to_path[sel_pdf]

        if st.button("🔍 Detect & Extract Tables", type="primary", key="btn_extract_tables"):
            with st.spinner("Extracting tables across pages..."):
                extracted_tables = TableExtractor.extract_tables_from_pdf(pdf_path)
                st.session_state.extracted_tables = extracted_tables

        tables: List[ExtractedTable] = st.session_state.get("extracted_tables", [])
        if tables:
            st.success(f"Extracted **{len(tables)}** tables from `{sel_pdf}`!")
            
            # Export all to Excel button
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                for idx, t in enumerate(tables, start=1):
                    t.to_dataframe().to_excel(writer, sheet_name=f"Page_{t.page_number}_T{idx}"[:31], index=False)
            excel_bytes = excel_buffer.getvalue()

            st.download_button(
                label="📥 Download All Tables as Excel (.xlsx)",
                data=excel_bytes,
                file_name=f"{Path(sel_pdf).stem}_all_tables.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            for idx, tab in enumerate(tables, start=1):
                df = tab.to_dataframe()
                st.markdown(f"#### Table {idx}: `{tab.table_id}` (Page {tab.page_number})")
                st.caption(f"Extraction Method: **{tab.extraction_method}** | Scanned: **{tab.is_scanned}**")
                st.dataframe(df, use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label=f"📥 Download Table {idx} (CSV)",
                        data=tab.to_csv_string(),
                        file_name=f"{sel_pdf}_table_{idx}.csv",
                        mime="text/csv",
                        key=f"dl_csv_{idx}"
                    )


# =====================================================================
# TAB 2: ADVANCED & MULTILINGUAL OCR
# =====================================================================
with tools_tabs[1]:
    st.subheader("👁️ Advanced & Multilingual OCR (Features #17 & #18)")
    st.markdown("Deskewing, CLAHE contrast enhancement, and Unicode preservation for English, Hindi, and Gujarati.")

    ocr_col1, ocr_col2 = st.columns(2)
    with ocr_col1:
        lang_choice = st.selectbox(
            "Target OCR Recognition Language",
            options=["eng", "hin", "guj", "eng+hin+guj"],
            format_func=lambda x: {
                "eng": "English (Latin Script)",
                "hin": "Hindi (हिंदी - Devanagari Script)",
                "guj": "Gujarati (ગુજરાતી - Gujarati Script)",
                "eng+hin+guj": "Multilingual (English + Hindi + Gujarati)"
            }.get(x, x)
        )
    with ocr_col2:
        apply_preprocess = st.checkbox("Enable Advanced Preprocessing (Deskew + CLAHE)", value=True)

    uploaded_img = st.file_uploader(
        "Upload Image or Document Scan for OCR",
        type=["png", "jpg", "jpeg", "webp", "bmp", "tiff", "pdf"]
    )

    if uploaded_img:
        # Load PIL image
        if uploaded_img.name.lower().endswith(".pdf"):
            import pymupdf as fitz
            doc_temp = fitz.open(stream=uploaded_img.read(), filetype="pdf")
            pix = doc_temp[0].get_pixmap(dpi=150)
            pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        else:
            pil_image = Image.open(uploaded_img)

        prev_col1, prev_col2 = st.columns(2)
        with prev_col1:
            st.image(pil_image, caption="Original Image", use_container_width=True)
        with prev_col2:
            if apply_preprocess:
                enhanced_img = MultilingualOCR.advanced_preprocess(pil_image)
                st.image(enhanced_img, caption="Preprocessed Image (Deskewed & CLAHE)", use_container_width=True)
            else:
                enhanced_img = pil_image

        if st.button("🚀 Run Multilingual OCR Extraction", type="primary"):
            with st.spinner("Extracting text and preserving Unicode script..."):
                ocr_res = MultilingualOCR.extract_multilingual_text(
                    image=pil_image,
                    lang=lang_choice,
                    preprocess=apply_preprocess
                )
                st.session_state.multilingual_ocr_result = ocr_res

        res = st.session_state.get("multilingual_ocr_result")
        if res:
            st.markdown("### 📜 Extracted Text & Metadata")
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric("Detected Script", res.get("language_name", "English"))
            with m_col2:
                st.metric("OCR Confidence Score", f"{res.get('confidence', 0)}%")
            with m_col3:
                st.metric("Extracted Word Count", res.get("word_count", 0))

            st.text_area("Unicode Extracted Text", value=res.get("text", ""), height=220)
            st.download_button(
                label="📥 Download Extracted Text (UTF-8)",
                data=res.get("text", "").encode("utf-8"),
                file_name=f"ocr_extracted_{res.get('detected_script', 'doc')}.txt",
                mime="text/plain"
            )


# =====================================================================
# TAB 3: VISUAL DOCUMENT & DIAGRAM CHAT
# =====================================================================
with tools_tabs[2]:
    st.subheader("🖼️ Visual Document & Diagram Explorer (Feature #19)")
    st.markdown("Inspect extracted diagrams, charts, and architectural figures from PDFs with vision descriptions.")

    pdf_files_vis = [name for name in doc_name_to_path.keys() if name.lower().endswith(".pdf")]
    if not pdf_files_vis:
        st.info("Upload a PDF with diagrams or figures to use Visual RAG.")
    else:
        sel_vis_pdf = st.selectbox("Select PDF Document", options=pdf_files_vis, key="vis_pdf_sel")
        vis_pdf_path = doc_name_to_path[sel_vis_pdf]

        if st.button("🔍 Extract Diagrams & Figures", type="primary", key="btn_extract_figs"):
            with st.spinner("Extracting embedded figures and generating vision descriptions..."):
                figs = VisualRAGManager.extract_figures_from_pdf(
                    file_path=vis_pdf_path,
                    user_id=st.session_state.user_id,
                    generate_descriptions=True
                )
                st.session_state.extracted_figures = figs

        figures: List = st.session_state.get("extracted_figures", [])
        if figures:
            st.success(f"Extracted **{len(figures)}** figures from `{sel_vis_pdf}`!")
            for fig in figures:
                st.markdown(f"#### Figure: `{fig.figure_id}` (Page {fig.page_number} - Type: **{fig.figure_type.upper()}**)")
                col_img, col_desc = st.columns([1, 1])
                with col_img:
                    if os.path.exists(fig.image_path):
                        st.image(fig.image_path, use_container_width=True)
                with col_desc:
                    st.info(fig.description)
                    if fig.detected_labels:
                        st.caption(f"**Detected Labels & Legends:** {', '.join(fig.detected_labels)}")


# =====================================================================
# TAB 4: EXCEL INTELLIGENCE
# =====================================================================
with tools_tabs[3]:
    st.subheader("📈 Excel & Spreadsheet Intelligence (Feature #20)")
    st.markdown("Deterministic Python calculations (Sum, Average, Min, Max, Top-N) without unsafe code execution.")

    excel_files = [name for name in doc_name_to_path.keys() if name.lower().endswith((".xlsx", ".xls", ".csv"))]
    uploaded_sheet_file = st.file_uploader("Upload New Spreadsheet (.xlsx, .csv)", type=["xlsx", "xls", "csv"], key="uploader_excel")

    target_sheet_path = None
    target_sheet_name = ""

    if uploaded_sheet_file:
        save_path = config.get_user_uploads_dir(st.session_state.user_id) / uploaded_sheet_file.name
        with open(save_path, "wb") as f_out:
            f_out.write(uploaded_sheet_file.read())
        target_sheet_path = str(save_path)
        target_sheet_name = uploaded_sheet_file.name
    elif excel_files:
        sel_excel = st.selectbox("Or choose existing uploaded spreadsheet", options=excel_files, key="sel_existing_excel")
        target_sheet_path = doc_name_to_path[sel_excel]
        target_sheet_name = sel_excel

    if target_sheet_path:
        sheets = ExcelWorkbookManager.load_workbook_sheets(target_sheet_path)
        sheet_names = list(sheets.keys())
        active_sheet_name = st.selectbox("Select Worksheet", options=sheet_names)
        active_df = sheets[active_sheet_name]

        st.markdown(f"**Worksheet Preview: `{active_sheet_name}` ({len(active_df)} rows, {len(active_df.columns)} columns)**")
        st.dataframe(active_df.head(10), use_container_width=True)

        st.markdown("##### 💬 Natural Language Spreadsheet Query")
        sample_q = "What is the total revenue?"
        user_excel_query = st.text_input("Ask a question about this spreadsheet", value=sample_q)

        if st.button("⚡ Calculate & Answer", type="primary", key="btn_calc_excel"):
            calc_res = ExcelIntelligenceEngine.query_spreadsheet(
                df=active_df,
                query=user_excel_query,
                sheet_name=active_sheet_name,
                file_name=target_sheet_name
            )
            st.success(calc_res.get("answer_text", ""))
            if "table_markdown" in calc_res:
                st.markdown(calc_res["table_markdown"])

        # Charting Generator
        st.markdown("##### 📊 Interactive Chart Generator")
        num_cols = active_df.select_dtypes(include=['number']).columns.tolist()
        all_cols = list(active_df.columns)
        if len(all_cols) >= 2 and num_cols:
            c_col1, c_col2, c_col3 = st.columns(3)
            with c_col1:
                chart_x = st.selectbox("X-Axis (Category / Label)", options=all_cols, index=0)
            with c_col2:
                chart_y = st.selectbox("Y-Axis (Numeric Metric)", options=num_cols, index=0)
            with c_col3:
                chart_t = st.selectbox("Chart Type", options=["bar", "line", "scatter"])

            if st.button("📈 Render Chart"):
                fig = ExcelIntelligenceEngine.generate_chart_figure(
                    df=active_df,
                    x_col=chart_x,
                    y_col=chart_y,
                    chart_type=chart_t
                )
                st.pyplot(fig)


# =====================================================================
# TAB 5: DOCUMENT COMPARISON
# =====================================================================
with tools_tabs[4]:
    st.subheader("⚖️ Document Comparison & Difference Detector (Feature #21)")
    st.markdown("Identify added, removed, and modified sections, plus changed dollar amounts and dates.")

    all_doc_names = list(doc_name_to_path.keys())
    if len(all_doc_names) < 2:
        st.info("You need at least two documents in your library to perform a comparison. Upload files in Documents!")
    else:
        cmp_col1, cmp_col2 = st.columns(2)
        with cmp_col1:
            doc_a = st.selectbox("Base Document (A)", options=all_doc_names, index=0)
        with cmp_col2:
            remaining = [d for d in all_doc_names if d != doc_a]
            doc_b = st.selectbox("Target Document (B)", options=remaining, index=0)

        if st.button("⚖️ Run Comparative Difference Analysis", type="primary", key="btn_run_compare"):
            with st.spinner("Analyzing structural, textual, and numerical differences..."):
                report = DocumentComparator.compare_documents(
                    file_path_a=doc_name_to_path[doc_a],
                    file_path_b=doc_name_to_path[doc_b]
                )
                st.session_state.comparison_report = report

        rep: ComparisonReport = st.session_state.get("comparison_report")
        if rep:
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            with stat_col1:
                st.metric("Semantic Similarity", f"{rep.similarity_score * 100:.1f}%")
            with stat_col2:
                st.metric("Lines Added", rep.total_added_lines)
            with stat_col3:
                st.metric("Lines Removed", rep.total_removed_lines)
            with stat_col4:
                st.metric("Changed Entities", len(rep.changed_entities))

            if rep.changed_entities:
                st.markdown("#### 🔢 Changed Amounts, Dates & Metrics")
                ent_rows = [{
                    "Entity Type": ent.entity_type.upper(),
                    "In Doc A": ent.old_value,
                    "In Doc B": ent.new_value,
                    "Context Reference": ent.context_new[:60]
                } for ent in rep.changed_entities]
                st.dataframe(pd.DataFrame(ent_rows), use_container_width=True)

            st.markdown("#### 📝 Markdown Difference Report")
            st.markdown(rep.to_markdown())
            st.download_button(
                label="📥 Download Comparison Report (Markdown)",
                data=rep.to_markdown(),
                file_name=f"comparison_{doc_a}_vs_{doc_b}.md",
                mime="text/markdown"
            )


# =====================================================================
# TAB 6: DOCUMENT SUMMARIZER
# =====================================================================
with tools_tabs[5]:
    st.subheader("📝 Hierarchical Document Summarization (Feature #22)")
    st.markdown("Generate short, detailed, or executive summaries for documents exceeding context windows.")

    if not all_doc_names:
        st.info("No documents uploaded yet. Upload a document in Documents to summarize!")
    else:
        sum_doc = st.selectbox("Select Document to Summarize", options=all_doc_names, key="sum_doc_sel")
        
        sum_col1, sum_col2, sum_col3 = st.columns(3)
        with sum_col1:
            sum_style = st.selectbox("Summary Mode", options=["detailed", "short", "executive", "section_wise"])
        with sum_col2:
            sum_lang = st.selectbox("Output Language", options=["English", "Hindi", "Gujarati"])
        with sum_col3:
            enable_page_range = st.checkbox("Limit to Page Range", value=False)

        page_range_val = None
        if enable_page_range:
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                p_start = st.number_input("Start Page", min_value=1, value=1)
            with p_col2:
                p_end = st.number_input("End Page", min_value=1, value=5)
            page_range_val = (p_start, p_end)

        if st.button("📝 Generate Document Summary", type="primary", key="btn_run_summary"):
            with st.spinner("Generating document summary..."):
                sum_res = DocumentSummarizer.summarize_document(
                    file_path=doc_name_to_path[sum_doc],
                    style=sum_style,
                    page_range=page_range_val,
                    target_language=sum_lang
                )
                st.session_state.document_summary_result = sum_res

        s_res = st.session_state.get("document_summary_result")
        if s_res:
            st.markdown(f"### 📄 Summary: `{s_res.get('file_name', '')}`")
            st.caption(f"Mode: **{s_res.get('mode_used', '')}** | Language: **{s_res.get('target_language', '')}** | Pages: **{s_res.get('pages_analyzed', 0)}**")
            st.markdown(s_res.get("summary", ""))

            st.download_button(
                label="📥 Download Summary Report",
                data=s_res.get("summary", ""),
                file_name=f"summary_{sum_doc}.md",
                mime="text/markdown"
            )
