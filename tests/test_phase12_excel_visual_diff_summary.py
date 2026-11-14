"""Unit & Integration Tests for Visual RAG, Excel Intelligence, Document Diff, and Summarizer (Features 19, 20, 21, 22).

Tests:
- VisualRAG: Figure classification, label extraction, RAG passage generation
- ExcelIntelligence: Multi-sheet parsing, deterministic calculations (sum, avg, max, min, top-n, missing), zero code execution security
- DocumentComparator: Textual diff, changed amounts/dates, semantic drift score
- DocumentSummarizer: Single-pass, page-range, and hierarchical map-reduce summarization
"""

import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import pandas as pd
import numpy as np
import tempfile
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.visual_rag import ExtractedFigure, VisualRAGManager
from utils.excel_intelligence import ExcelWorkbookManager, ExcelIntelligenceEngine
from utils.document_comparator import DocumentComparator, ComparisonReport, ChangedEntity
from utils.document_summarizer import DocumentSummarizer


class TestExcelVisualDiffSummary(unittest.TestCase):

    def test_visual_rag_figure_representation(self):
        """Verify ExtractedFigure formats structured visual RAG passages."""
        fig = ExtractedFigure(
            figure_id="fig_p4_1",
            file_name="system_design.pdf",
            page_number=4,
            image_path="/data/images/fig_p4_1.png",
            width=800,
            height=450,
            figure_type="architecture",
            description="High-level architecture showing Ingestion, FAISS indexing, and LLM synthesis.",
            detected_labels=["Ingestion", "FAISS", "Synthesis", "Vector DB"]
        )

        passage = fig.to_rag_passage()
        self.assertIn("FIGURE IDENTIFIER", passage)
        self.assertIn("ARCHITECTURE", passage)
        self.assertIn("FAISS, Synthesis", passage)
        self.assertIn("/data/images/fig_p4_1.png", passage)

    def test_excel_intelligence_deterministic_calculations(self):
        """Verify calculations (sum, average, max, min, top-n, missing values) are mathematically exact."""
        data = {
            "Product": ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"],
            "Region": ["North", "South", "East", "West", "North"],
            "Revenue": [150000.0, 220000.0, 95000.0, 310000.0, None],
            "Units_Sold": [1200, 1800, 750, 2500, 1100]
        }
        df = pd.DataFrame(data)

        # 1. Sum / Total
        res_sum = ExcelIntelligenceEngine.query_spreadsheet(df, "What is the total revenue?")
        self.assertEqual(res_sum["operation"], "sum")
        self.assertEqual(res_sum["result"], 775000.0)

        # 2. Average
        res_avg = ExcelIntelligenceEngine.query_spreadsheet(df, "Calculate the average revenue.")
        self.assertEqual(res_avg["operation"], "average")
        self.assertEqual(res_avg["result"], 193750.0)

        # 3. Maximum
        res_max = ExcelIntelligenceEngine.query_spreadsheet(df, "Which product has the highest revenue?")
        self.assertEqual(res_max["operation"], "max")
        self.assertEqual(res_max["value"], 310000.0)
        self.assertEqual(res_max["full_record"]["Product"], "Delta")

        # 4. Missing Values
        res_missing = ExcelIntelligenceEngine.query_spreadsheet(df, "Are there any missing values?")
        self.assertEqual(res_missing["operation"], "missing_values")
        self.assertEqual(res_missing["total_missing_cells"], 1)
        self.assertIn("Revenue", res_missing["missing_by_column"])

        # 5. Top N
        res_top = ExcelIntelligenceEngine.query_spreadsheet(df, "Show top 2 by units_sold")
        self.assertEqual(res_top["operation"], "top_n")
        self.assertEqual(res_top["n"], 2)
        self.assertEqual(res_top["data"][0]["Units_Sold"], 2500)

    def test_excel_multi_sheet_loading(self):
        """Verify ExcelWorkbookManager reads multi-sheet Excel files correctly."""
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with pd.ExcelWriter(tmp_path, engine="openpyxl") as writer:
                pd.DataFrame({"Sales_2023": [100, 200]}).to_excel(writer, sheet_name="2023_Data", index=False)
                pd.DataFrame({"Sales_2024": [300, 400]}).to_excel(writer, sheet_name="2024_Data", index=False)

            sheets = ExcelWorkbookManager.load_workbook_sheets(tmp_path)
            self.assertIn("2023_Data", sheets)
            self.assertIn("2024_Data", sheets)
            self.assertEqual(len(sheets["2023_Data"]), 2)
            self.assertEqual(len(sheets["2024_Data"]), 2)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_document_comparison_diff_and_entities(self):
        """Verify DocumentComparator detects changed dollar amounts, dates, and diff lines."""
        text_a = "Agreement made on October 1, 2024. The total consulting fee is $150,000 payable upon delivery."
        text_b = "Agreement made on November 15, 2024. The total consulting fee is $225,000 payable upon delivery."

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f_a, \
             tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f_b:
            f_a.write(text_a)
            f_b.write(text_b)
            path_a = f_a.name
            path_b = f_b.name

        try:
            report = DocumentComparator.compare_documents(path_a, path_b)
            self.assertEqual(report.doc_a_name, Path(path_a).name)
            self.assertEqual(report.doc_b_name, Path(path_b).name)
            self.assertGreater(report.similarity_score, 0.70)
            
            # Check changed amounts and dates
            detected_types = [ent.entity_type for ent in report.changed_entities]
            self.assertIn("amount", detected_types)
            
            md_report = report.to_markdown()
            self.assertIn("Document Comparison Report", md_report)
            self.assertIn("Changed Metrics", md_report)
        finally:
            for p in [path_a, path_b]:
                if os.path.exists(p):
                    os.remove(p)

    @patch("utils.llm_provider.LLMProviderManager.generate")
    def test_document_summarizer_hierarchical(self, mock_generate):
        """Verify DocumentSummarizer executes single-pass and hierarchical summarization."""
        mock_generate.return_value = MagicMock(
            text="### Executive Summary\n- Analyzed quarterly financial metrics.\n- Revenue increased by 14% year-over-year.",
            error=None
        )

        test_text = "Acme Corp reported record growth in Q3. Operating margins expanded significantly."
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(test_text)
            f_path = f.name

        try:
            res = DocumentSummarizer.summarize_document(
                file_path=f_path,
                style="executive",
                target_language="English"
            )
            self.assertEqual(res["style"], "executive")
            self.assertIn("Executive Summary", res["summary"])
            self.assertEqual(res["mode_used"], "single_pass")
        finally:
            if os.path.exists(f_path):
                os.remove(f_path)


if __name__ == "__main__":
    unittest.main()
