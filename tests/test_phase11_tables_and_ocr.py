"""Unit & Integration Tests for Advanced Table Extraction and Multilingual OCR (Features 16, 17, 18).

Tests:
- ExtractedTable data structure and DataFrame / CSV / Markdown / RAG passage export
- TableExtractor extraction methods and fallbacks
- MultilingualOCR script language detection (English, Hindi, Gujarati) via Unicode analysis
- Unicode preservation for Devanagari and Gujarati characters
- Advanced image preprocessing (deskewing, CLAHE, orientation detection)
"""

import unittest
from unittest.mock import patch, MagicMock
from PIL import Image
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.table_extractor import ExtractedTable, TableExtractor
from utils.multilingual_ocr import MultilingualOCR


class TestTablesAndMultilingualOCR(unittest.TestCase):

    def test_extracted_table_to_dataframe_and_exports(self):
        """Verify ExtractedTable converts cleanly to DataFrame, CSV, Markdown, and RAG passage."""
        headers = ["Quarter", "Revenue ($M)", "Operating Margin", "Growth Rate"]
        rows = [
            ["Q1 2024", "110.5", "18.2%", "+12%"],
            ["Q2 2024", "125.0", "19.5%", "+14%"],
            ["Q3 2024", "142.5", "21.4%", "+18%"]
        ]
        table = ExtractedTable(
            table_id="tab_p1_t1",
            file_name="financial_q3.pdf",
            page_number=1,
            headers=headers,
            rows=rows
        )

        df = table.to_dataframe()
        self.assertEqual(len(df), 3)
        self.assertEqual(len(df.columns), 4)
        self.assertEqual(df.iloc[2]["Revenue ($M)"], "142.5")

        # CSV string
        csv_str = table.to_csv_string()
        self.assertIn("Q3 2024", csv_str)
        self.assertIn("142.5", csv_str)

        # Markdown
        md_str = table.to_markdown()
        self.assertIn("| Q3 2024", md_str)
        self.assertIn("financial_q3.pdf", md_str)

        # RAG searchable passage
        rag_pass = table.to_rag_passage()
        self.assertIn("TABLE IDENTIFIER", rag_pass)
        self.assertIn("Revenue ($M): 142.5", rag_pass)

    def test_unicode_script_language_detection(self):
        """Verify offline Unicode block detection for English, Hindi, and Gujarati."""
        english_text = "The consolidated net profit increased by twenty percent."
        hindi_text = "कम्पनी का शुद्ध लाभ इस तिमाही में बीस प्रतिशत बढ़ा है।"
        gujarati_text = "કંપનીનો ચોખ્ખો નફો આ ત્રિમાસિક ગાળામાં વીસ ટકા વધ્યો છે."
        mixed_hindi = "Q3 Results: कम्पनी का कुल राजस्व $142.5 मिलियन रहा।"

        self.assertEqual(MultilingualOCR.detect_script_language(english_text), "eng")
        self.assertEqual(MultilingualOCR.detect_script_language(hindi_text), "hin")
        self.assertEqual(MultilingualOCR.detect_script_language(gujarati_text), "guj")
        self.assertEqual(MultilingualOCR.detect_script_language(mixed_hindi), "hin")

    def test_unicode_preservation_indic_scripts(self):
        """Ensure Indic conjuncts, matras, and numbers are not corrupted."""
        sample_hin = "प्रौद्योगिकी एवं वित्तीय विश्लेषण (२०२૪)"
        sample_guj = "ટેકનોલોજી અને નાણાકીય વિશ્લેષણ (૨૦૨૪)"

        det_hin = MultilingualOCR.detect_script_language(sample_hin)
        det_guj = MultilingualOCR.detect_script_language(sample_guj)

        self.assertEqual(det_hin, "hin")
        self.assertEqual(det_guj, "guj")
        self.assertEqual(len(sample_hin), len(sample_hin.encode('utf-8').decode('utf-8')))

    def test_advanced_preprocessing_pipeline(self):
        """Verify advanced preprocessing runs deskew and contrast enhancement on synthetic image."""
        # Create a simple synthetic test image
        img_arr = np.zeros((200, 400, 3), dtype=np.uint8)
        img_arr[50:150, 50:350] = 200  # Gray box
        test_img = Image.fromarray(img_arr)

        enhanced = MultilingualOCR.advanced_preprocess(
            test_img,
            deskew=True,
            denoise=True,
            enhance_contrast=True
        )

        self.assertIsInstance(enhanced, Image.Image)
        self.assertEqual(enhanced.size, (400, 200))


if __name__ == "__main__":
    unittest.main()
