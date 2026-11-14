"""Advanced PDF Table Extraction Engine for Corpus Mind.

Supports:
- Digital & Scanned PDF table detection
- Multiple tables per page with reading order preservation
- Merged cell handling and irregular/borderless table extraction
- Multi-engine extraction: pdfplumber -> PyMuPDF find_tables() -> regex text layout fallback
- Structured representation with CSV, Excel, and Markdown export
- RAG integration: converts tables to rich searchable text passages for FAISS & BM25
"""

import os
import sys
import re
import io
import csv
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.ocr import OCRManager

logger = logging.getLogger(__name__)


@dataclass
class ExtractedTable:
    table_id: str
    file_name: str
    page_number: int
    headers: List[str]
    rows: List[List[str]]
    bbox: Optional[Tuple[float, float, float, float]] = None
    confidence: float = 1.0
    extraction_method: str = "pdfplumber"
    is_scanned: bool = False

    def to_dataframe(self) -> pd.DataFrame:
        """Convert extracted table to pandas DataFrame."""
        if not self.headers and not self.rows:
            return pd.DataFrame()
        
        # If headers are missing, synthesize col1, col2, ...
        if not self.headers and self.rows:
            num_cols = max(len(r) for r in self.rows)
            headers = [f"Column_{i+1}" for i in range(num_cols)]
            rows = self.rows
        else:
            headers = self.headers
            rows = self.rows

        # Normalize row lengths to match header length
        normalized_rows = []
        for r in rows:
            if len(r) < len(headers):
                normalized_rows.append(r + [""] * (len(headers) - len(r)))
            elif len(r) > len(headers):
                normalized_rows.append(r[:len(headers)])
            else:
                normalized_rows.append(r)

        return pd.DataFrame(normalized_rows, columns=headers)

    def to_csv_string(self) -> str:
        """Export table to CSV text string."""
        df = self.to_dataframe()
        return df.to_csv(index=False)

    def to_markdown(self) -> str:
        """Export table to readable markdown table string with metadata."""
        df = self.to_dataframe()
        if df.empty:
            return f"*(Table {self.table_id} on Page {self.page_number} is empty)*"
        
        meta_str = f"**Table {self.table_id} (Page {self.page_number}, {self.file_name}):**\n"
        return meta_str + df.to_markdown(index=False)

    def to_rag_passage(self) -> str:
        """Format table into a high-saliency searchable text passage for RAG indexing."""
        df = self.to_dataframe()
        if df.empty:
            return ""

        passage_lines = [
            f"TABLE IDENTIFIER: [{self.file_name}, p. {self.page_number}, Table {self.table_id}]",
            f"HEADERS: {' | '.join(str(h) for h in df.columns)}"
        ]
        
        # Include row-by-row key-value representations for high semantic retrieval match
        for idx, row in df.iterrows():
            row_items = [f"{col}: {val}" for col, val in row.items() if str(val).strip()]
            if row_items:
                passage_lines.append(f"Row {idx+1}: {', '.join(row_items)}")

        # Append markdown view
        passage_lines.append("\nMarkdown View:\n" + df.to_markdown(index=False))
        return "\n".join(passage_lines)


class TableExtractor:
    """Robust multi-engine table detector and extractor for PDF documents."""

    @classmethod
    def extract_tables_from_pdf(cls, file_path: str) -> List[ExtractedTable]:
        """Extract all tables across all pages of a PDF using fallback strategies."""
        path_obj = Path(file_path)
        if not path_obj.exists():
            logger.error(f"File not found: {file_path}")
            return []

        all_tables: List[ExtractedTable] = []

        # 1. Primary Strategy: pdfplumber (best for table borders and cell alignments)
        try:
            tables_plumb = cls._extract_with_pdfplumber(file_path)
            if tables_plumb:
                all_tables.extend(tables_plumb)
                logger.info(f"pdfplumber extracted {len(tables_plumb)} tables from {path_obj.name}")
        except Exception as e:
            logger.warning(f"pdfplumber extraction error on {path_obj.name}: {e}")

        # 2. Secondary Strategy: PyMuPDF find_tables() (for pages where pdfplumber found nothing)
        pages_with_tables = {t.page_number for t in all_tables}
        try:
            tables_mupdf = cls._extract_with_pymupdf(file_path, exclude_pages=pages_with_tables)
            if tables_mupdf:
                all_tables.extend(tables_mupdf)
                logger.info(f"PyMuPDF extracted {len(tables_mupdf)} additional tables from {path_obj.name}")
        except Exception as e:
            logger.warning(f"PyMuPDF table extraction failed on {path_obj.name}: {e}")

        # 3. Tertiary Strategy: Scanned pages table OCR detection
        try:
            pages_now = {t.page_number for t in all_tables}
            tables_scanned = cls._extract_scanned_tables(file_path, exclude_pages=pages_now)
            if tables_scanned:
                all_tables.extend(tables_scanned)
                logger.info(f"OCR extracted {len(tables_scanned)} scanned tables from {path_obj.name}")
        except Exception as e:
            logger.warning(f"Scanned table extraction error on {path_obj.name}: {e}")

        # Sort tables by page number and table_id
        all_tables.sort(key=lambda t: (t.page_number, t.table_id))
        return all_tables

    @classmethod
    def _extract_with_pdfplumber(cls, file_path: str) -> List[ExtractedTable]:
        """Extract tables using pdfplumber with multiple extraction strategies."""
        import pdfplumber
        extracted = []
        file_name = Path(file_path).name

        with pdfplumber.open(file_path) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                # Try default line-based detection
                tables = page.extract_tables()
                if not tables:
                    # Fallback to text-based layout settings for borderless tables
                    settings = {
                        "vertical_strategy": "text",
                        "horizontal_strategy": "text",
                        "snap_tolerance": 4,
                        "join_tolerance": 4
                    }
                    tables = page.extract_tables(table_settings=settings)

                for t_idx, raw_table in enumerate(tables, start=1):
                    if not raw_table or len(raw_table) < 2:
                        continue

                    # Clean whitespace and handle None cells
                    cleaned_rows = [
                        [str(cell).strip() if cell is not None else "" for cell in row]
                        for row in raw_table
                    ]
                    # Filter out completely empty rows
                    cleaned_rows = [r for r in cleaned_rows if any(r)]
                    if len(cleaned_rows) < 2:
                        continue

                    # First row as header, remaining as data rows
                    headers = cleaned_rows[0]
                    data_rows = cleaned_rows[1:]

                    table_id = f"table_p{page_idx}_t{t_idx}"
                    extracted.append(ExtractedTable(
                        table_id=table_id,
                        file_name=file_name,
                        page_number=page_idx,
                        headers=headers,
                        rows=data_rows,
                        extraction_method="pdfplumber",
                        is_scanned=False
                    ))
        return extracted

    @classmethod
    def _extract_with_pymupdf(cls, file_path: str, exclude_pages: set) -> List[ExtractedTable]:
        """Extract tables using PyMuPDF find_tables() API."""
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz

        extracted = []
        file_name = Path(file_path).name
        doc = fitz.open(file_path)

        for page_num in range(len(doc)):
            page_idx = page_num + 1
            if page_idx in exclude_pages:
                continue

            page = doc[page_num]
            # Use PyMuPDF table finder
            try:
                tabs = page.find_tables()
                for t_idx, tab in enumerate(tabs.tables, start=1):
                    df = tab.to_pandas()
                    if df.empty or len(df) == 0:
                        continue

                    headers = [str(c) for c in df.columns]
                    rows = [[str(val) for val in r] for r in df.values]
                    table_id = f"mupdf_p{page_idx}_t{t_idx}"

                    extracted.append(ExtractedTable(
                        table_id=table_id,
                        file_name=file_name,
                        page_number=page_idx,
                        headers=headers,
                        rows=rows,
                        bbox=tab.bbox,
                        extraction_method="pymupdf_find_tables",
                        is_scanned=False
                    ))
            except Exception:
                pass
        return extracted

    @classmethod
    def _extract_scanned_tables(cls, file_path: str, exclude_pages: set) -> List[ExtractedTable]:
        """Detect and extract grid tables on scanned pages using PyMuPDF pixmaps + OCR."""
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz

        extracted = []
        file_name = Path(file_path).name
        doc = fitz.open(file_path)

        for page_num in range(len(doc)):
            page_idx = page_num + 1
            if page_idx in exclude_pages:
                continue

            page = doc[page_num]
            native_text = page.get_text("text") or ""
            # If native text is very low, it's likely a scan
            if len(re.findall(r'\w+', native_text)) < 25:
                # Render page image
                pix = page.get_pixmap(dpi=150)
                from PIL import Image
                pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # Run OCR with TSV output to find structured word boxes
                try:
                    import pytesseract
                    tsv_data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DATAFRAME)
                    tsv_data = tsv_data[tsv_data.text.notnull() & (tsv_data.text.str.strip() != "")]
                    
                    if len(tsv_data) > 10:
                        # Group text by line_num
                        lines = []
                        for _, line_group in tsv_data.groupby(['block_num', 'par_num', 'line_num']):
                            line_words = line_group.sort_values('left')['text'].tolist()
                            if len(line_words) >= 2:
                                lines.append(line_words)

                        if len(lines) >= 3:
                            headers = lines[0]
                            rows = lines[1:]
                            table_id = f"ocr_table_p{page_idx}_t1"
                            extracted.append(ExtractedTable(
                                table_id=table_id,
                                file_name=file_name,
                                page_number=page_idx,
                                headers=headers,
                                rows=rows,
                                confidence=0.75,
                                extraction_method="ocr_grid_scan",
                                is_scanned=True
                            ))
                except Exception as ocr_err:
                    logger.debug(f"OCR table parse failed on page {page_idx}: {ocr_err}")

        return extracted

    @classmethod
    def export_all_tables_to_excel(cls, tables: List[ExtractedTable], output_path: str):
        """Export a list of ExtractedTable instances into a multi-sheet Excel file."""
        if not tables:
            return
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            for idx, table in enumerate(tables, start=1):
                df = table.to_dataframe()
                sheet_name = f"P{table.page_number}_T{idx}"[:31]  # Excel max sheet name is 31 chars
                df.to_excel(writer, sheet_name=sheet_name, index=False)
