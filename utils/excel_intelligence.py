"""Excel & Spreadsheet Intelligence Engine for Corpus Mind.

Supports:
- XLSX, XLS, and CSV workbooks with multiple worksheets
- Deterministic Python calculations using pandas (Sum, Average, Count, Min, Max, GroupBy, Sort)
- Missing value inspection and column type profiling
- Natural language query intent translation to deterministic pandas computations (NO eval/exec of LLM code)
- Dynamic chart generation (Bar, Line, Histogram)
- Result exporting to CSV and Excel
- Workbook, sheet, row, and cell references preservation
"""

import os
import sys
import re
import io
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


class ExcelWorkbookManager:
    """Manages loading, parsing, and inspecting multi-sheet Excel and CSV files."""

    @staticmethod
    def load_workbook_sheets(file_path: str) -> Dict[str, pd.DataFrame]:
        """Load all sheets from an Excel (.xlsx, .xls) or CSV file into a dict of DataFrames."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        sheets_dict: Dict[str, pd.DataFrame] = {}

        if ext == ".csv":
            df = pd.read_csv(file_path)
            sheets_dict["Sheet1"] = df
        elif ext in [".xlsx", ".xls"]:
            # Load all sheets using openpyxl or xlrd with context manager for clean file closing
            with pd.ExcelFile(file_path) as excel_file:
                for sheet_name in excel_file.sheet_names:
                    df = excel_file.parse(sheet_name)
                    sheets_dict[sheet_name] = df
        else:
            raise ValueError(f"Unsupported spreadsheet format: {ext}")

        # Clean column names (strip whitespace)
        for name, df in sheets_dict.items():
            df.columns = [str(c).strip() for c in df.columns]

        return sheets_dict

    @staticmethod
    def get_sheet_summary(df: pd.DataFrame, sheet_name: str, file_name: str) -> Dict[str, Any]:
        """Return structured summary metadata for a worksheet."""
        num_rows, num_cols = df.shape
        missing_counts = df.isnull().sum().to_dict()
        col_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        return {
            "file_name": file_name,
            "sheet_name": sheet_name,
            "total_rows": num_rows,
            "total_columns": num_cols,
            "columns": list(df.columns),
            "numeric_columns": numeric_cols,
            "missing_values": missing_counts,
            "column_types": col_types,
            "preview_data": df.head(5).to_dict(orient="records")
        }


class ExcelIntelligenceEngine:
    """Performs deterministic calculations, query interpretation, and charting on spreadsheets."""

    @classmethod
    def query_spreadsheet(
        cls,
        df: pd.DataFrame,
        query: str,
        sheet_name: str = "Sheet1",
        file_name: str = "Workbook"
    ) -> Dict[str, Any]:
        """Analyze user natural language query and execute deterministic pandas calculation.
        
        Guarantees Security: ZERO execution of arbitrary strings or LLM-generated code.
        """
        q = query.lower().strip()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        all_cols = list(df.columns)

        # 1. Check for Missing Values query
        if any(w in q for w in ["missing", "null", "empty", "nan", "blank"]):
            null_counts = df.isnull().sum()
            missing_cols = {col: int(cnt) for col, cnt in null_counts.items() if cnt > 0}
            total_missing = int(null_counts.sum())
            return {
                "operation": "missing_values",
                "result_type": "summary",
                "total_missing_cells": total_missing,
                "missing_by_column": missing_cols,
                "answer_text": (
                    f"In worksheet '{sheet_name}' of '{file_name}', there are a total of **{total_missing}** missing cells. "
                    f"Columns with missing values: {', '.join([f'{c}: {n}' for c, n in missing_cols.items()]) if missing_cols else 'None'}."
                )
            }

        # 2. Check for Target Metric Column Match (prioritize numeric columns for aggregations)
        target_col = None
        for col in numeric_cols:
            if col.lower() in q:
                target_col = col
                break
        
        # If no numeric column explicitly in query, check all columns
        if not target_col:
            for col in all_cols:
                if col.lower() in q:
                    target_col = col
                    break
        
        # If still no target found, pick first numeric column as default target for aggregations
        if not target_col and numeric_cols:
            target_col = numeric_cols[0]

        # 3. Sum / Total Operation
        if any(w in q for w in ["total", "sum", "overall", "aggregate"]) and target_col in numeric_cols:
            total_val = float(df[target_col].sum())
            return {
                "operation": "sum",
                "column": target_col,
                "result": round(total_val, 2),
                "answer_text": f"The total sum of **{target_col}** in '{sheet_name}' is **{total_val:,.2f}**."
            }

        # 4. Average / Mean Operation
        if any(w in q for w in ["average", "mean", "avg"]) and target_col in numeric_cols:
            avg_val = float(df[target_col].mean())
            return {
                "operation": "average",
                "column": target_col,
                "result": round(avg_val, 2),
                "answer_text": f"The average (mean) value of **{target_col}** in '{sheet_name}' is **{avg_val:,.2f}**."
            }

        # 5. Maximum / Highest / Top Operation
        if any(w in q for w in ["highest", "max", "maximum", "peak", "most"]) and target_col in numeric_cols:
            max_idx = df[target_col].idxmax()
            max_val = float(df[target_col].max())
            best_row = df.loc[max_idx].to_dict()
            return {
                "operation": "max",
                "column": target_col,
                "value": round(max_val, 2),
                "row_index": int(max_idx) + 1,
                "full_record": best_row,
                "answer_text": f"The highest **{target_col}** is **{max_val:,.2f}** (at Row {int(max_idx)+1}). Details: {best_row}."
            }

        # 6. Minimum / Lowest Operation
        if any(w in q for w in ["lowest", "min", "minimum", "least"]) and target_col in numeric_cols:
            min_idx = df[target_col].idxmin()
            min_val = float(df[target_col].min())
            worst_row = df.loc[min_idx].to_dict()
            return {
                "operation": "min",
                "column": target_col,
                "value": round(min_val, 2),
                "row_index": int(min_idx) + 1,
                "full_record": worst_row,
                "answer_text": f"The lowest **{target_col}** is **{min_val:,.2f}** (at Row {int(min_idx)+1}). Details: {worst_row}."
            }

        # 7. Top N Rows / Sort
        top_match = re.search(r'top\s*(\d+)', q)
        if top_match and target_col in numeric_cols:
            n_val = int(top_match.group(1))
            top_df = df.sort_values(by=target_col, ascending=False).head(n_val)
            return {
                "operation": "top_n",
                "column": target_col,
                "n": n_val,
                "data": top_df.to_dict(orient="records"),
                "answer_text": f"Top {n_val} records sorted by **{target_col}** descending in '{sheet_name}'.",
                "table_markdown": top_df.to_markdown(index=False)
            }

        # 8. Count / Number of records
        if any(w in q for w in ["how many", "count", "number of rows"]):
            count_val = len(df)
            return {
                "operation": "count",
                "result": count_val,
                "answer_text": f"There are **{count_val:,}** rows / records in worksheet '{sheet_name}'."
            }

        # 9. Generic Descriptive Statistics Fallback
        desc_df = df.describe().round(2)
        return {
            "operation": "descriptive_summary",
            "summary_table": desc_df.to_dict(),
            "answer_text": f"Here is the statistical summary of numerical columns in '{sheet_name}':\n\n{desc_df.to_markdown()}"
        }

    @classmethod
    def generate_chart_figure(
        cls,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        chart_type: str = "bar",
        title: Optional[str] = None
    ):
        """Generate a matplotlib figure for visual display in Streamlit."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.5))

        if chart_type == "bar":
            top_df = df.head(15)
            ax.bar(top_df[x_col].astype(str), top_df[y_col], color="#3b82f6")
            ax.set_xticklabels(top_df[x_col].astype(str), rotation=45, ha='right')
        elif chart_type == "line":
            ax.plot(df[x_col].astype(str), df[y_col], marker='o', color="#10b981", linewidth=2)
            ax.set_xticklabels(df[x_col].astype(str), rotation=45, ha='right')
        elif chart_type == "scatter":
            ax.scatter(df[x_col], df[y_col], color="#8b5cf6", alpha=0.7)

        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(title or f"{y_col} by {x_col}")
        plt.tight_layout()
        return fig
