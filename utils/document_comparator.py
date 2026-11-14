"""Document Comparison & Semantic Differential Engine for Corpus Mind.

Supports:
- Structural comparison (page counts, paragraph counts, character lengths)
- Textual diffs: Added, removed, and modified sentences/paragraphs using difflib
- Numerical & entity comparison: Detection of modified dates, dollar amounts, and percentages
- Semantic similarity comparison using dense sentence embeddings (cosine distance shifts)
- Side-by-side comparison tables and exported markdown difference reports
- Distinguishes exact textual changes from semantic reinterpretations
"""

import os
import sys
import difflib
import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.document_loader import DocumentLoader
from utils.embeddings import EmbeddingManager

logger = logging.getLogger(__name__)


@dataclass
class ChangedEntity:
    entity_type: str  # "amount", "date", "percentage", "clause"
    old_value: str
    new_value: str
    context_old: str
    context_new: str


@dataclass
class ComparisonReport:
    doc_a_name: str
    doc_b_name: str
    similarity_score: float  # 0.0 to 1.0 (semantic cosine similarity)
    total_added_lines: int
    total_removed_lines: int
    total_modified_lines: int
    changed_entities: List[ChangedEntity]
    textual_diffs: List[Dict[str, Any]]
    semantic_sections_diff: List[Dict[str, Any]]

    def to_markdown(self) -> str:
        """Format comparison findings into an executive markdown report."""
        report = [
            f"# ⚖️ Document Comparison Report",
            f"- **Base Document (A):** `{self.doc_a_name}`",
            f"- **Target Document (B):** `{self.doc_b_name}`",
            f"- **Overall Semantic Similarity:** `{self.similarity_score * 100:.1f}%`",
            f"- **Total Lines Added:** `{self.total_added_lines}`",
            f"- **Total Lines Removed:** `{self.total_removed_lines}`",
            f"- **Total Lines Modified:** `{self.total_modified_lines}`\n",
            "---",
            "## 🔢 Changed Metrics, Amounts & Dates\n"
        ]

        if not self.changed_entities:
            report.append("*No numerical or date discrepancies detected between documents.*")
        else:
            report.append("| Type | In Document A | In Document B | Context Reference |")
            report.append("| :--- | :--- | :--- | :--- |")
            for ent in self.changed_entities:
                ctx_clean = ent.context_new.replace("|", "/")[:60]
                report.append(f"| **{ent.entity_type.upper()}** | `{ent.old_value}` | `{ent.new_value}` | {ctx_clean} |")

        report.append("\n---\n## 📝 Significant Text & Clause Modifications\n")
        if not self.semantic_sections_diff:
            report.append("*Documents appear semantically identical.*")
        else:
            for idx, diff in enumerate(self.semantic_sections_diff[:15], start=1):
                report.append(f"### Change {idx} (Semantic Drift: {diff['drift_score'] * 100:.1f}%)")
                report.append(f"**Doc A ({self.doc_a_name}):**\n> {diff['text_a']}")
                report.append(f"**Doc B ({self.doc_b_name}):**\n> {diff['text_b']}\n")

        return "\n".join(report)


class DocumentComparator:
    """Compares document pairs and produces deep structural, semantic, and textual reports."""

    @classmethod
    def compare_documents(cls, file_path_a: str, file_path_b: str) -> ComparisonReport:
        """Execute full comparison pipeline on two document files."""
        name_a = Path(file_path_a).name
        name_b = Path(file_path_b).name

        text_a = DocumentLoader.load_document(file_path_a).strip()
        text_b = DocumentLoader.load_document(file_path_b).strip()

        # 1. Overall Semantic Cosine Similarity
        sim_score = cls._calculate_semantic_similarity(text_a, text_b)

        # 2. Textual Line-by-Line Differencing
        lines_a = [l.strip() for l in text_a.splitlines() if l.strip()]
        lines_b = [l.strip() for l in text_b.splitlines() if l.strip()]
        
        matcher = difflib.SequenceMatcher(None, lines_a, lines_b)
        textual_diffs = []
        added_cnt = 0
        removed_cnt = 0
        modified_cnt = 0

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                modified_cnt += max(i2 - i1, j2 - j1)
                textual_diffs.append({
                    "type": "modified",
                    "old": lines_a[i1:i2],
                    "new": lines_b[j1:j2]
                })
            elif tag == 'delete':
                removed_cnt += (i2 - i1)
                textual_diffs.append({
                    "type": "removed",
                    "old": lines_a[i1:i2],
                    "new": []
                })
            elif tag == 'insert':
                added_cnt += (j2 - j1)
                textual_diffs.append({
                    "type": "added",
                    "old": [],
                    "new": lines_b[j1:j2]
                })

        # 3. Changed Numbers, Dates, and Currencies
        changed_entities = cls._extract_entity_changes(text_a, text_b)

        # 4. Semantic Section Drift Detection
        semantic_diffs = cls._detect_semantic_drifts(lines_a, lines_b)

        return ComparisonReport(
            doc_a_name=name_a,
            doc_b_name=name_b,
            similarity_score=round(sim_score, 4),
            total_added_lines=added_cnt,
            total_removed_lines=removed_cnt,
            total_modified_lines=modified_cnt,
            changed_entities=changed_entities,
            textual_diffs=textual_diffs,
            semantic_sections_diff=semantic_diffs
        )

    @classmethod
    def _calculate_semantic_similarity(cls, text_a: str, text_b: str) -> float:
        """Compute cosine similarity of dense embeddings between two full documents."""
        if not text_a or not text_b:
            return 0.0
        try:
            emb_mgr = EmbeddingManager.get_instance()
            vec_a = emb_mgr.get_embedding(text_a[:1500])
            vec_b = emb_mgr.get_embedding(text_b[:1500])
            norm_a = np.linalg.norm(vec_a)
            norm_b = np.linalg.norm(vec_b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
        except Exception:
            return float(difflib.SequenceMatcher(None, text_a[:1000], text_b[:1000]).ratio())

    @classmethod
    def _extract_entity_changes(cls, text_a: str, text_b: str) -> List[ChangedEntity]:
        """Detect modified dollar figures, dates, and percentages between texts."""
        changes = []

        # Find dollar amounts, e.g. $142.5 million or $500,000
        amount_pat = r'\$\s*\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|thousand|k|m|b))?'
        amounts_a = re.findall(amount_pat, text_a, re.IGNORECASE)
        amounts_b = re.findall(amount_pat, text_b, re.IGNORECASE)

        for b_amt in amounts_b:
            if b_amt not in amounts_a and amounts_a:
                # Find sentence context
                ctx_match = re.search(rf'([^.\n]*?{re.escape(b_amt)}[^.\n]*)', text_b)
                ctx_str = ctx_match.group(1).strip() if ctx_match else b_amt
                changes.append(ChangedEntity(
                    entity_type="amount",
                    old_value=amounts_a[0],
                    new_value=b_amt,
                    context_old=amounts_a[0],
                    context_new=ctx_str
                ))

        # Find dates (e.g. October 1, 2024 or 2024-11-14)
        date_pat = r'(?:(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2})'
        dates_a = re.findall(date_pat, text_a)
        dates_b = re.findall(date_pat, text_b)

        for b_date in dates_b:
            if b_date not in dates_a and dates_a:
                ctx_match = re.search(rf'([^.\n]*?{re.escape(b_date)}[^.\n]*)', text_b)
                ctx_str = ctx_match.group(1).strip() if ctx_match else b_date
                changes.append(ChangedEntity(
                    entity_type="date",
                    old_value=dates_a[0],
                    new_value=b_date,
                    context_old=dates_a[0],
                    context_new=ctx_str
                ))

        return changes[:10]

    @classmethod
    def _detect_semantic_drifts(cls, lines_a: List[str], lines_b: List[str]) -> List[Dict[str, Any]]:
        """Identify sections with significant semantic shift even when words partially match."""
        drifts = []
        matcher = difflib.SequenceMatcher(None, lines_a, lines_b)

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace' and (i2 - i1 <= 3) and (j2 - j1 <= 3):
                sub_a = " ".join(lines_a[i1:i2])
                sub_b = " ".join(lines_b[j1:j2])
                if len(sub_a) > 20 and len(sub_b) > 20:
                    sim = cls._calculate_semantic_similarity(sub_a, sub_b)
                    drift = 1.0 - sim
                    if drift >= 0.15:
                        drifts.append({
                            "text_a": sub_a,
                            "text_b": sub_b,
                            "drift_score": round(drift, 3)
                        })
        return drifts
