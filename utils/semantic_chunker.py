import os
import sys
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.embeddings import EmbeddingManager

logger = logging.getLogger(__name__)


class SemanticChunker:
    """Advanced semantic document chunker.
    
    Splits text by semantic boundaries computed via adjacent sentence embedding
    distance, preserving markdown headings, tables, code blocks, and section continuity.
    """

    def __init__(
        self,
        similarity_threshold: Optional[float] = None,
        min_chunk_size: Optional[int] = None,
        max_chunk_size: Optional[int] = None,
        embedding_model: Optional[str] = None
    ):
        self.similarity_threshold = (
            similarity_threshold if similarity_threshold is not None 
            else getattr(config, "SEMANTIC_SIMILARITY_THRESHOLD", 0.65)
        )
        self.min_chunk_size = (
            min_chunk_size if min_chunk_size is not None 
            else getattr(config, "SEMANTIC_MIN_CHUNK_SIZE", 150)
        )
        self.max_chunk_size = (
            max_chunk_size if max_chunk_size is not None 
            else getattr(config, "SEMANTIC_MAX_CHUNK_SIZE", 800)
        )
        self.embedding_model = embedding_model or getattr(config, "DEFAULT_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    @staticmethod
    def _is_table_or_code_block(block: str) -> bool:
        """Detect markdown tables, code fences, or structured data."""
        stripped = block.strip()
        if stripped.startswith("```") or stripped.endswith("```"):
            return True
        lines = stripped.split("\n")
        table_lines = sum(1 for line in lines if line.strip().startswith("|") and line.strip().endswith("|"))
        return table_lines >= 2

    ABBREVIATIONS = {"mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "vs.", "etc.", "fig.", "e.g.", "i.e."}

    def _split_into_sentences_and_blocks(self, text: str) -> List[str]:
        """Split text into sentences while keeping markdown tables, headings, and code intact."""
        if not text or not text.strip():
            return []

        # Split into initial paragraphs
        raw_paragraphs = re.split(r'\n{2,}', text.strip())
        units: List[str] = []

        for para in raw_paragraphs:
            para = para.strip()
            if not para:
                continue

            # Keep tables and code blocks whole
            if self._is_table_or_code_block(para):
                units.append(para)
                continue

            # Split paragraph into sentences respecting abbreviations without look-behind errors
            raw_sentences = re.split(r'(?<=[.?!])\s+', para)
            merged: List[str] = []
            for s in raw_sentences:
                s_clean = s.strip()
                if not s_clean:
                    continue
                if merged:
                    last_tokens = merged[-1].split()
                    last_word = last_tokens[-1].lower() if last_tokens else ""
                    if last_word in self.ABBREVIATIONS or (len(last_word) == 2 and last_word.endswith('.')):
                        merged[-1] = merged[-1] + " " + s_clean
                        continue
                merged.append(s_clean)

            units.extend(merged)

        return units

    def split_text(self, text: str) -> List[str]:
        """Semantically chunk text based on cosine distance between adjacent sentence units.
        
        Falls back gracefully to recursive splitting on model errors or empty text.
        """
        if not text or not text.strip():
            return []

        units = self._split_into_sentences_and_blocks(text)
        if not units:
            return []

        # If total text length is already within max chunk size, return as a single chunk
        if len(text.strip()) <= self.max_chunk_size:
            return [text.strip()]

        if len(units) <= 1:
            return [units[0]]

        try:
            # Embed sentence units in batch
            embeddings = EmbeddingManager.get_embeddings(units, model_name=self.embedding_model)
            if embeddings.shape[0] != len(units):
                raise ValueError("Embedding count mismatch with sentence units.")

            # Normalize embeddings for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            norm_emb = embeddings / norms

            # Compute adjacent sentence cosine similarities
            # sim[i] = similarity between unit[i] and unit[i+1]
            sims = np.sum(norm_emb[:-1] * norm_emb[1:], axis=1)

            chunks: List[str] = []
            current_chunk_units: List[str] = [units[0]]
            current_len = len(units[0])

            for i in range(len(sims)):
                sim = float(sims[i])
                next_unit = units[i + 1]
                next_len = len(next_unit)

                # Boundary condition:
                # 1. Similarity dropped below threshold AND current chunk has met minimum size, OR
                # 2. Adding the next unit would strictly exceed max_chunk_size
                should_split = False
                if (current_len + next_len > self.max_chunk_size) and current_len >= self.min_chunk_size:
                    should_split = True
                elif sim < self.similarity_threshold and current_len >= self.min_chunk_size:
                    should_split = True

                if should_split:
                    chunks.append(" ".join(current_chunk_units).strip())
                    current_chunk_units = [next_unit]
                    current_len = next_len
                else:
                    current_chunk_units.append(next_unit)
                    current_len += next_len + 1

            if current_chunk_units:
                chunks.append(" ".join(current_chunk_units).strip())

            return [c for c in chunks if len(c.strip()) > 5]

        except Exception as e:
            logger.warning(f"Semantic chunking encountered error, using recursive fallback: {e}")
            from utils.text_splitter import RecursiveTextSplitter
            return RecursiveTextSplitter(chunk_size=self.max_chunk_size).split_text(text)
