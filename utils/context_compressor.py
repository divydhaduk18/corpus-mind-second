import os
import sys
import re
import logging
from typing import List, Dict, Any, Optional
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.embeddings import EmbeddingManager

logger = logging.getLogger(__name__)


class ContextualCompressor:
    """Contextual compression engine.
    
    Extracts strictly the query-relevant sentences and passages from retrieved chunks,
    filtering out extraneous boilerplate while rigorously preserving document citations,
    section contexts, page numbers, and factual accuracy.
    """

    @classmethod
    def compress_chunk(
        cls,
        query: str,
        chunk: Dict[str, Any],
        relevance_threshold: Optional[float] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compress a single chunk by extracting sentences semantically relevant to query."""
        if not getattr(config, "CONTEXTUAL_COMPRESSION_ENABLED", True):
            return chunk

        text = chunk.get("text", "")
        if not text or len(text.strip()) < 100:
            return chunk

        threshold = (
            relevance_threshold if relevance_threshold is not None 
            else getattr(config, "COMPRESSION_SIMILARITY_THRESHOLD", 0.35)
        )
        resolved_model = model_name or getattr(config, "DEFAULT_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

        # Split into individual sentences
        raw_sentences = re.split(r'(?<=[.?!])\s+', text.strip())
        sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 10]

        if len(sentences) <= 2:
            return chunk

        try:
            # Generate query embedding
            query_emb = EmbeddingManager.get_embedding(query, model_name=resolved_model)
            q_norm = np.linalg.norm(query_emb)
            if q_norm == 0:
                return chunk
            q_unit = query_emb / q_norm

            # Generate sentence embeddings
            sent_embs = EmbeddingManager.get_embeddings(sentences, model_name=resolved_model)
            norms = np.linalg.norm(sent_embs, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            sent_units = sent_embs / norms

            # Cosine similarities
            similarities = np.dot(sent_units, q_unit)

            # Select sentences exceeding threshold or top 2 highest if all below
            selected_indices = [i for i, score in enumerate(similarities) if score >= threshold]

            if not selected_indices:
                # Keep top 2 highest scoring sentences as conservative extraction
                top_indices = np.argsort(similarities)[-2:]
                selected_indices = sorted(top_indices.tolist())

            # Always maintain natural reading order of sentences
            selected_indices.sort()
            extracted_sentences = [sentences[i] for i in selected_indices]
            compressed_text = " ".join(extracted_sentences).strip()

            if len(compressed_text) >= 30 and len(compressed_text) < len(text) * 0.95:
                compressed_chunk = chunk.copy()
                compressed_chunk["original_text"] = text
                compressed_chunk["text"] = compressed_text
                compressed_chunk["is_compressed"] = True
                compressed_chunk["compression_ratio"] = float(len(compressed_text) / len(text))
                return compressed_chunk

            return chunk

        except Exception as e:
            logger.warning(f"Contextual compression failed for chunk {chunk.get('chunk_id')}: {e}. Retaining original.")
            return chunk

    @classmethod
    def compress_chunks(
        cls,
        query: str,
        chunks: List[Dict[str, Any]],
        relevance_threshold: Optional[float] = None,
        model_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Compress a list of retrieved chunks before passing to LLM context assembly."""
        if not chunks or not query or not query.strip():
            return chunks

        return [
            cls.compress_chunk(
                query=query,
                chunk=c,
                relevance_threshold=relevance_threshold,
                model_name=model_name
            )
            for c in chunks
        ]
