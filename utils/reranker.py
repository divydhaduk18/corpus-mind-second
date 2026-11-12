import os
import sys
import logging
from typing import List, Dict, Any, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """High-performance Cross-Encoder reranker using Sentence Transformers.
    
    Scores (query, passage) pairs jointly with deep cross-attention to achieve
    state-of-the-art document re-ranking over candidate pools retrieved from
    BM25 and FAISS dense vector search.
    """

    _model_cache: Dict[str, Any] = {}

    @classmethod
    def get_model_name(cls, model_name: Optional[str] = None) -> str:
        """Resolve reranking model name from parameter, config, or default."""
        return model_name or getattr(
            config, 
            "DEFAULT_RERANKER_MODEL", 
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    @classmethod
    def load_model(cls, model_name: Optional[str] = None):
        """Load and cache CrossEncoder singleton model instance."""
        resolved = cls.get_model_name(model_name)
        if resolved in cls._model_cache:
            return cls._model_cache[resolved]

        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading Cross-Encoder reranking model: {resolved}")
            # Ensure offline mode if environment flag is set
            model = CrossEncoder(resolved)
            cls._model_cache[resolved] = model
            return model
        except Exception as e:
            logger.error(f"Failed to load Cross-Encoder model '{resolved}': {e}", exc_info=True)
            return None

    @classmethod
    def rerank(
        cls,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 4,
        model_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Rerank a list of candidate chunk dictionaries for a given query.
        
        Args:
            query: The user search query string.
            candidates: List of candidate chunk dictionaries containing 'text'.
            top_k: Number of highest-ranked chunks to return.
            model_name: Optional model identifier.
            
        Returns:
            List of reranked candidate dictionaries with 'rerank_score' attached, sorted descending.
        """
        if not candidates or not query or not query.strip():
            return candidates[:top_k] if candidates else []

        model = cls.load_model(model_name)
        if model is None:
            logger.warning("Cross-Encoder unavailable; falling back to candidate initial ranking order.")
            return candidates[:top_k]

        try:
            # Prepare (query, text) pairs for cross-attention scoring
            pairs: List[Tuple[str, str]] = []
            for c in candidates:
                text = c.get("text", "")
                pairs.append((query.strip(), text.strip()))

            scores = model.predict(pairs)

            # Attach scores and sort candidates
            reranked = []
            for idx, candidate in enumerate(candidates):
                item = candidate.copy()
                raw_score = float(scores[idx]) if hasattr(scores, "__getitem__") else float(scores)
                item["rerank_score"] = raw_score
                item["pre_rerank_score"] = candidate.get("score", 0.0)
                item["score"] = raw_score
                item["search_mode"] = f"{candidate.get('search_mode', 'Hybrid')} + CrossEncoder"
                reranked.append(item)

            reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
            return reranked[:top_k]

        except Exception as e:
            logger.error(f"Error during Cross-Encoder reranking inference: {e}", exc_info=True)
            return candidates[:top_k]
