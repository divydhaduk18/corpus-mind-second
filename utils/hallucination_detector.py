import os
import sys
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.embeddings import EmbeddingManager
from utils.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


class HallucinationDetector:
    """Rigorous answer verification and hallucination detection engine.
    
    Validates generated model responses against retrieved source document passages.
    Extracts individual factual claims and verifies semantic entailment and key entity
    presence to calculate an objective faithfulness score.
    """

    REFUSAL_PATTERNS = [
        r"does not provide sufficient evidence",
        r"do not provide sufficient evidence",
        r"not mentioned in the (?:retrieved|provided) documents",
        r"cannot be answered from the (?:retrieved|provided)",
        r"no information (?:available|provided|found)",
        r"not stated in the (?:retrieved|provided) context"
    ]

    @classmethod
    def is_refusal_answer(cls, answer: str) -> bool:
        """Check if the assistant explicitly and correctly stated lack of document evidence."""
        answer_lower = answer.lower()
        return any(re.search(pat, answer_lower) for pat in cls.REFUSAL_PATTERNS)

    @classmethod
    def extract_atomic_claims(cls, answer: str) -> List[str]:
        """Extract individual factual claims from the assistant's answer."""
        if not answer or not answer.strip():
            return []

        # Remove source citation tags like [file.pdf, p. 2]
        clean_ans = re.sub(r'\[[^\]]+,\s*p\.\s*\d+\]', '', answer)
        # Remove bold/italic markdown
        clean_ans = re.sub(r'[*`_#]', '', clean_ans)
        
        # Split into sentences
        raw_sentences = re.split(r'(?<=[.?!])\s+', clean_ans.strip())
        claims = []

        for s in raw_sentences:
            s_clean = s.strip()
            # Filter out micro-phrases, conversational greetings, and source headers
            if len(s_clean) < 15:
                continue
            if s_clean.lower().startswith(("here is", "sure,", "certainly", "based on", "sources & citations", "sources:")):
                continue
            claims.append(s_clean)

        return claims

    @classmethod
    def verify_answer(
        cls,
        query: str,
        answer: str,
        retrieved_chunks: List[Dict[str, Any]],
        faithfulness_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Verify generated answer against retrieved context.
        
        Returns:
            Dict containing:
                - faithfulness_score (float 0.0 to 1.0)
                - status: 'VERIFIED', 'PARTIALLY_GROUNDED', 'UNSUPPORTED', or 'REFUSAL_VERIFIED'
                - verified_claims: List[Dict]
                - unsupported_claims: List[str]
                - qualification_warning: Optional[str]
        """
        threshold = (
            faithfulness_threshold if faithfulness_threshold is not None
            else getattr(config, "FAITHFULNESS_THRESHOLD", 0.70)
        )

        # 1. Grounded refusal handling (honest admission of insufficient evidence is 100% faithful)
        if cls.is_refusal_answer(answer):
            return {
                "faithfulness_score": 1.0,
                "status": "REFUSAL_VERIFIED",
                "verified_claims": [{"claim": answer, "grounded": True, "support_score": 1.0}],
                "unsupported_claims": [],
                "qualification_warning": None,
                "summary": "Verified: Assistant correctly adhered to grounding rules by stating lack of evidence."
            }

        if not retrieved_chunks:
            return {
                "faithfulness_score": 0.0,
                "status": "UNSUPPORTED",
                "verified_claims": [],
                "unsupported_claims": [answer],
                "qualification_warning": "Warning: Answer was generated without any retrieved document context.",
                "summary": "Unsupported: Zero document context available to verify answer."
            }

        claims = cls.extract_atomic_claims(answer)
        if not claims:
            return {
                "faithfulness_score": 1.0,
                "status": "VERIFIED",
                "verified_claims": [],
                "unsupported_claims": [],
                "qualification_warning": None,
                "summary": "Verified: No unsupported claims detected."
            }

        # Combine retrieved passages for entailment comparison
        context_texts = [c.get("text", "") for c in retrieved_chunks if c.get("text")]
        combined_context = " ".join(context_texts)

        # Check each claim against the context
        verified_claims = []
        unsupported_claims = []

        # Use CrossEncoder for deep semantic cross-attention verification if available
        ce_model = CrossEncoderReranker.load_model()

        for claim in claims:
            is_supported = False
            max_support_score = 0.0

            # 1. Quick exact keyword/token subset check
            claim_words = [w.lower() for w in re.findall(r'\b\w+\b', claim) if len(w) > 3]
            if claim_words:
                found_words = sum(1 for w in claim_words if w in combined_context.lower())
                keyword_ratio = found_words / len(claim_words)
            else:
                keyword_ratio = 1.0

            # 2. Deep cross-attention score if CrossEncoder loaded
            if ce_model is not None and context_texts:
                pairs = [(claim, c_text) for c_text in context_texts]
                try:
                    scores = ce_model.predict(pairs)
                    max_ce_score = float(np.max(scores))
                    # ms-marco-MiniLM scores: positive values indicate strong relevance/support
                    max_support_score = max_ce_score
                    if max_ce_score >= -1.0 and keyword_ratio >= 0.50:
                        is_supported = True
                    elif keyword_ratio >= 0.85:
                        is_supported = True
                except Exception as ce_err:
                    logger.warning(f"CrossEncoder verification error: {ce_err}")
                    is_supported = (keyword_ratio >= 0.65)
            else:
                is_supported = (keyword_ratio >= 0.65)
                max_support_score = keyword_ratio

            claim_record = {
                "claim": claim,
                "grounded": is_supported,
                "support_score": round(max_support_score, 3),
                "keyword_overlap": round(keyword_ratio, 3)
            }

            if is_supported:
                verified_claims.append(claim_record)
            else:
                unsupported_claims.append(claim)

        # Compute aggregate faithfulness score
        total_claims = len(claims)
        supported_count = len(verified_claims)
        score = supported_count / total_claims if total_claims > 0 else 1.0

        if score >= 0.80:
            status = "VERIFIED"
            warning = None
        elif score >= 0.50:
            status = "PARTIALLY_GROUNDED"
            warning = f"Notice: Answer is partially grounded ({int(score * 100)}% faithful). Some claims could not be verified in the documents."
        else:
            status = "UNSUPPORTED"
            warning = f"Caution: Low grounding score ({int(score * 100)}% faithful). The answer may contain unsupported statements or hallucinations."

        return {
            "faithfulness_score": round(score, 3),
            "status": status,
            "verified_claims": verified_claims,
            "unsupported_claims": unsupported_claims,
            "qualification_warning": warning,
            "summary": f"{supported_count}/{total_claims} claims verified against source context."
        }
