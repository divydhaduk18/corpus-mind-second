import os
import sys
import re
import logging
from typing import List, Dict, Any, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.chat import OllamaChatManager

logger = logging.getLogger(__name__)


class QueryProcessor:
    """Handles LLM-based query rewriting and multi-query expansion for advanced RAG retrieval."""

    PRONOUN_PATTERN = re.compile(
        r'\b(it|its|they|them|their|this|that|these|those|he|him|his|she|her|the same|above|previous)\b',
        re.IGNORECASE
    )

    @classmethod
    def needs_rewriting(cls, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> bool:
        """Heuristic check to determine if a query contains coreferences or follow-up ambiguities."""
        if not conversation_history:
            return False
        clean_q = query.strip()
        if len(clean_q.split()) <= 4:
            return True
        if cls.PRONOUN_PATTERN.search(clean_q):
            return True
        if clean_q.lower().startswith(("and ", "what about", "how about", "why", "where is it", "who did")):
            return True
        return False

    @classmethod
    def clean_llm_query_output(cls, raw_output: str, original_query: str) -> str:
        """Sanitize LLM output by removing conversational filler and quotes."""
        if not raw_output or not raw_output.strip():
            return original_query.strip()

        cleaned = raw_output.strip()
        # Remove common markdown bolding or backticks
        cleaned = re.sub(r'[*`_]', '', cleaned)
        # Remove conversational prefixes
        prefixes = [
            r'^(?:rewritten|standalone|search|rephrased)?\s*query:\s*',
            r'^(?:here is|here\'s)\s*(?:the)?\s*(?:rewritten|rephrased)?\s*query:?\s*',
            r'^"|"$',
            r"^'|'$"
        ]
        for p in prefixes:
            cleaned = re.sub(p, '', cleaned, flags=re.IGNORECASE).strip()

        # Take first non-empty line
        lines = [l.strip() for l in cleaned.split("\n") if l.strip()]
        if lines:
            first_line = lines[0]
            # Strip list numbers like "1. "
            first_line = re.sub(r'^\d+[\.\)]\s*', '', first_line).strip()
            if len(first_line) > 3:
                return first_line

        return original_query.strip()

    @classmethod
    def rewrite_query(
        cls,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None
    ) -> str:
        """Rewrite conversational/follow-up queries into standalone self-contained search queries.
        
        Uses recent conversation context to resolve coreferences like 'it', 'they', 'that'.
        Falls back safely to original query if LLM is unreachable or disabled.
        """
        if not query or not query.strip():
            return ""

        if not getattr(config, "QUERY_REWRITING_ENABLED", True):
            return query.strip()

        if not conversation_history:
            return query.strip()

        # Format concise history representation
        recent_history = conversation_history[-4:]
        history_lines = []
        for msg in recent_history:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "").strip()
            # Truncate lengthy assistant messages
            if len(content) > 150:
                content = content[:150] + "..."
            history_lines.append(f"{role}: {content}")

        history_str = "\n".join(history_lines)
        system_instruction = (
            "You are an expert search query formulation assistant. "
            "Given the chat history and the user's latest follow-up question, "
            "rephrase the follow-up question into a complete, self-contained search query. "
            "Resolve any pronouns (it, they, that, this) using the history. "
            "Preserve all specific names, entities, and technical keywords. "
            "Output ONLY the rephrased query string without explanations, quotes, or preambles."
        )

        user_prompt = f"Chat History:\n{history_str}\n\nFollow-up Question: {query}\n\nStandalone Search Query:"

        try:
            target_model = model or getattr(config, "DEFAULT_MODEL", "llama3.2")
            response_gen = OllamaChatManager.generate_chat_response(
                model=target_model,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt}
                ],
                settings={"temperature": 0.1, "max_tokens": 64}
            )
            raw_response = "".join(list(response_gen)).strip()

            if "Connection error:" in raw_response or "Error:" in raw_response:
                return query.strip()

            rewritten = cls.clean_llm_query_output(raw_response, query)
            logger.info(f"Query rewritten: '{query}' -> '{rewritten}'")
            return rewritten

        except Exception as e:
            logger.warning(f"Query rewriting failed with error: {e}. Falling back to original query.")
            return query.strip()

    @classmethod
    def generate_multi_queries(
        cls,
        query: str,
        num_queries: int = 3,
        model: Optional[str] = None
    ) -> List[str]:
        """Generate multiple alternative search queries from different angles to improve retrieval recall.
        
        Always includes the original query as the primary query.
        Falls back to rule-based variations if LLM is unavailable.
        """
        clean_q = query.strip()
        if not clean_q:
            return []

        if not getattr(config, "MULTI_QUERY_ENABLED", True) or num_queries <= 1:
            return [clean_q]

        system_instruction = (
            f"You are an AI language model assistant. Generate {num_queries} distinct, alternative search queries "
            "to retrieve relevant documents for the user's question. Capture different perspectives, synonyms, "
            "and keyword phrasings. Provide each query on a new line prefixed with a number (1., 2., 3.). "
            "Output ONLY the numbered queries, with no introduction or conclusion."
        )

        user_prompt = f"Original Question: {clean_q}\n\nAlternative Search Queries:"

        try:
            target_model = model or getattr(config, "DEFAULT_MODEL", "llama3.2")
            response_gen = OllamaChatManager.generate_chat_response(
                model=target_model,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt}
                ],
                settings={"temperature": 0.3, "max_tokens": 128}
            )
            raw_response = "".join(list(response_gen)).strip()

            queries = [clean_q]
            if "Connection error:" not in raw_response and "Error:" not in raw_response:
                lines = raw_response.split("\n")
                for line in lines:
                    cleaned_line = re.sub(r'^\d+[\.\)]\s*', '', line.strip()).strip()
                    cleaned_line = re.sub(r'^["\']|["\']$', '', cleaned_line).strip()
                    if cleaned_line and cleaned_line.lower() != clean_q.lower() and len(cleaned_line) > 5:
                        if cleaned_line not in queries:
                            queries.append(cleaned_line)
                    if len(queries) >= num_queries:
                        break

            # If LLM didn't return enough or was offline, generate heuristic keyword perspectives
            if len(queries) < num_queries:
                words = clean_q.split()
                if len(words) > 3:
                    # Noun/Keyword core perspective
                    key_terms = [w for w in words if len(w) > 3 and w.lower() not in {"what", "which", "where", "when", "does", "have", "with", "from"}]
                    if key_terms:
                        heuristic_query = " ".join(key_terms)
                        if heuristic_query not in queries:
                            queries.append(heuristic_query)

            return queries[:num_queries]

        except Exception as e:
            logger.warning(f"Multi-query generation failed: {e}. Returning original query.")
            return [clean_q]
