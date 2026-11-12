import os
import sys
import logging
from typing import List, Dict, Any, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.database import db

logger = logging.getLogger(__name__)


class ConversationRAGManager:
    """Manages multi-tenant conversation-aware RAG context, token budgeting, and session isolation."""

    @classmethod
    def get_isolated_session_messages(
        cls,
        user_id: int,
        session_id: int,
        max_turns: Optional[int] = None,
        max_tokens: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """Retrieve recent messages strictly isolated by user_id and session_id with token budgeting.
        
        Enforces database-level isolation ensuring cross-user session tampering is rejected.
        """
        if user_id is None or session_id is None:
            return []

        resolved_turns = max_turns or getattr(config, "MAX_CONVERSATION_HISTORY_TURNS", 6)
        resolved_tokens = max_tokens or getattr(config, "MAX_CONVERSATION_HISTORY_TOKENS", 1000)

        # Verify session belongs strictly to authenticated user
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM chat_sessions WHERE id = ? AND user_id = ?",
                (session_id, user_id)
            )
            session_row = cursor.fetchone()
            if not session_row:
                logger.warning(f"Security: User {user_id} attempted access to invalid/unauthorized session {session_id}.")
                return []

            cursor.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id DESC",
                (session_id,)
            )
            raw_rows = cursor.fetchall()

        # Build chronological sliding window within token and turn budgets
        selected_messages = []
        accumulated_chars = 0
        max_chars = resolved_tokens * 4

        for row in raw_rows:
            content = row["content"] or ""
            role = row["role"] or "user"
            
            # Don't exceed turn or character budget
            if len(selected_messages) >= resolved_turns:
                break
            if accumulated_chars + len(content) > max_chars and len(selected_messages) >= 2:
                break

            selected_messages.append({"role": role, "content": content})
            accumulated_chars += len(content)

        # Reverse to chronological order (oldest to newest)
        selected_messages.reverse()
        return selected_messages

    @classmethod
    def contextualize_user_query(
        cls,
        current_query: str,
        user_id: int,
        session_id: int,
        model: Optional[str] = None
    ) -> str:
        """Analyze conversational history to contextualize and resolve follow-up query references."""
        from utils.query_processor import QueryProcessor

        history = cls.get_isolated_session_messages(
            user_id=user_id,
            session_id=session_id,
            max_turns=4
        )

        if not history:
            return current_query.strip()

        # Apply LLM query rewriting if ambiguous pronouns or follow-up phrasing detected
        if QueryProcessor.needs_rewriting(current_query, history):
            return QueryProcessor.rewrite_query(current_query, history, model=model)

        return current_query.strip()
