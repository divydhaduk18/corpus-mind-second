import os
import sys
import unittest
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.database import db
from utils.auth import AuthManager
from utils.context_compressor import ContextualCompressor
from utils.conversation_rag import ConversationRAGManager


class TestPhase5CompressionAndConversation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create test users and sessions
        cls.user1 = db.create_user(f"conv_u1_{int(time.time())}", AuthManager.hash_password("pw123"))
        cls.user2 = db.create_user(f"conv_u2_{int(time.time())}", AuthManager.hash_password("pw123"))

        cls.sess1 = db.create_chat_session(cls.user1, "Session User 1")
        cls.sess2 = db.create_chat_session(cls.user2, "Session User 2")

        # Add messages to sess1
        db.add_message(cls.sess1, "user", "What is the revenue for Acme in Q3 2026?")
        db.add_message(cls.sess1, "assistant", "Acme revenue was $48.5 million.")
        db.add_message(cls.sess1, "user", "How much was gross margin?")
        db.add_message(cls.sess1, "assistant", "Gross margin was 64.2%.")

        # Add messages to sess2
        db.add_message(cls.sess2, "user", "Project Beta acquisition secrets.")

    @classmethod
    def tearDownClass(cls):
        db.delete_user(cls.user1)
        db.delete_user(cls.user2)

    def test_contextual_compression(self):
        """Compression must extract query-relevant sentences while preserving metadata."""
        query = "How much did Acme invest in artificial intelligence research?"
        chunk = {
            "chunk_id": 10,
            "doc_id": 2,
            "file_name": "acme_report.pdf",
            "page_number": 2,
            "section": "R&D",
            "text": (
                "The corporate cafeteria will be undergoing renovations starting next Monday. "
                "Staff should utilize the third floor dining options in the interim. "
                "Acme invested $12.3 million in artificial intelligence research and development. "
                "The annual parking permit renewals must be submitted before Friday."
            )
        }
        compressed = ContextualCompressor.compress_chunk(query, chunk)
        self.assertIn("12.3 million", compressed["text"])
        self.assertIn("artificial intelligence", compressed["text"])
        # Irrelevant cafeteria/parking sentences should be filtered out
        self.assertNotIn("cafeteria", compressed["text"])
        self.assertNotIn("parking", compressed["text"])
        # Metadata must be strictly preserved
        self.assertEqual(compressed["file_name"], "acme_report.pdf")
        self.assertEqual(compressed["page_number"], 2)
        self.assertEqual(compressed["chunk_id"], 10)

    def test_conversation_session_isolation(self):
        """User 2 must be strictly prevented from reading User 1's chat messages."""
        # User 1 accessing own session -> Success
        u1_msgs = ConversationRAGManager.get_isolated_session_messages(self.user1, self.sess1)
        self.assertEqual(len(u1_msgs), 4)

        # User 2 attempting to access User 1's session -> Must return empty list (Rejected)
        u2_illicit = ConversationRAGManager.get_isolated_session_messages(self.user2, self.sess1)
        self.assertEqual(u2_illicit, [])

    def test_token_and_turn_budget_enforcement(self):
        """Conversation history must respect maximum turns and token limits."""
        msgs = ConversationRAGManager.get_isolated_session_messages(
            self.user1,
            self.sess1,
            max_turns=2
        )
        self.assertEqual(len(msgs), 2)
        # Should be the most recent 2 turns
        self.assertEqual(msgs[0]["content"], "How much was gross margin?")
        self.assertEqual(msgs[1]["content"], "Gross margin was 64.2%.")


if __name__ == "__main__":
    unittest.main()
