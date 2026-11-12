import os
import sys
import unittest
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.query_processor import QueryProcessor


class TestPhase4Queries(unittest.TestCase):
    def test_clean_llm_query_output(self):
        """Verify cleaning filters out conversational noise, quotes, and preambles."""
        raw1 = 'Here is the rewritten query: "Acme Q3 2026 consolidated revenue and gross margin"'
        cleaned1 = QueryProcessor.clean_llm_query_output(raw1, "what about revenue")
        self.assertEqual(cleaned1, "Acme Q3 2026 consolidated revenue and gross margin")

        raw2 = "1. FAISS IndexFlatIP cosine similarity configuration"
        cleaned2 = QueryProcessor.clean_llm_query_output(raw2, "how does index work")
        self.assertEqual(cleaned2, "FAISS IndexFlatIP cosine similarity configuration")

        raw3 = ""
        cleaned3 = QueryProcessor.clean_llm_query_output(raw3, "fallback query")
        self.assertEqual(cleaned3, "fallback query")

    def test_rewrite_query_with_mocked_llm(self):
        """Query rewriting must resolve pronouns based on recent chat history."""
        history = [
            {"role": "user", "content": "Tell me about Acme Corporation's financial results in Q3 2026."},
            {"role": "assistant", "content": "Acme reported $48.5 million in revenue with 64.2% gross margin."}
        ]
        follow_up = "How much did it invest in AI research?"

        def mock_llm_response(*args, **kwargs):
            yield "How much did Acme Corporation invest in AI research in Q3 2026?"

        with patch("utils.chat.OllamaChatManager.generate_chat_response", side_effect=mock_llm_response):
            rewritten = QueryProcessor.rewrite_query(follow_up, conversation_history=history)
            self.assertIn("Acme Corporation", rewritten)
            self.assertIn("AI research", rewritten)

    def test_rewrite_query_fallback_on_offline(self):
        """When LLM is unreachable, must gracefully return original query."""
        def mock_offline(*args, **kwargs):
            yield "Connection error: Could not reach Ollama at http://localhost:11434."

        with patch("utils.chat.OllamaChatManager.generate_chat_response", side_effect=mock_offline):
            res = QueryProcessor.rewrite_query("what about revenue?", conversation_history=[])
            self.assertEqual(res, "what about revenue?")

    def test_multi_query_generation_with_mocked_llm(self):
        """Multi-query generation must produce multiple diverse queries and include original."""
        original_query = "What database does Corpus Mind use?"

        def mock_multi_queries(*args, **kwargs):
            yield "1. Corpus Mind database architecture and storage\n2. SQLite configuration and write-ahead logging in Corpus Mind\n3. What vector database is used in Corpus Mind"

        with patch("utils.chat.OllamaChatManager.generate_chat_response", side_effect=mock_multi_queries):
            queries = QueryProcessor.generate_multi_queries(original_query, num_queries=3)
            self.assertEqual(len(queries), 3)
            self.assertEqual(queries[0], original_query)
            self.assertIn("Corpus Mind database architecture", queries[1])


if __name__ == "__main__":
    unittest.main()
