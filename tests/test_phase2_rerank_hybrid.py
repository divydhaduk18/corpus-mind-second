import os
import sys
import unittest
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.reranker import CrossEncoderReranker
from utils.rag import BM25Retriever, NumpyVectorStore


class TestPhase2RerankerAndHybrid(unittest.TestCase):
    def test_cross_encoder_reranking_basic(self):
        """Verify CrossEncoder accurately reranks candidate passages based on query relevance."""
        query = "What is the capital of France?"
        candidates = [
            {"chunk_id": 1, "text": "The Eiffel tower is located in Paris, which is the capital of France.", "score": 0.5},
            {"chunk_id": 2, "text": "London is the capital of England and features the Big Ben.", "score": 0.8},
            {"chunk_id": 3, "text": "Apples and oranges are nutritious fruits rich in vitamins.", "score": 0.2},
            {"chunk_id": 4, "text": "France has a rich history and its capital city is Paris.", "score": 0.4}
        ]

        reranked = CrossEncoderReranker.rerank(query, candidates, top_k=2)
        self.assertEqual(len(reranked), 2)
        # Both top results should identify Paris as the capital
        self.assertIn("Paris", reranked[0]["text"])
        self.assertIn("capital", reranked[0]["text"].lower())
        self.assertIn("Paris", reranked[1]["text"])
        self.assertIn("rerank_score", reranked[0])

    def test_cross_encoder_empty_and_graceful_fallback(self):
        """CrossEncoder must handle empty queries, empty candidates, and invalid input without crashing."""
        self.assertEqual(CrossEncoderReranker.rerank("", [{"text": "Sample"}]), [{"text": "Sample"}])
        self.assertEqual(CrossEncoderReranker.rerank("Query", []), [])
        self.assertEqual(CrossEncoderReranker.rerank("Query", None), [])

    def test_cross_encoder_caching(self):
        """CrossEncoder model must be cached as a singleton in memory."""
        m1 = CrossEncoderReranker.load_model()
        m2 = CrossEncoderReranker.load_model()
        self.assertIs(m1, m2)

    def test_bm25_retriever(self):
        """Verify inverted index Okapi BM25 retriever functionality."""
        corpus = [
            "Quick brown fox jumps over the lazy dog",
            "Artificial intelligence and retrieval augmented generation systems",
            "SQLite database write ahead logging for high concurrency"
        ]
        bm25 = BM25Retriever(corpus)
        results = bm25.search("retrieval generation", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][1], 1)  # Corpus index 1


if __name__ == "__main__":
    unittest.main()
