"""Unit tests for Phase 14: RAGAS Evaluation, Retrieval Benchmarks, and Scalable Indexing."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.retrieval_benchmark import RetrievalBenchmarkEngine
from utils.ragas_evaluator import RAGASEvaluator
from utils.scalable_index import ScalableIndexManager
from utils.database import db


class TestPhase14EvaluationAndBenchmarks(unittest.TestCase):
    """Test suite for evaluation, benchmark metrics, and scalable indexing."""

    def test_dcg_and_ndcg_calculation(self):
        """Verify DCG and nDCG calculations follow correct logarithmic discounting."""
        # Perfect relevance
        perfect_relevance = [2, 2, 1, 0, 0]
        ndcg_perfect = RetrievalBenchmarkEngine.calculate_ndcg(perfect_relevance, k=5)
        self.assertEqual(ndcg_perfect, 1.0)

        # Imperfect relevance
        imperfect_relevance = [0, 1, 2, 0, 0]
        ndcg_imperfect = RetrievalBenchmarkEngine.calculate_ndcg(imperfect_relevance, k=5)
        self.assertGreater(ndcg_imperfect, 0.0)
        self.assertLess(ndcg_imperfect, 1.0)

        # Zero relevance
        zero_relevance = [0, 0, 0, 0]
        ndcg_zero = RetrievalBenchmarkEngine.calculate_ndcg(zero_relevance, k=4)
        self.assertEqual(ndcg_zero, 0.0)

    @patch("utils.rag.RAGManager.retrieve_context")
    def test_retrieval_benchmark_execution(self, mock_retrieve):
        """Verify benchmark pipeline produces all required metrics (Recall, MRR, nDCG, HitRate)."""
        mock_retrieve.return_value = [
            {"file_name": "vendor_contract_2024.pdf", "text": "October 1, 2024 effective date", "score": 0.95},
            {"file_name": "other_doc.pdf", "text": "Some general info", "score": 0.50}
        ]

        custom_dataset = [
            {
                "query_id": "test_q1",
                "type": "exact_keyword",
                "query": "effective date",
                "relevant_doc_names": ["vendor_contract_2024.pdf"],
                "relevant_phrases": ["October 1, 2024"],
                "is_unanswerable": False
            }
        ]

        metrics = RetrievalBenchmarkEngine.evaluate_retrieval_configuration(
            user_id=1,
            search_mode="hybrid",
            enable_rerank=False,
            top_k=2,
            dataset=custom_dataset
        )

        self.assertIn("recall_at_1", metrics)
        self.assertIn("mrr", metrics)
        self.assertIn("ndcg_at_5", metrics)
        self.assertIn("precision_at_k", metrics)
        self.assertIn("hit_rate_at_k", metrics)
        self.assertIn("avg_latency_ms", metrics)
        self.assertEqual(metrics["recall_at_1"], 1.0)
        self.assertEqual(metrics["hit_rate_at_k"], 1.0)

    def test_hardware_monitoring(self):
        """Verify hardware monitoring retrieves CPU and memory metrics."""
        hw = ScalableIndexManager.get_system_metrics()
        self.assertIn("cpu_percent", hw)
        self.assertIn("rss_ram_mb", hw)
        self.assertIn("vms_ram_mb", hw)

    def test_ragas_database_persistence(self):
        """Verify RAGAS evaluation results are persisted and queryable in SQLite."""
        test_dataset_name = "Unit Test Evaluation Run"
        test_summary = {
            "faithfulness": 0.92,
            "answer_relevancy": 0.88,
            "context_precision": 0.85,
            "context_recall": 0.90,
            "answer_correctness": 0.89,
            "context_utilization": 0.95,
            "avg_latency_ms": 120.5
        }

        eval_id = db.save_ragas_evaluation(
            user_id=1,
            dataset_name=test_dataset_name,
            provider="ollama",
            model="llama3.2",
            embedding_model="all-MiniLM-L6-v2",
            metrics=test_summary
        )
        self.assertIsNotNone(eval_id)

        history = db.get_ragas_evaluations_by_user(user_id=1)
        self.assertTrue(any(e["id"] == eval_id for e in history))


if __name__ == "__main__":
    unittest.main()
