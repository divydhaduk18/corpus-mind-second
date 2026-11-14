"""Unit & Integration Tests for LLM Evaluation & Benchmarking Engine.

Tests:
- LLMEvaluator running test questions against provider models
- Measurement of TTFT, total latency, TPS, faithfulness, correctness, and citations
- RAM measurement via psutil
- Persistent storage of benchmark summaries
"""

import unittest
from unittest.mock import patch, MagicMock
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.llm_evaluator import LLMEvaluator, DEFAULT_BENCHMARK_SUITE
from utils.llm_provider import LLMResponse


class TestLLMEvaluator(unittest.TestCase):

    def test_default_benchmark_suite_integrity(self):
        """Ensure benchmark suite contains diverse query categories and ground truth."""
        self.assertGreaterEqual(len(DEFAULT_BENCHMARK_SUITE), 4)
        for item in DEFAULT_BENCHMARK_SUITE:
            self.assertIn("id", item)
            self.assertIn("question", item)
            self.assertIn("retrieved_context", item)
            self.assertIn("key_facts", item)

    def test_ram_measurement(self):
        """Verify psutil memory measurement returns a positive MB value."""
        ram_mb = LLMEvaluator.get_system_ram_mb()
        self.assertIsInstance(ram_mb, float)
        self.assertGreater(ram_mb, 0.0)

    @patch("utils.llm_provider.LLMProviderManager.generate")
    @patch("utils.hallucination_detector.HallucinationDetector.verify_answer")
    def test_evaluator_metrics_calculation(self, mock_verify, mock_generate):
        """Verify LLMEvaluator aggregates latency, correctness, and faithfulness accurately."""
        # Mock LLM generation
        mock_generate.return_value = LLMResponse(
            text="The Master Services Agreement effective date is October 1, 2024 and expires on September 30, 2027 [vendor_contract_2024.pdf, p. 1].",
            model="llama3.2",
            provider="ollama",
            completion_tokens=25,
            latency_ms=250.0,
            ttft_ms=65.0,
            tokens_per_second=100.0
        )

        # Mock faithfulness verification
        mock_verify.return_value = {
            "status": "VERIFIED",
            "faithfulness_score": 0.95,
            "claim_verifications": []
        }

        # Run minimal single-question suite
        mini_suite = [DEFAULT_BENCHMARK_SUITE[0]]
        summary = LLMEvaluator.evaluate_model_on_suite(
            provider_name="ollama",
            model_name="llama3.2",
            test_suite=mini_suite
        )

        self.assertEqual(summary["provider"], "ollama")
        self.assertEqual(summary["model"], "llama3.2")
        self.assertEqual(summary["questions_evaluated"], 1)
        self.assertEqual(summary["error_rate"], 0.0)
        self.assertEqual(summary["avg_latency_ms"], 250.0)
        self.assertEqual(summary["avg_ttft_ms"], 65.0)
        self.assertEqual(summary["avg_faithfulness"], 0.95)
        self.assertEqual(summary["correctness_rate"], 1.0)
        self.assertEqual(summary["citation_rate"], 1.0)
        self.assertIn("detailed_results", summary)

    def test_save_and_list_benchmarks(self):
        """Verify benchmark summaries persist to json and can be listed."""
        mock_summary = {
            "provider": "ollama",
            "model": "test_model_save",
            "timestamp": "2026-11-14T12:00:00",
            "questions_evaluated": 4,
            "avg_latency_ms": 150.0,
            "avg_ttft_ms": 40.0,
            "avg_tokens_per_second": 35.0,
            "avg_faithfulness": 0.92,
            "correctness_rate": 1.0,
            "citation_rate": 1.0,
            "error_rate": 0.0,
            "ram_delta_mb": 0.5,
            "detailed_results": []
        }

        LLMEvaluator._save_benchmark_summary(mock_summary)
        saved = LLMEvaluator.list_saved_benchmarks()
        self.assertTrue(any(s.get("model") == "test_model_save" for s in saved))


if __name__ == "__main__":
    unittest.main()
