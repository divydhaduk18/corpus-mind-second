import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.hallucination_detector import HallucinationDetector


class TestPhase7HallucinationDetector(unittest.TestCase):
    def setUp(self):
        self.context = [
            {
                "chunk_id": 1,
                "file_name": "acme_q3_report.pdf",
                "page_number": 1,
                "text": "Acme Corporation Q3 2026 total revenue reached $48.5 million representing a 28% increase year over year. Gross margins improved to 64.2%."
            },
            {
                "chunk_id": 2,
                "file_name": "acme_q3_report.pdf",
                "page_number": 2,
                "text": "Acme invested $12.3 million in artificial intelligence research and deep learning algorithms."
            }
        ]

    def test_verified_answer(self):
        """Answer directly supported by context must receive high faithfulness score and VERIFIED status."""
        answer = "Acme Corporation total revenue reached $48.5 million representing a 28% increase year over year [acme_q3_report.pdf, p. 1]."
        res = HallucinationDetector.verify_answer(
            query="What was Acme's revenue in Q3 2026?",
            answer=answer,
            retrieved_chunks=self.context
        )
        self.assertGreaterEqual(res["faithfulness_score"], 0.80)
        self.assertEqual(res["status"], "VERIFIED")
        self.assertEqual(len(res["unsupported_claims"]), 0)

    def test_refusal_answer_zero_hallucination(self):
        """Honest refusal stating lack of evidence must be marked as REFUSAL_VERIFIED with score 1.0."""
        refusal = "The retrieved documents do not provide sufficient evidence to answer this question."
        res = HallucinationDetector.verify_answer(
            query="What is Acme's stock price target in 2030?",
            answer=refusal,
            retrieved_chunks=self.context
        )
        self.assertEqual(res["faithfulness_score"], 1.0)
        self.assertEqual(res["status"], "REFUSAL_VERIFIED")
        self.assertIsNone(res["qualification_warning"])

    def test_hallucinated_unsupported_answer(self):
        """Fabricated answer with facts not in context must be marked UNSUPPORTED with low score."""
        hallucination = "Acme was founded in 1912 by Johnathan Sterling in Seattle, Washington and produces electric airplanes."
        res = HallucinationDetector.verify_answer(
            query="Who founded Acme?",
            answer=hallucination,
            retrieved_chunks=self.context
        )
        self.assertLess(res["faithfulness_score"], 0.50)
        self.assertEqual(res["status"], "UNSUPPORTED")
        self.assertIsNotNone(res["qualification_warning"])


if __name__ == "__main__":
    unittest.main()
