"""LLM Evaluation & Benchmarking Engine for Corpus Mind.

Provides side-by-side evaluation across multiple models with identical questions & retrieved context,
measuring TTFT, total latency, TPS, faithfulness, correctness, citations, RAM usage, and errors.
"""

import os
import sys
import time
import json
import psutil
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.llm_provider import LLMProviderManager, LLMResponse
from utils.prompt_manager import PromptManager
from utils.hallucination_detector import HallucinationDetector

logger = logging.getLogger(__name__)


# Default benchmark queries with context & ground-truth facts for fair evaluation
DEFAULT_BENCHMARK_SUITE = [
    {
        "id": "bench_q1",
        "category": "Contract & Dates",
        "question": "What is the effective start date and expiration date of the Master Services Agreement?",
        "ground_truth": "The effective date is October 1, 2024 and expiration date is September 30, 2027.",
        "key_facts": ["October 1, 2024", "September 30, 2027", "Master Services Agreement"],
        "retrieved_context": [
            {
                "file_name": "vendor_contract_2024.pdf",
                "page_number": 1,
                "section": "Article I: Term",
                "text": "This Master Services Agreement ('Agreement') is made effective as of October 1, 2024 ('Effective Date') and shall continue in full force until September 30, 2027 ('Expiration Date'), unless earlier terminated in accordance with Section 8."
            }
        ]
    },
    {
        "id": "bench_q2",
        "category": "Financial Metrics",
        "question": "What were the total Q3 revenues and operating profit margin reported by Acme Corp?",
        "ground_truth": "Acme Corp reported Q3 revenues of $142.5 million and an operating profit margin of 21.4%.",
        "key_facts": ["$142.5 million", "21.4%", "Q3", "operating profit"],
        "retrieved_context": [
            {
                "file_name": "acme_q3_report.pdf",
                "page_number": 4,
                "section": "Financial Highlights",
                "text": "In the third fiscal quarter of 2024, Acme Corp achieved record consolidated net revenues of $142.5 million, representing a 14% year-over-year expansion. GAAP operating margin reached 21.4% compared to 18.2% in the prior year period."
            }
        ]
    },
    {
        "id": "bench_q3",
        "category": "Technical Architecture",
        "question": "What vector dimension does the default all-MiniLM-L6-v2 embedding model produce?",
        "ground_truth": "The all-MiniLM-L6-v2 model produces 384-dimensional dense vector embeddings.",
        "key_facts": ["384", "all-MiniLM-L6-v2", "dimension"],
        "retrieved_context": [
            {
                "file_name": "architecture_guide.md",
                "page_number": 2,
                "section": "Embeddings & Vectors",
                "text": "Corpus Mind utilizes sentence-transformers/all-MiniLM-L6-v2 by default. It maps sentences and document chunks to a 384-dimensional dense vector space, providing an optimal balance between low memory footprint and high retrieval accuracy on CPU architectures."
            }
        ]
    },
    {
        "id": "bench_q4",
        "category": "Negative & Unanswerable",
        "question": "What is the CEO's personal home address and private mobile phone number?",
        "ground_truth": "The retrieved documents do not provide sufficient evidence to answer this question.",
        "key_facts": ["do not provide sufficient evidence", "not available"],
        "retrieved_context": [
            {
                "file_name": "executive_bios.txt",
                "page_number": 1,
                "section": "Leadership Overview",
                "text": "Chief Executive Officer Jane Doe leads global operations from the corporate headquarters in San Francisco, CA. She joined the executive committee in 2021 after serving as Senior VP of Product Development."
            }
        ]
    }
]


class LLMEvaluator:
    """Runs automated multi-model benchmark comparisons measuring performance, quality, and resources."""

    @staticmethod
    def get_system_ram_mb() -> float:
        """Measure current process RSS memory in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0

    @classmethod
    def evaluate_model_on_suite(
        cls,
        provider_name: str,
        model_name: str,
        test_suite: Optional[List[Dict[str, Any]]] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run standard evaluation suite against a single provider and model."""
        suite = test_suite or DEFAULT_BENCHMARK_SUITE
        settings = settings or {"temperature": 0.0, "max_tokens": 350}

        results = []
        total_latency = 0.0
        total_ttft = 0.0
        total_tps = 0.0
        correct_count = 0
        citation_count = 0
        faithfulness_scores = []
        errors_count = 0

        start_ram = cls.get_system_ram_mb()

        for item in suite:
            q_id = item["id"]
            question = item["question"]
            context = item["retrieved_context"]
            ground_truth = item.get("ground_truth", "")
            key_facts = item.get("key_facts", [])

            # Format strictly grounded prompt
            sys_prompt, user_prompt, expected_citations = PromptManager.format_grounded_prompt(
                template_name="question_answering",
                query=question,
                rag_chunks=context
            )

            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ]

            # Generate response via provider manager
            resp: LLMResponse = LLMProviderManager.generate(
                provider_name=provider_name,
                model=model_name,
                messages=messages,
                settings=settings
            )

            if resp.error:
                errors_count += 1
                results.append({
                    "id": q_id,
                    "question": question,
                    "error": resp.error,
                    "status": "ERROR"
                })
                continue

            # 1. Faithfulness Verification via HallucinationDetector
            v_res = HallucinationDetector.verify_answer(
                query=question,
                answer=resp.text,
                retrieved_chunks=context
            )
            f_score = v_res.get("faithfulness_score", 0.0)
            faithfulness_scores.append(f_score)

            # 2. Key Fact & Correctness Match
            matched_facts = 0
            for fact in key_facts:
                if fact.lower() in resp.text.lower():
                    matched_facts += 1
            is_correct = (matched_facts / len(key_facts) >= 0.5) if key_facts else True
            if is_correct:
                correct_count += 1

            # 3. Citation Check
            has_citation = any(cit.lower() in resp.text.lower() for cit in expected_citations) or ("[" in resp.text and "]" in resp.text)
            if has_citation or q_id == "bench_q4":  # Negative question doesn't require citations
                citation_count += 1

            total_latency += resp.latency_ms
            total_ttft += resp.ttft_ms
            total_tps += resp.tokens_per_second

            results.append({
                "id": q_id,
                "category": item.get("category", "General"),
                "question": question,
                "generated_answer": resp.text,
                "latency_ms": resp.latency_ms,
                "ttft_ms": resp.ttft_ms,
                "tokens_per_second": resp.tokens_per_second,
                "faithfulness_score": f_score,
                "status": v_res.get("status", "UNKNOWN"),
                "is_correct": is_correct,
                "has_citation": has_citation
            })

        end_ram = cls.get_system_ram_mb()
        valid_count = max(1, len(results) - errors_count)

        summary = {
            "provider": provider_name,
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "questions_evaluated": len(suite),
            "error_rate": round(errors_count / len(suite), 4),
            "avg_latency_ms": round(total_latency / valid_count, 2),
            "avg_ttft_ms": round(total_ttft / valid_count, 2),
            "avg_tokens_per_second": round(total_tps / valid_count, 2),
            "avg_faithfulness": round(sum(faithfulness_scores) / valid_count, 4) if faithfulness_scores else 0.0,
            "correctness_rate": round(correct_count / len(suite), 4),
            "citation_rate": round(citation_count / len(suite), 4),
            "ram_delta_mb": round(end_ram - start_ram, 2),
            "detailed_results": results
        }

        # Save benchmark run to persistent storage
        cls._save_benchmark_summary(summary)

        return summary

    @classmethod
    def _save_benchmark_summary(cls, summary: Dict[str, Any]):
        try:
            save_dir = config.BENCHMARK_RESULTS_DIR
            save_dir.mkdir(parents=True, exist_ok=True)
            fname = f"benchmark_{summary['provider']}_{summary['model'].replace(':', '_')}_{int(time.time())}.json"
            out_path = save_dir / fname
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save benchmark summary: {e}")

    @classmethod
    def list_saved_benchmarks(cls) -> List[Dict[str, Any]]:
        """List previously saved benchmark runs."""
        save_dir = config.BENCHMARK_RESULTS_DIR
        if not save_dir.exists():
            return []
        benchmarks = []
        for p in sorted(save_dir.glob("benchmark_*.json"), reverse=True):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    benchmarks.append(json.load(f))
            except Exception:
                pass
        return benchmarks
