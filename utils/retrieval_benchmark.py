"""Comprehensive Retrieval Benchmarking Engine for Corpus Mind.

Evaluates retrieval quality across algorithms and configurations:
- Recall@1, Recall@3, Recall@5, Recall@10
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain: nDCG@5, nDCG@10
- Precision@K & Hit Rate@K
- Average Latency (ms) and Throughput (queries/second)
- Supports comparative benchmarking across BM25, FAISS, Hybrid (RRF/Weighted), and Cross-Encoder Reranking
"""

import os
import sys
import time
import json
import math
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.rag import RAGManager

logger = logging.getLogger(__name__)

# Standardized Ground-Truth Evaluation Dataset
# Covers exact keyword, semantic, paraphrased, multi-chunk, and negative unanswerable queries
RETRIEVAL_EVALUATION_DATASET = [
    {
        "query_id": "ret_01",
        "type": "exact_keyword",
        "query": "Master Services Agreement effective date and expiration date",
        "relevant_doc_names": ["vendor_contract_2024.pdf"],
        "relevant_phrases": ["October 1, 2024", "September 30, 2027", "Article I: Term"],
        "is_unanswerable": False
    },
    {
        "query_id": "ret_02",
        "type": "semantic",
        "query": "financial health and operational profitability in the third quarter",
        "relevant_doc_names": ["acme_q3_report.pdf"],
        "relevant_phrases": ["142.5 million", "21.4%", "GAAP operating margin"],
        "is_unanswerable": False
    },
    {
        "query_id": "ret_03",
        "type": "paraphrased",
        "query": "how many dimensions does the sentence embedding space use",
        "relevant_doc_names": ["architecture_guide.md"],
        "relevant_phrases": ["384-dimensional dense vector space", "all-MiniLM-L6-v2"],
        "is_unanswerable": False
    },
    {
        "query_id": "ret_04",
        "type": "multi_chunk",
        "query": "company vacation policy, paid time off, and bereavement leave guidelines",
        "relevant_doc_names": ["employee_handbook.txt"],
        "relevant_phrases": ["vacation policy", "paid time off", "bereavement leave"],
        "is_unanswerable": False
    },
    {
        "query_id": "ret_05",
        "type": "negative_unanswerable",
        "query": "chief executive officer personal mobile phone and home street address",
        "relevant_doc_names": [],
        "relevant_phrases": [],
        "is_unanswerable": True
    }
]


class RetrievalBenchmarkEngine:
    """Calculates standardized retrieval metrics and runs comparative evaluations."""

    @staticmethod
    def calculate_dcg(relevance_scores: List[int], k: int) -> float:
        """Calculate Discounted Cumulative Gain at K: DCG@K = sum((2^rel - 1) / log2(rank + 1))."""
        dcg = 0.0
        for i, rel in enumerate(relevance_scores[:k], start=1):
            if rel > 0:
                dcg += (math.pow(2, rel) - 1.0) / math.log2(i + 1)
        return dcg

    @classmethod
    def calculate_ndcg(cls, relevance_scores: List[int], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain at K (nDCG@K)."""
        actual_dcg = cls.calculate_dcg(relevance_scores, k)
        ideal_relevance = sorted(relevance_scores, reverse=True)
        ideal_dcg = cls.calculate_dcg(ideal_relevance, k)
        return round(actual_dcg / ideal_dcg, 4) if ideal_dcg > 0 else 0.0

    @classmethod
    def evaluate_retrieval_configuration(
        cls,
        user_id: int,
        search_mode: str = "hybrid",
        enable_rerank: bool = True,
        top_k: int = 5,
        dataset: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Run benchmark evaluation for a specific retrieval configuration and calculate IR metrics."""
        eval_suite = dataset or RETRIEVAL_EVALUATION_DATASET

        recalls_at_1 = []
        recalls_at_3 = []
        recalls_at_5 = []
        recalls_at_10 = []
        mrr_scores = []
        ndcg_at_5_scores = []
        ndcg_at_10_scores = []
        precision_at_k = []
        hit_rates = []
        latencies_ms = []
        query_details = []

        # Temporarily configure reranking setting
        old_rerank_setting = getattr(config, "RERANKING_ENABLED", True)
        config.RERANKING_ENABLED = enable_rerank

        try:
            for item in eval_suite:
                q_id = item["query_id"]
                query = item["query"]
                target_docs = [d.lower() for d in item.get("relevant_doc_names", [])]
                target_phrases = [p.lower() for p in item.get("relevant_phrases", [])]
                is_negative = item.get("is_unanswerable", False)

                # Execute retrieval with timed latency measurement
                start_t = time.perf_counter()
                results = RAGManager.retrieve_context(
                    query=query,
                    user_id=user_id,
                    top_k=max(10, top_k),
                    search_mode=search_mode
                )
                duration_ms = (time.perf_counter() - start_t) * 1000.0
                latencies_ms.append(duration_ms)

                if is_negative:
                    # For negative queries, success means zero confidence false matches
                    continue

                # Judge relevance of each retrieved chunk (1 if relevant, 0 if not)
                binary_relevance = []
                first_relevant_rank = None

                for rank_idx, chunk in enumerate(results, start=1):
                    c_file = chunk.get("file_name", "").lower()
                    c_text = chunk.get("text", "").lower()

                    matches_doc = any(td in c_file for td in target_docs) if target_docs else True
                    matches_phrase = any(tp in c_text for tp in target_phrases) if target_phrases else False
                    is_rel = 1 if (matches_doc and matches_phrase) else 0
                    binary_relevance.append(is_rel)

                    if is_rel == 1 and first_relevant_rank is None:
                        first_relevant_rank = rank_idx

                # Compute Metrics for this query
                r_at_1 = 1.0 if (len(binary_relevance) >= 1 and binary_relevance[0] == 1) else 0.0
                r_at_3 = 1.0 if any(binary_relevance[:3]) else 0.0
                r_at_5 = 1.0 if any(binary_relevance[:5]) else 0.0
                r_at_10 = 1.0 if any(binary_relevance[:10]) else 0.0

                mrr = (1.0 / first_relevant_rank) if first_relevant_rank else 0.0
                ndcg_5 = cls.calculate_ndcg(binary_relevance, k=5)
                ndcg_10 = cls.calculate_ndcg(binary_relevance, k=10)
                prec_k = (sum(binary_relevance[:top_k]) / top_k) if top_k > 0 else 0.0
                hit = 1.0 if any(binary_relevance[:top_k]) else 0.0

                recalls_at_1.append(r_at_1)
                recalls_at_3.append(r_at_3)
                recalls_at_5.append(r_at_5)
                recalls_at_10.append(r_at_10)
                mrr_scores.append(mrr)
                ndcg_at_5_scores.append(ndcg_5)
                ndcg_at_10_scores.append(ndcg_10)
                precision_at_k.append(prec_k)
                hit_rates.append(hit)

                query_details.append({
                    "query_id": q_id,
                    "query": query,
                    "first_relevant_rank": first_relevant_rank or "Not Found",
                    "latency_ms": round(duration_ms, 2),
                    "mrr": round(mrr, 4),
                    "ndcg_5": ndcg_5,
                    "results_count": len(results)
                })

        finally:
            config.RERANKING_ENABLED = old_rerank_setting

        valid_count = max(1, len(recalls_at_5))
        avg_latency = round(sum(latencies_ms) / max(1, len(latencies_ms)), 2)
        throughput = round(1000.0 / avg_latency, 2) if avg_latency > 0 else 0.0

        summary = {
            "timestamp": datetime.now().isoformat(),
            "search_mode": search_mode,
            "cross_encoder_rerank": enable_rerank,
            "top_k": top_k,
            "queries_evaluated": len(eval_suite),
            "recall_at_1": round(sum(recalls_at_1) / valid_count, 4),
            "recall_at_3": round(sum(recalls_at_3) / valid_count, 4),
            "recall_at_5": round(sum(recalls_at_5) / valid_count, 4),
            "recall_at_10": round(sum(recalls_at_10) / valid_count, 4),
            "mrr": round(sum(mrr_scores) / valid_count, 4),
            "ndcg_at_5": round(sum(ndcg_at_5_scores) / valid_count, 4),
            "ndcg_at_10": round(sum(ndcg_at_10_scores) / valid_count, 4),
            "precision_at_k": round(sum(precision_at_k) / valid_count, 4),
            "hit_rate_at_k": round(sum(hit_rates) / valid_count, 4),
            "avg_latency_ms": avg_latency,
            "throughput_qps": throughput,
            "query_details": query_details
        }

        # Save benchmark summary
        cls._save_benchmark_summary(summary)
        return summary

    @classmethod
    def _save_benchmark_summary(cls, summary: Dict[str, Any]):
        try:
            save_dir = config.RETRIEVAL_BENCHMARK_DIR
            save_dir.mkdir(parents=True, exist_ok=True)
            mode_tag = f"{summary['search_mode']}_rerank{summary['cross_encoder_rerank']}"
            fname = f"retrieval_bench_{mode_tag}_{int(time.time())}.json"
            out_path = save_dir / fname
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not persist retrieval benchmark: {e}")

    @classmethod
    def list_saved_retrieval_benchmarks(cls) -> List[Dict[str, Any]]:
        save_dir = config.RETRIEVAL_BENCHMARK_DIR
        if not save_dir.exists():
            return []
        benchmarks = []
        for p in sorted(save_dir.glob("retrieval_bench_*.json"), reverse=True):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    benchmarks.append(json.load(f))
            except Exception:
                pass
        return benchmarks
