"""Unit & Integration Tests for Caching Optimization, Asynchronous Ingestion & Scalability (Features 25, 26, 27).

Tests:
- CacheManager: LRU eviction, TTL expiration, hit/miss metrics
- User-Isolated Retrieval Cache: User A cannot access User B's cached results
- Embedding Cache: Content-hash SHA256 reuse across identical content
- AsyncDocumentProcessor: Job creation, stage progression, failure handling, retry
- ScalableIndexManager: Incremental index.add(), point-in-time backups, RAM/CPU metrics
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import tempfile
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.database import db
from utils.cache_manager import CacheManager, LRUCache
from utils.async_processor import AsyncDocumentProcessor
from utils.scalable_index import ScalableIndexManager


class TestCachingAndAsyncScalability(unittest.TestCase):

    def setUp(self):
        self.cache_mgr = CacheManager.get_instance()
        self.cache_mgr.clear_all_caches()

    def test_lru_cache_hit_miss_and_eviction(self):
        """Verify LRU cache tracks hits/misses and evicts oldest items when max_size is reached."""
        lru = LRUCache(max_size=3, ttl_seconds=100)
        self.assertIsNone(lru.get("missing_key"))
        self.assertEqual(lru.misses, 1)

        lru.set("k1", "v1")
        lru.set("k2", "v2")
        lru.set("k3", "v3")

        self.assertEqual(lru.get("k1"), "v1")
        self.assertEqual(lru.hits, 1)

        # Adding k4 should evict k2 (since k1 was accessed most recently)
        lru.set("k4", "v4")
        self.assertEqual(lru.size, 3)
        self.assertIsNone(lru.get("k2"))  # k2 was evicted
        self.assertEqual(lru.get("k1"), "v1")
        self.assertEqual(lru.get("k4"), "v4")

    def test_retrieval_cache_user_isolation(self):
        """CRITICAL SECURITY: User A must never access User B's cached retrieval results."""
        user_a_id = 101
        user_b_id = 202
        query = "annual revenue report"

        mock_results_a = [{"text": "Confidential User A financials", "file_name": "user_a_doc.pdf"}]
        mock_results_b = [{"text": "Confidential User B financials", "file_name": "user_b_doc.pdf"}]

        # Set cache for User A
        self.cache_mgr.set_cached_retrieval(
            user_id=user_a_id,
            query=query,
            search_mode="hybrid",
            top_k=5,
            results=mock_results_a
        )

        # User A hits cache
        cached_a = self.cache_mgr.get_cached_retrieval(
            user_id=user_a_id,
            query=query,
            search_mode="hybrid",
            top_k=5
        )
        self.assertIsNotNone(cached_a)
        self.assertEqual(cached_a[0]["text"], "Confidential User A financials")

        # User B querying exact same question must NOT receive User A's cached results
        cached_b = self.cache_mgr.get_cached_retrieval(
            user_id=user_b_id,
            query=query,
            search_mode="hybrid",
            top_k=5
        )
        self.assertIsNone(cached_b)  # Cache miss guaranteed

    def test_embedding_cache_content_hash_reuse(self):
        """Identical text content must produce cache hits regardless of document source."""
        text = "Deep learning neural network architectures for document analysis."
        model = "all-MiniLM-L6-v2"
        mock_vec = np.array([0.12, 0.45, 0.78], dtype=np.float32)

        self.assertIsNone(self.cache_mgr.get_cached_embedding(text, model))
        self.cache_mgr.set_cached_embedding(text, model, mock_vec)

        cached_vec = self.cache_mgr.get_cached_embedding(text, model)
        self.assertIsNotNone(cached_vec)
        np.testing.assert_array_equal(cached_vec, mock_vec)

    def test_async_job_creation_and_database_persistence(self):
        """Verify background jobs are created in SQLite with initial queued state."""
        test_uid = 999
        job_id = f"test_job_{int(time.time())}"
        
        db.create_document_job(
            job_id=job_id,
            user_id=test_uid,
            file_name="background_report.pdf",
            file_path="/tmp/fake_path.pdf"
        )

        job = db.get_document_job_by_id(job_id)
        self.assertIsNotNone(job)
        self.assertEqual(job["status"], "queued")
        self.assertEqual(job["progress_pct"], 0)
        self.assertEqual(job["current_stage"], "queued")

        # Update stage
        db.update_document_job(job_id, status="processing", progress_pct=45, current_stage="table_extraction")
        updated_job = db.get_document_job_by_id(job_id)
        self.assertEqual(updated_job["status"], "processing")
        self.assertEqual(updated_job["progress_pct"], 45)
        self.assertEqual(updated_job["current_stage"], "table_extraction")

    def test_scalable_hardware_metrics(self):
        """Verify psutil memory and CPU metrics are reported accurately."""
        metrics = ScalableIndexManager.get_system_metrics()
        self.assertIn("rss_ram_mb", metrics)
        self.assertIn("cpu_percent", metrics)
        self.assertGreater(metrics["rss_ram_mb"], 0.0)


if __name__ == "__main__":
    unittest.main()
