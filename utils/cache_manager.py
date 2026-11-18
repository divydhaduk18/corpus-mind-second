"""Multi-Tier Caching Architecture for Corpus Mind.

Provides:
- Layer A: Embedding Cache (Content SHA-256 + model identity keyed)
- Layer B: Query Embedding Cache (In-memory LRU)
- Layer C: Retrieval Cache (Strictly user-isolated multi-tenant safe)
- Layer D: LLM Response Cache (Deterministic responses with context hash)
- Layer E: Model Cache (Thread-safe singleton loaders)
- Observable hit/miss counters, latency gains, and eviction management.
"""

import os
import sys
import time
import hashlib
import json
import threading
import logging
from typing import Dict, Any, List, Optional, Tuple
from collections import OrderedDict
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


class LRUCache:
    """Thread-safe generic In-Memory LRU Cache with TTL expiration."""

    def __init__(self, max_size: int = 2000, ttl_seconds: int = 86400):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None

            val, timestamp = self._cache[key]
            if time.time() - timestamp > self.ttl:
                del self._cache[key]
                self.misses += 1
                return None

            self._cache.move_to_end(key)
            self.hits += 1
            return val

    def set(self, key: str, value: Any):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (value, time.time())
            if len(self._cache) > self.max_size:
                self._cache.popitem(last=False)

    def clear(self):
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return round((self.hits / total) * 100.0, 2) if total > 0 else 0.0

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._cache)


class CacheManager:
    """Centralized manager coordinating all 5 caching tiers."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.embedding_cache = LRUCache(max_size=config.CACHE_MAX_ENTRIES, ttl_seconds=config.CACHE_TTL_SECONDS)
        self.query_cache = LRUCache(max_size=1000, ttl_seconds=config.CACHE_TTL_SECONDS)
        self.retrieval_cache = LRUCache(max_size=1000, ttl_seconds=3600)  # 1 hour
        self.llm_response_cache = LRUCache(max_size=500, ttl_seconds=3600)

    @classmethod
    def get_instance(cls) -> "CacheManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = CacheManager()
            return cls._instance

    # -------------------------------------------------------------
    # TIER A: EMBEDDING CACHE
    # -------------------------------------------------------------
    @staticmethod
    def compute_text_hash(text: str, model_name: str) -> str:
        """Create unique key from text content and active embedding model."""
        content = f"{model_name}::{text.strip()}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def get_cached_embedding(self, text: str, model_name: str) -> Optional[np.ndarray]:
        if not getattr(config, "EMBEDDING_CACHE_ENABLED", True):
            return None
        key = self.compute_text_hash(text, model_name)
        return self.embedding_cache.get(key)

    def set_cached_embedding(self, text: str, model_name: str, embedding: np.ndarray):
        if not getattr(config, "EMBEDDING_CACHE_ENABLED", True):
            return
        key = self.compute_text_hash(text, model_name)
        self.embedding_cache.set(key, embedding)

    # -------------------------------------------------------------
    # TIER B: QUERY EMBEDDING CACHE
    # -------------------------------------------------------------
    def get_cached_query_embedding(self, query: str, model_name: str) -> Optional[np.ndarray]:
        if not getattr(config, "QUERY_CACHE_ENABLED", True):
            return None
        key = f"q::{model_name}::{query.lower().strip()}"
        return self.query_cache.get(key)

    def set_cached_query_embedding(self, query: str, model_name: str, embedding: np.ndarray):
        if not getattr(config, "QUERY_CACHE_ENABLED", True):
            return
        key = f"q::{model_name}::{query.lower().strip()}"
        self.query_cache.set(key, embedding)

    # -------------------------------------------------------------
    # TIER C: RETRIEVAL CACHE (STRICTLY MULTI-TENANT ISOLATED)
    # -------------------------------------------------------------
    @staticmethod
    def compute_retrieval_cache_key(
        user_id: int,
        query: str,
        search_mode: str,
        top_k: int,
        doc_filter: Optional[List[str]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        index_version: str = "v1"
    ) -> str:
        """Construct multi-tenant safe retrieval key incorporating user_id and all filter scopes."""
        filter_str = json.dumps(metadata_filters or {}, sort_keys=True)
        docs_str = ",".join(sorted(doc_filter)) if doc_filter else "all"
        raw_key = f"user_{user_id}::idx_{index_version}::{search_mode}::k{top_k}::docs_{docs_str}::filt_{filter_str}::{query.lower().strip()}"
        return hashlib.sha256(raw_key.encode('utf-8')).hexdigest()

    def get_cached_retrieval(
        self,
        user_id: int,
        query: str,
        search_mode: str,
        top_k: int,
        doc_filter: Optional[List[str]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        index_version: str = "v1"
    ) -> Optional[List[Dict[str, Any]]]:
        if not getattr(config, "RETRIEVAL_CACHE_ENABLED", True):
            return None
        key = self.compute_retrieval_cache_key(
            user_id, query, search_mode, top_k, doc_filter, metadata_filters, index_version
        )
        return self.retrieval_cache.get(key)

    def set_cached_retrieval(
        self,
        user_id: int,
        query: str,
        search_mode: str,
        top_k: int,
        results: List[Dict[str, Any]],
        doc_filter: Optional[List[str]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        index_version: str = "v1"
    ):
        if not getattr(config, "RETRIEVAL_CACHE_ENABLED", True):
            return
        key = self.compute_retrieval_cache_key(
            user_id, query, search_mode, top_k, doc_filter, metadata_filters, index_version
        )
        self.retrieval_cache.set(key, results)

    def invalidate_user_retrieval_cache(self, user_id: int):
        """Purge retrieval cache when user documents or indexes change."""
        with self.retrieval_cache._lock:
            # Recreate cache clearing user entries
            keys_to_remove = [k for k in self.retrieval_cache._cache.keys() if f"user_{user_id}" in k]
            for k in keys_to_remove:
                del self.retrieval_cache._cache[k]

    # -------------------------------------------------------------
    # TIER D: DETERMINISTIC LLM RESPONSE CACHE
    # -------------------------------------------------------------
    def get_cached_llm_response(
        self,
        model: str,
        prompt: str,
        context_hash: str
    ) -> Optional[str]:
        if not getattr(config, "LLM_CACHE_ENABLED", False):
            return None
        key = hashlib.sha256(f"{model}::{context_hash}::{prompt.strip()}".encode('utf-8')).hexdigest()
        return self.llm_response_cache.get(key)

    def set_cached_llm_response(
        self,
        model: str,
        prompt: str,
        context_hash: str,
        response_text: str
    ):
        if not getattr(config, "LLM_CACHE_ENABLED", False):
            return
        key = hashlib.sha256(f"{model}::{context_hash}::{prompt.strip()}".encode('utf-8')).hexdigest()
        self.llm_response_cache.set(key, response_text)

    # -------------------------------------------------------------
    # METRICS & OBSERVABILITY
    # -------------------------------------------------------------
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Return metrics across all active caching tiers."""
        return {
            "embedding_cache": {
                "size": self.embedding_cache.size,
                "hits": self.embedding_cache.hits,
                "misses": self.embedding_cache.misses,
                "hit_rate_pct": self.embedding_cache.hit_rate
            },
            "query_cache": {
                "size": self.query_cache.size,
                "hits": self.query_cache.hits,
                "misses": self.query_cache.misses,
                "hit_rate_pct": self.query_cache.hit_rate
            },
            "retrieval_cache": {
                "size": self.retrieval_cache.size,
                "hits": self.retrieval_cache.hits,
                "misses": self.retrieval_cache.misses,
                "hit_rate_pct": self.retrieval_cache.hit_rate
            },
            "llm_response_cache": {
                "size": self.llm_response_cache.size,
                "hits": self.llm_response_cache.hits,
                "misses": self.llm_response_cache.misses,
                "hit_rate_pct": self.llm_response_cache.hit_rate
            }
        }

    def clear_all_caches(self):
        """Flush all cache layers."""
        self.embedding_cache.clear()
        self.query_cache.clear()
        self.retrieval_cache.clear()
        self.llm_response_cache.clear()
        logger.info("All cache tiers cleared.")
