import os
import sys
import pickle
import math
import re
import time
import hashlib
import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.embeddings import EmbeddingManager, EmbeddingDimensionMismatchError
from utils.database import db
from utils.reranker import CrossEncoderReranker
from utils.query_processor import QueryProcessor
from utils.context_compressor import ContextualCompressor

logger = logging.getLogger(__name__)

# Attempt to import faiss
try:
    import faiss
    FAISS_AVAILABLE = True
    logger.info("FAISS library detected and available.")
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS library not found. Native optimized NumPy vector store will be used.")


class BM25Retriever:
    """High-performance Okapi BM25 retriever with inverted index postings."""

    def __init__(self, corpus_texts: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_texts = corpus_texts
        self.N = len(corpus_texts)
        self.doc_tokens = [self._tokenize(t) for t in corpus_texts]
        self.doc_lens = np.array([len(tokens) for tokens in self.doc_tokens], dtype=np.float32)
        self.avgdl = float(np.mean(self.doc_lens)) if len(self.doc_lens) > 0 else 1.0

        # Build Inverted Index: token -> list of (doc_idx, term_frequency)
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = {}
        for doc_idx, tokens in enumerate(self.doc_tokens):
            counts: Dict[str, int] = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            for t, tf in counts.items():
                if t not in self.inverted_index:
                    self.inverted_index[t] = []
                self.inverted_index[t].append((doc_idx, tf))

        # Precompute IDF for all terms in vocabulary
        self.idf: Dict[str, float] = {}
        for term, postings in self.inverted_index.items():
            df = len(postings)
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def search(self, query: str, top_k: int = 20) -> List[Tuple[float, int]]:
        query_tokens = self._tokenize(query)
        if not query_tokens or self.N == 0:
            return []

        scores = np.zeros(self.N, dtype=np.float32)

        # Inverted index traversal: only touch documents containing query words
        for token in query_tokens:
            if token not in self.inverted_index:
                continue
            token_idf = self.idf[token]
            postings = self.inverted_index[token]

            for doc_idx, tf in postings:
                dl = self.doc_lens[doc_idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
                num = tf * (self.k1 + 1.0)
                scores[doc_idx] += token_idf * (num / denom)

        # Retrieve top scores
        nonzero_indices = np.where(scores > 0)[0]
        if len(nonzero_indices) == 0:
            return []

        if len(nonzero_indices) > top_k:
            part_idx = np.argpartition(scores[nonzero_indices], -top_k)[-top_k:]
            sorted_candidates = nonzero_indices[part_idx[np.argsort(scores[nonzero_indices[part_idx]])[::-1]]]
        else:
            sorted_candidates = nonzero_indices[np.argsort(scores[nonzero_indices])[::-1]]

        results = [(float(scores[idx]), int(idx)) for idx in sorted_candidates if scores[idx] > 0]
        return results


class NumpyVectorStore:
    """Pre-normalized 2D matrix vector store with vectorized cosine search."""

    def __init__(self):
        self.matrix: Optional[np.ndarray] = None  # shape: (N, D), pre-normalized float32
        self.metadata: List[Dict[str, Any]] = []

    def set_vectors(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self.matrix = embeddings / norms
        self.metadata = metadata

    def search(self, query_emb: np.ndarray, k: int) -> List[Tuple[float, int]]:
        if self.matrix is None or len(self.matrix) == 0:
            return []

        q = np.asarray(query_emb, dtype=np.float32).flatten()
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q_unit = q / q_norm

        # Cosine similarity via single matrix dot product
        similarities = np.dot(self.matrix, q_unit)
        top_k = min(k, len(similarities))

        if len(similarities) > top_k:
            part_idx = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = part_idx[np.argsort(similarities[part_idx])[::-1]]
        else:
            top_indices = np.argsort(similarities)[::-1]

        return [(float(similarities[idx]), int(idx)) for idx in top_indices]

    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump((self.matrix, self.metadata), f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, filepath: str):
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.matrix, self.metadata = pickle.load(f)


class CachedRetrievalState:
    """In-memory cache capsule for a user's index."""

    def __init__(
        self,
        user_id: int,
        version: str,
        embedding_model: str,
        dimension: int,
        metadata: List[Dict[str, Any]],
        bm25: Optional[BM25Retriever],
        faiss_index: Any,
        numpy_store: Optional[NumpyVectorStore],
        file_mtime: float
    ):
        self.user_id = user_id
        self.version = version
        self.embedding_model = embedding_model
        self.dimension = dimension
        self.metadata = metadata
        self.bm25 = bm25
        self.faiss_index = faiss_index
        self.numpy_store = numpy_store
        self.file_mtime = file_mtime
        self.last_accessed = time.time()


class RAGManager:
    """Manages multi-tenant RAG indexing, in-memory caching, and hybrid retrieval."""

    # In-memory retrieval state cache keyed strictly by user_id
    _user_cache: Dict[int, CachedRetrievalState] = {}

    @classmethod
    def get_user_index_path(cls, user_id: int) -> str:
        return str(config.get_user_vector_store_dir(user_id) / "vector_index.bin")

    @classmethod
    def get_user_metadata_path(cls, user_id: int) -> str:
        return str(config.get_user_vector_store_dir(user_id) / "vector_metadata.pkl")

    @classmethod
    def invalidate_cache(cls, user_id: Optional[int] = None):
        """Invalidate in-memory cached state for a specific user or all users."""
        if user_id is not None:
            cls._user_cache.pop(user_id, None)
            logger.info(f"Invalidated in-memory retrieval cache for user_id={user_id}.")
        else:
            cls._user_cache.clear()
            logger.info("Cleared all in-memory retrieval caches.")

    @classmethod
    def load_user_retrieval_state(cls, user_id: int) -> Optional[CachedRetrievalState]:
        """Load or retrieve the in-memory cached state for a specific user.
        
        Loads once per index version. Detects updates and reloads safely.
        """
        meta_path = cls.get_user_metadata_path(user_id)
        index_path = cls.get_user_index_path(user_id)

        if not os.path.exists(meta_path) or not os.path.exists(index_path):
            cls._user_cache.pop(user_id, None)
            return None

        current_mtime = max(os.path.getmtime(meta_path), os.path.getmtime(index_path))

        # Check in-memory cache validity
        cached = cls._user_cache.get(user_id)
        if cached is not None and cached.file_mtime >= current_mtime:
            cached.last_accessed = time.time()
            return cached

        # Load from disk
        logger.info(f"Loading retrieval index state from disk into memory for user_id={user_id}...")
        try:
            with open(meta_path, 'rb') as f:
                meta_payload = pickle.load(f)

            if isinstance(meta_payload, dict):
                metadata = meta_payload.get("chunks", [])
                stored_model = meta_payload.get("embedding_model", config.DEFAULT_EMBEDDING_MODEL)
                dimension = meta_payload.get("dimension", 384)
                version = meta_payload.get("version", str(current_mtime))
            else:
                # Legacy metadata list format
                metadata = meta_payload
                stored_model = getattr(config, "DEFAULT_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
                dimension = EmbeddingManager.get_model_dimension(stored_model)
                version = str(current_mtime)

            if not metadata:
                return None

            # Validate compatibility with currently configured embedding model
            EmbeddingManager.validate_compatibility(stored_model, dimension)

            # Build BM25 retriever once
            corpus_texts = [m['text'] for m in metadata]
            bm25 = BM25Retriever(corpus_texts)

            # Load Dense vector index
            faiss_index = None
            numpy_store = None

            if FAISS_AVAILABLE:
                try:
                    faiss_index = faiss.read_index(index_path)
                except Exception as fe:
                    logger.warning(f"Failed to read FAISS index for user {user_id}, falling back to numpy: {fe}")
                    faiss_index = None

            if faiss_index is None:
                numpy_store = NumpyVectorStore()
                numpy_store.load(index_path)

            cached_state = CachedRetrievalState(
                user_id=user_id,
                version=version,
                embedding_model=stored_model,
                dimension=dimension,
                metadata=metadata,
                bm25=bm25,
                faiss_index=faiss_index,
                numpy_store=numpy_store,
                file_mtime=current_mtime
            )

            cls._user_cache[user_id] = cached_state
            return cached_state

        except Exception as e:
            logger.error(f"Error loading retrieval state for user_id={user_id}: {e}", exc_info=True)
            return None

    @classmethod
    def index_user_chunks(cls, user_id: int, chunks: List[Dict[str, Any]]) -> bool:
        """Create/rebuild vector and BM25 index strictly for user_id with batching and validation."""
        if not chunks:
            logger.warning(f"No chunks provided to index for user {user_id}.")
            return False

        try:
            texts = [c['text'] for c in chunks]
            model_name = getattr(config, "DEFAULT_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            expected_dim = EmbeddingManager.get_model_dimension(model_name)

            # Generate dense embeddings in batches with strict validation
            embeddings = EmbeddingManager.get_embeddings(texts, model_name=model_name, batch_size=32)
            if embeddings.shape[0] != len(chunks) or embeddings.shape[1] != expected_dim:
                raise ValueError(
                    f"Generated embeddings shape {embeddings.shape} does not match ({len(chunks)}, {expected_dim})."
                )

            # Assign sequential chunk_id and enforce user_id, parent_id, category in every metadata record
            metadata_records = []
            for idx, c in enumerate(chunks):
                record = {
                    "chunk_id": idx,
                    "doc_id": c.get("doc_id"),
                    "user_id": user_id,
                    "file_name": c.get("file_name", "unknown"),
                    "page_number": c.get("page_number", 1),
                    "section": c.get("section", "General"),
                    "file_type": c.get("file_type", "Document"),
                    "category": c.get("category", "General"),
                    "parent_id": c.get("parent_id", f"p_{idx}"),
                    "parent_text": c.get("parent_text", c["text"]),
                    "text": c["text"]
                }
                metadata_records.append(record)

            index_path = cls.get_user_index_path(user_id)
            meta_path = cls.get_user_metadata_path(user_id)

            if FAISS_AVAILABLE:
                norm_embeddings = embeddings.copy()
                faiss.normalize_L2(norm_embeddings)
                index = faiss.IndexFlatIP(expected_dim)
                index.add(norm_embeddings)
                faiss.write_index(index, index_path)
            else:
                store = NumpyVectorStore()
                store.set_vectors(embeddings, metadata_records)
                store.save(index_path)

            # Save versioned metadata bundle
            version_id = hashlib.sha256(f"{user_id}_{time.time()}_{len(chunks)}".encode()).hexdigest()[:16]
            meta_payload = {
                "version": version_id,
                "user_id": user_id,
                "embedding_model": model_name,
                "dimension": expected_dim,
                "chunks": metadata_records
            }
            with open(meta_path, "wb") as f:
                pickle.dump(meta_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Invalidate in-memory cache to force immediate reload
            cls.invalidate_cache(user_id)
            logger.info(f"User {user_id}: successfully indexed {len(chunks)} chunks with model '{model_name}'.")
            return True

        except Exception as e:
            logger.error(f"Failed to index documents for user {user_id}: {e}", exc_info=True)
            return False

    @classmethod
    def rebuild_user_index(cls, user_id: int, strategy: Optional[str] = None) -> Tuple[bool, int, int]:
        """Rebuild vector & BM25 index exclusively for a specific authenticated user."""
        from utils.document_loader import DocumentLoader
        from utils.text_splitter import RecursiveTextSplitter

        user_docs = db.get_documents_by_user(user_id)
        if not user_docs:
            for path in [cls.get_user_index_path(user_id), cls.get_user_metadata_path(user_id)]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass
            cls.invalidate_cache(user_id)
            return True, 0, 0

        resolved_strategy = strategy
        if resolved_strategy is None:
            if getattr(config, "PARENT_CHILD_ENABLED", True):
                resolved_strategy = "hierarchical"
            else:
                resolved_strategy = getattr(config, "DEFAULT_CHUNKING_STRATEGY", "semantic")

        splitter = RecursiveTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
        all_chunks = []

        for doc in user_docs:
            f_path = doc['file_path']
            if not f_path or not os.path.exists(f_path):
                continue

            # Load page-structured data
            pages = DocumentLoader.load_document_pages(f_path)
            ext = os.path.splitext(doc['file_name'])[1].lower()
            file_type = "Image OCR" if ext in config.SUPPORTED_IMAGE_EXTENSIONS else "Document"
            category = doc.get("category", "General") if isinstance(doc, dict) else "General"

            chunks = splitter.split_pages(
                pages=pages,
                doc_id=doc['id'],
                user_id=user_id,
                file_name=doc['file_name'],
                file_type=file_type,
                strategy=resolved_strategy
            )
            for c in chunks:
                c["category"] = category
            all_chunks.extend(chunks)

            # Ingest extracted PDF tables as dedicated high-saliency RAG chunks
            if ext == ".pdf" and getattr(config, "ENABLE_TABLE_EXTRACTION_ON_UPLOAD", True):
                try:
                    from utils.table_extractor import TableExtractor
                    tables = TableExtractor.extract_tables_from_pdf(f_path)
                    for t in tables:
                        t_passage = t.to_rag_passage()
                        if t_passage.strip():
                            all_chunks.append({
                                "text": t_passage,
                                "doc_id": doc['id'],
                                "user_id": user_id,
                                "file_name": doc['file_name'],
                                "page_number": t.page_number,
                                "section": f"Table {t.table_id}",
                                "file_type": "Table Data",
                                "category": category
                            })
                except Exception as t_err:
                    logger.debug(f"Table ingestion error for {doc['file_name']}: {t_err}")

            # Ingest extracted PDF figures / charts as visual RAG chunks
            if ext == ".pdf" and getattr(config, "ENABLE_IMAGE_EXTRACTION_ON_UPLOAD", True):
                try:
                    from utils.visual_rag import VisualRAGManager
                    figures = VisualRAGManager.extract_figures_from_pdf(f_path, user_id=user_id, generate_descriptions=True)
                    for fig in figures:
                        f_passage = fig.to_rag_passage()
                        if f_passage.strip():
                            all_chunks.append({
                                "text": f_passage,
                                "doc_id": doc['id'],
                                "user_id": user_id,
                                "file_name": doc['file_name'],
                                "page_number": fig.page_number,
                                "section": f"Figure {fig.figure_id} ({fig.figure_type})",
                                "file_type": "Visual Diagram",
                                "category": category
                            })
                except Exception as f_err:
                    logger.debug(f"Visual figure ingestion error for {doc['file_name']}: {f_err}")

        if not all_chunks:
            cls.invalidate_cache(user_id)
            return True, len(user_docs), 0

        success = cls.index_user_chunks(user_id, all_chunks)
        return success, len(user_docs), len(all_chunks)

    @classmethod
    def rebuild_index_from_db(cls, user_id: Optional[int] = None) -> Tuple[bool, int, int]:
        """Rebuild index with user-scoping.
        
        If user_id is provided, rebuilds that user's store.
        If user_id is None (admin context), rebuilds each user's store independently without mixing.
        """
        if user_id is not None:
            return cls.rebuild_user_index(user_id)

        # Admin global rebuild: iterate each user and build isolated user index
        users = db.get_all_users()
        total_docs = 0
        total_chunks = 0
        overall_success = True

        for u in users:
            s, d_c, c_c = cls.rebuild_user_index(u['id'])
            if not s:
                overall_success = False
            total_docs += d_c
            total_chunks += c_c

        return overall_success, total_docs, total_chunks

    @staticmethod
    def _deduplicate_chunks(chunks: List[Dict[str, Any]], max_items: int) -> List[Dict[str, Any]]:
        """Deduplicate passages with high string overlap or identical content."""
        deduped = []
        seen_texts = set()

        for chunk in chunks:
            # Normalized fingerprint: lowercase alphanumeric only
            fingerprint = re.sub(r'\W+', '', chunk['text'].lower())
            if not fingerprint or fingerprint in seen_texts:
                continue

            # Check if this fingerprint is a high-overlap substring of an existing chunk
            is_redundant = False
            for seen in seen_texts:
                if len(fingerprint) > 40 and len(seen) > 40:
                    if fingerprint in seen or seen in fingerprint:
                        is_redundant = True
                        break

            if not is_redundant:
                seen_texts.add(fingerprint)
                deduped.append(chunk)

            if len(deduped) >= max_items:
                break

        return deduped

    @staticmethod
    def _matches_filters(
        item: Dict[str, Any], 
        doc_filter: Optional[List[str]] = None, 
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Evaluate whether a metadata item satisfies document name and metadata filters."""
        if doc_filter and item.get('file_name') not in doc_filter:
            return False

        if not metadata_filters:
            return True

        for key, val in metadata_filters.items():
            if val is None or val == "" or val == []:
                continue

            if key == "file_name":
                names = [val] if isinstance(val, str) else val
                if item.get("file_name") not in names:
                    return False
            elif key == "doc_id":
                ids = [val] if isinstance(val, int) else val
                if item.get("doc_id") not in ids:
                    return False
            elif key == "page_number":
                if isinstance(val, tuple) and len(val) == 2:
                    if not (val[0] <= item.get("page_number", 1) <= val[1]):
                        return False
                else:
                    pages = [val] if isinstance(val, int) else val
                    if item.get("page_number") not in pages:
                        return False
            elif key == "file_type":
                types = [val] if isinstance(val, str) else val
                if item.get("file_type") not in types:
                    return False
            elif key == "category":
                cats = [val.lower()] if isinstance(val, str) else [c.lower() for c in val]
                if item.get("category", "General").lower() not in cats:
                    return False

        return True

    @classmethod
    def _retrieve_single_query(
        cls,
        state: CachedRetrievalState,
        query: str,
        candidate_k: int = 20,
        similarity_threshold: float = 0.15,
        search_mode: str = "hybrid",
        fusion_method: str = "rrf",
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        doc_filter: Optional[List[str]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve candidate chunks for a single search query."""
        # 1. Sparse BM25 Search
        bm25_results = state.bm25.search(query, top_k=candidate_k) if state.bm25 else []

        # 2. Dense Vector Search
        dense_results: List[Tuple[float, int]] = []
        try:
            query_emb = EmbeddingManager.get_embedding(query, model_name=state.embedding_model)
            if state.faiss_index is not None:
                q_emb = query_emb.reshape(1, -1).copy()
                faiss.normalize_L2(q_emb)
                scores, indices = state.faiss_index.search(q_emb, candidate_k)
                for score, idx in zip(scores[0], indices[0]):
                    if idx != -1 and idx < len(state.metadata):
                        dense_results.append((float(score), int(idx)))
            elif state.numpy_store is not None:
                dense_results = state.numpy_store.search(query_emb, candidate_k)
        except Exception as dense_err:
            logger.error(f"Dense search error for query '{query}': {dense_err}")

        # 3. Filter candidates by metadata & document name
        filtered_dense = [
            (score, idx) for score, idx in dense_results 
            if cls._matches_filters(state.metadata[idx], doc_filter=doc_filter, metadata_filters=metadata_filters)
        ]
        filtered_bm25 = [
            (score, idx) for score, idx in bm25_results 
            if cls._matches_filters(state.metadata[idx], doc_filter=doc_filter, metadata_filters=metadata_filters)
        ]

        raw_candidates = []

        if search_mode == "bm25":
            for score, idx in filtered_bm25:
                item = state.metadata[idx].copy()
                item['score'] = score
                item['search_mode'] = 'BM25 Sparse'
                raw_candidates.append(item)
            return raw_candidates

        if search_mode == "dense":
            for score, idx in filtered_dense:
                if score < similarity_threshold:
                    continue
                item = state.metadata[idx].copy()
                item['score'] = score
                item['search_mode'] = 'Dense Vector'
                raw_candidates.append(item)
            return raw_candidates

        # Hybrid Search Fusion: RRF or Weighted Normalized Score Fusion
        dense_map = {idx: s for s, idx in filtered_dense}
        bm25_map = {idx: s for s, idx in filtered_bm25}
        all_candidate_indices = set(dense_map.keys()).union(set(bm25_map.keys()))

        if fusion_method == "weighted" and all_candidate_indices:
            # Min-Max Normalization of dense and BM25 scores
            d_scores = [dense_map.get(idx, 0.0) for idx in all_candidate_indices]
            b_scores = [bm25_map.get(idx, 0.0) for idx in all_candidate_indices]

            min_d, max_d = min(d_scores), max(d_scores)
            min_b, max_b = min(b_scores), max(b_scores)

            fused_candidates = []
            for idx in all_candidate_indices:
                raw_d = dense_map.get(idx, 0.0)
                raw_b = bm25_map.get(idx, 0.0)

                norm_d = (raw_d - min_d) / (max_d - min_d + 1e-6) if max_d > min_d else raw_d
                norm_b = (raw_b - min_b) / (max_b - min_b + 1e-6) if max_b > min_b else (1.0 if raw_b > 0 else 0.0)

                combined_score = (dense_weight * norm_d) + (sparse_weight * norm_b)
                item = state.metadata[idx].copy()
                item['score'] = combined_score
                item['dense_score'] = raw_d
                item['bm25_score'] = raw_b
                item['search_mode'] = 'Hybrid Weighted Normalized'
                fused_candidates.append(item)

            fused_candidates.sort(key=lambda x: x['score'], reverse=True)
            return fused_candidates

        # Default Hybrid: Reciprocal Rank Fusion (RRF)
        rrf_k = getattr(config, 'RRF_K', 60)
        rrf_scores: Dict[int, float] = {}

        for rank, (score, idx) in enumerate(filtered_dense, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_k + rank))

        for rank, (score, idx) in enumerate(filtered_bm25, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_k + rank))

        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        for idx, rrf_score in sorted_rrf:
            item = state.metadata[idx].copy()
            item['score'] = rrf_score
            item['dense_score'] = dense_map.get(idx, 0.0)
            item['bm25_score'] = bm25_map.get(idx, 0.0)
            item['search_mode'] = 'Hybrid RRF'
            raw_candidates.append(item)

        return raw_candidates

    @classmethod
    def retrieve_context(
        cls, 
        query: str, 
        user_id: int,
        top_k: int = 4, 
        similarity_threshold: float = 0.15,
        search_mode: str = "hybrid",
        doc_filter: Optional[List[str]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        enable_reranking: Optional[bool] = None,
        reranker_model: Optional[str] = None,
        fusion_method: Optional[str] = None,
        dense_weight: Optional[float] = None,
        sparse_weight: Optional[float] = None,
        enable_multi_query: Optional[bool] = None,
        multi_query_count: Optional[int] = None,
        enable_parent_child: Optional[bool] = None,
        enable_contextual_compression: Optional[bool] = None,
        compression_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant context chunks strictly for the authenticated user_id using cached state.
        
        Executes the full upgraded 10-step Advanced RAG pipeline:
        Multi-Query -> Metadata Filtering -> Dense+Sparse Retrieval -> RRF/Weighted Fusion
        -> Cross-Encoder Reranking -> Parent-Child Expansion -> Contextual Compression.
        """
        if user_id is None:
            logger.error("Security violation: retrieve_context called without authenticated user_id.")
            return []

        if not query or not query.strip():
            return []

        state = cls.load_user_retrieval_state(user_id)
        if not state or not state.metadata:
            logger.info(f"No indexed documents found in state for user_id={user_id}.")
            return []

        # Resolve configuration defaults
        use_rerank = (
            enable_reranking if enable_reranking is not None 
            else getattr(config, "RERANKING_ENABLED", True)
        )
        use_multi_q = (
            enable_multi_query if enable_multi_query is not None 
            else getattr(config, "MULTI_QUERY_ENABLED", True)
        )
        num_multi_q = (
            multi_query_count if multi_query_count is not None 
            else getattr(config, "MULTI_QUERY_COUNT", 3)
        )
        use_parent_child = (
            enable_parent_child if enable_parent_child is not None 
            else getattr(config, "PARENT_CHILD_ENABLED", True)
        )
        use_compression = (
            enable_contextual_compression if enable_contextual_compression is not None 
            else getattr(config, "CONTEXTUAL_COMPRESSION_ENABLED", True)
        )
        resolved_fusion = fusion_method or getattr(config, "HYBRID_FUSION_METHOD", "rrf")
        w_dense = dense_weight if dense_weight is not None else getattr(config, "DENSE_SEARCH_WEIGHT", 0.5)
        w_sparse = sparse_weight if sparse_weight is not None else getattr(config, "SPARSE_SEARCH_WEIGHT", 0.5)

        candidate_k = max(top_k * 4, 20)

        # 1. Multi-Query Expansion
        if use_multi_q and search_mode in ("hybrid", "dense", "bm25"):
            query_variations = QueryProcessor.generate_multi_queries(query, num_queries=num_multi_q)
        else:
            query_variations = [query.strip()]

        # 2. Candidate Retrieval across all query variations with RRF aggregation
        if len(query_variations) == 1:
            candidates = cls._retrieve_single_query(
                state=state,
                query=query_variations[0],
                candidate_k=candidate_k,
                similarity_threshold=similarity_threshold,
                search_mode=search_mode,
                fusion_method=resolved_fusion,
                dense_weight=w_dense,
                sparse_weight=w_sparse,
                doc_filter=doc_filter,
                metadata_filters=metadata_filters
            )
        else:
            # Multi-query RRF combination
            rrf_k = getattr(config, 'RRF_K', 60)
            multi_q_scores: Dict[int, float] = {}
            candidate_item_map: Dict[int, Dict[str, Any]] = {}

            for q_var in query_variations:
                q_candidates = cls._retrieve_single_query(
                    state=state,
                    query=q_var,
                    candidate_k=candidate_k,
                    similarity_threshold=similarity_threshold,
                    search_mode=search_mode,
                    fusion_method=resolved_fusion,
                    dense_weight=w_dense,
                    sparse_weight=w_sparse,
                    doc_filter=doc_filter,
                    metadata_filters=metadata_filters
                )
                for rank, c in enumerate(q_candidates, start=1):
                    cid = c['chunk_id']
                    multi_q_scores[cid] = multi_q_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))
                    if cid not in candidate_item_map:
                        candidate_item_map[cid] = c

            sorted_multi_q = sorted(multi_q_scores.items(), key=lambda x: x[1], reverse=True)
            candidates = []
            for cid, score in sorted_multi_q:
                item = candidate_item_map[cid].copy()
                item['score'] = score
                item['search_mode'] = f"{item.get('search_mode', 'Hybrid')} + MultiQuery"
                candidates.append(item)

        if not candidates:
            return []

        # 3. Cross-Encoder Reranking
        if use_rerank and len(candidates) > 1:
            candidates = CrossEncoderReranker.rerank(
                query=query,
                candidates=candidates,
                top_k=candidate_k,
                model_name=reranker_model
            )

        # 4. Parent-Child Hierarchical Context Expansion
        if use_parent_child:
            seen_parent_ids = set()
            expanded_candidates = []
            for c in candidates:
                p_id = c.get("parent_id")
                p_text = c.get("parent_text")
                # Expand to parent text if available and longer than child text
                if p_text and len(p_text.strip()) > len(c.get("text", "").strip()):
                    if p_id and p_id in seen_parent_ids:
                        continue
                    if p_id:
                        seen_parent_ids.add(p_id)
                    expanded_item = c.copy()
                    expanded_item["child_text"] = c.get("text", "")
                    expanded_item["text"] = p_text
                    expanded_item["is_parent_expanded"] = True
                    expanded_candidates.append(expanded_item)
                else:
                    expanded_candidates.append(c)
            candidates = expanded_candidates

        # 5. Contextual Compression (extract query-relevant facts)
        if use_compression:
            candidates = ContextualCompressor.compress_chunks(
                query=query,
                chunks=candidates,
                relevance_threshold=compression_threshold
            )

        # 6. Deduplicate redundant passages and return final top_k
        return cls._deduplicate_chunks(candidates, max_items=top_k)
