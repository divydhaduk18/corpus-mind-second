import os
import sys
import unittest
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.database import db
from utils.auth import AuthManager
from utils.rag import RAGManager


class TestMetadataFiltering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.username = f"meta_user_{int(time.time())}"
        cls.user_id = db.create_user(cls.username, AuthManager.hash_password("pw_meta_123"))

        # Create multi-category chunks for testing metadata filtering
        cls.sample_chunks = [
            {
                "chunk_id": 0,
                "doc_id": 101,
                "user_id": cls.user_id,
                "file_name": "q3_financials.pdf",
                "page_number": 1,
                "section": "Revenue",
                "file_type": "Document",
                "category": "Financial",
                "text": "Acme Corporation Q3 consolidated revenue was $48.5 million."
            },
            {
                "chunk_id": 1,
                "doc_id": 101,
                "user_id": cls.user_id,
                "file_name": "q3_financials.pdf",
                "page_number": 2,
                "section": "R&D",
                "file_type": "Document",
                "category": "Financial",
                "text": "Acme invested $12.3 million in artificial intelligence research."
            },
            {
                "chunk_id": 2,
                "doc_id": 102,
                "user_id": cls.user_id,
                "file_name": "employee_handbook.txt",
                "page_number": 1,
                "section": "Benefits",
                "file_type": "Document",
                "category": "HR",
                "text": "Employees receive 25 days of paid time off PTO annually."
            },
            {
                "chunk_id": 3,
                "doc_id": 103,
                "user_id": cls.user_id,
                "file_name": "server_architecture.png",
                "page_number": 1,
                "section": "Infra",
                "file_type": "Image OCR",
                "category": "Technical",
                "text": "Server architecture diagram: SQLite with WAL mode and FAISS IndexFlatIP."
            }
        ]

        # Index chunks
        success = RAGManager.index_user_chunks(cls.user_id, cls.sample_chunks)
        assert success, "Failed to index test chunks"

    @classmethod
    def tearDownClass(cls):
        db.delete_user(cls.user_id)
        RAGManager.invalidate_cache(cls.user_id)

    def test_filter_by_file_name(self):
        """Must filter exclusively to specified file name."""
        results = RAGManager.retrieve_context(
            query="revenue and financial figures",
            user_id=self.user_id,
            top_k=5,
            doc_filter=["q3_financials.pdf"]
        )
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertEqual(r["file_name"], "q3_financials.pdf")

    def test_filter_by_category(self):
        """Must filter strictly by category."""
        results = RAGManager.retrieve_context(
            query="policies and information",
            user_id=self.user_id,
            top_k=5,
            metadata_filters={"category": "HR"}
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["category"], "HR")
        self.assertIn("paid time off", results[0]["text"])

    def test_filter_by_file_type(self):
        """Must filter strictly by file_type (e.g. Image OCR)."""
        results = RAGManager.retrieve_context(
            query="server architecture and database",
            user_id=self.user_id,
            top_k=5,
            metadata_filters={"file_type": "Image OCR"}
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["file_type"], "Image OCR")
        self.assertIn("FAISS", results[0]["text"])

    def test_filter_by_page_number(self):
        """Must filter strictly by page number."""
        results = RAGManager.retrieve_context(
            query="Acme research and investments",
            user_id=self.user_id,
            top_k=5,
            metadata_filters={"page_number": 2}
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["page_number"], 2)
        self.assertIn("12.3 million", results[0]["text"])

    def test_mandatory_user_id_access_control(self):
        """Omitting user_id must immediately return empty list and reject query."""
        results = RAGManager.retrieve_context(
            query="Acme revenue",
            user_id=None,
            top_k=5
        )
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
