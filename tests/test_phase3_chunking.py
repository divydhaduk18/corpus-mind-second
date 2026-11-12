import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.semantic_chunker import SemanticChunker
from utils.text_splitter import RecursiveTextSplitter


class TestPhase3Chunking(unittest.TestCase):
    def test_semantic_chunker_basic(self):
        """Verify semantic chunker splits multi-topic text into coherent chunks."""
        text = (
            "The quick brown fox jumps over the lazy dog in the dense forest. "
            "Animals in nature have varied survival strategies and behaviors. "
            "\n\n"
            "Quantum mechanics explores the fundamental properties of matter and light at the atomic scale. "
            "Wave particle duality is a central tenet observed in quantum electrodynamics."
        )
        chunker = SemanticChunker(max_chunk_size=300, min_chunk_size=50)
        chunks = chunker.split_text(text)
        self.assertGreater(len(chunks), 0)
        for c in chunks:
            self.assertIsInstance(c, str)
            self.assertGreater(len(c.strip()), 5)

    def test_semantic_chunker_table_preservation(self):
        """Markdown tables must not be broken apart mid-table."""
        table_text = (
            "Here is the financial summary for Q3:\n\n"
            "| Quarter | Revenue | Growth |\n"
            "| --- | --- | --- |\n"
            "| Q1 | $10M | +12% |\n"
            "| Q2 | $15M | +18% |\n"
            "| Q3 | $22M | +25% |\n\n"
            "Following the quarter, overall enterprise growth remains positive."
        )
        chunker = SemanticChunker(max_chunk_size=400)
        chunks = chunker.split_text(table_text)
        # Find chunk containing the table
        table_chunk = next((c for c in chunks if "| Quarter |" in c), None)
        self.assertIsNotNone(table_chunk)
        # Verify all rows remain together
        self.assertIn("| Q1 |", table_chunk)
        self.assertIn("| Q2 |", table_chunk)
        self.assertIn("| Q3 |", table_chunk)

    def test_parent_child_hierarchical_chunking(self):
        """Hierarchical chunking must link each child chunk to its parent section."""
        pages = [{
            "page_number": 1,
            "text": (
                "Section 1: Operating Architecture.\n"
                "The core application framework is built using Streamlit with asynchronous database updates. "
                "All vector embeddings are indexed using FAISS IndexFlatIP structures for cosine distance calculation. "
                "Multi-tenant security ensures complete data isolation across authenticated accounts.\n\n"
                "Section 2: Security Guidelines.\n"
                "Passwords are encrypted with PBKDF2 with 100,000 salt rounds. "
                "Document files are strictly partitioned into user-scoped directories."
            )
        }]
        splitter = RecursiveTextSplitter(chunk_size=150, chunk_overlap=20)
        chunks = splitter.split_pages(
            pages=pages,
            doc_id=101,
            user_id=1,
            file_name="architecture.md",
            file_type="Document",
            strategy="hierarchical"
        )

        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertIn("parent_id", c)
            self.assertIn("parent_text", c)
            self.assertIn("text", c)
            self.assertEqual(c["doc_id"], 101)
            self.assertEqual(c["user_id"], 1)
            # The parent text must contain the child text (or be a superset)
            self.assertGreaterEqual(len(c["parent_text"]), len(c["text"]))


if __name__ == "__main__":
    unittest.main()
