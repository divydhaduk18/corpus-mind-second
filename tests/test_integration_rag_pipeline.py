import os
import sys
import unittest
import time
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.database import db
from utils.auth import AuthManager
from utils.rag import RAGManager
from utils.text_splitter import RecursiveTextSplitter
from utils.chat import PromptGroundingManager, OllamaChatManager
from utils.conversation_rag import ConversationRAGManager
from utils.hallucination_detector import HallucinationDetector


class TestIntegrationRAGPipeline(unittest.TestCase):
    """End-to-End integration test validating all 10 Advanced RAG enhancements working seamlessly together."""

    @classmethod
    def setUpClass(cls):
        cls.username = f"integ_user_{int(time.time())}"
        cls.user_id = db.create_user(cls.username, AuthManager.hash_password("pw_integ_123"))

        # Ingest document pages using Hierarchical Parent-Child chunking
        splitter = RecursiveTextSplitter(chunk_size=200, chunk_overlap=30)
        pages = [
            {
                "page_number": 1,
                "text": (
                    "Acme Corporation reported consolidated revenue of $48.5 million for Q3 2026. "
                    "This represents an annualized increase of 28% year over year. "
                    "Gross profit margins widened to 64.2% across North America and European divisions."
                )
            },
            {
                "page_number": 2,
                "text": (
                    "Capital expenditures in artificial intelligence research totaled $12.3 million. "
                    "The company deployed dense vector indexing using FAISS IndexFlatIP structures. "
                    "All database transactions utilize SQLite configured with Write-Ahead Logging."
                )
            }
        ]

        chunks = splitter.split_pages(
            pages=pages,
            doc_id=501,
            user_id=cls.user_id,
            file_name="acme_q3_filing.pdf",
            file_type="Document",
            strategy="hierarchical"
        )
        for c in chunks:
            c["category"] = "Financial"

        success = RAGManager.index_user_chunks(cls.user_id, chunks)
        assert success, "Failed to index hierarchical chunks"

        # Create chat session for conversational continuity
        cls.session_id = db.create_chat_session(cls.user_id, "Integration Session")
        db.add_message(cls.session_id, "user", "What was Acme's total revenue in Q3 2026?")
        db.add_message(cls.session_id, "assistant", "Acme reported consolidated revenue of $48.5 million in Q3 2026 [acme_q3_filing.pdf, p. 1].")

    @classmethod
    def tearDownClass(cls):
        db.delete_user(cls.user_id)
        RAGManager.invalidate_cache(cls.user_id)

    def test_end_to_end_conversational_rag_pipeline(self):
        """Execute full 10-step Advanced RAG flow:
        Conversational Context -> Query Rewriting -> Multi-Query -> Hybrid Fusion -> 
        Cross-Encoder Rerank -> Parent Expansion -> Compression -> Grounding -> Hallucination Verification.
        """
        follow_up_query = "How much did it invest in artificial intelligence research?"

        # Mock LLM calls for query rewriting & multi-query to ensure 100% offline determinism
        def mock_llm_responses(model, messages, settings=None):
            prompt_content = messages[-1]["content"] if messages else ""
            if "rephrase the follow-up question" in messages[0]["content"].lower():
                yield "How much did Acme Corporation invest in artificial intelligence research?"
            elif "generate" in messages[0]["content"].lower() and "alternative search queries" in messages[0]["content"].lower():
                yield "1. Acme AI research capital expenditures\n2. Acme Corporation artificial intelligence investment\n3. AI R&D budget in Q3"
            elif "please answer the user query based strictly" in prompt_content.lower():
                yield "Acme Corporation invested $12.3 million in artificial intelligence research [acme_q3_filing.pdf, p. 2]."
            else:
                yield "The retrieved documents do not provide sufficient evidence to answer this question."

        with patch("utils.chat.OllamaChatManager.generate_chat_response", side_effect=mock_llm_responses):
            # Step 1: Contextualize user query using conversational history
            retrieval_query = ConversationRAGManager.contextualize_user_query(
                current_query=follow_up_query,
                user_id=self.user_id,
                session_id=self.session_id
            )
            self.assertIn("Acme", retrieval_query)
            self.assertIn("artificial intelligence", retrieval_query)

            # Step 2: Retrieve context using full advanced pipeline
            retrieved_chunks = RAGManager.retrieve_context(
                query=retrieval_query,
                user_id=self.user_id,
                top_k=3,
                search_mode="hybrid",
                fusion_method="weighted",
                dense_weight=0.6,
                sparse_weight=0.4,
                enable_reranking=True,
                enable_multi_query=True,
                enable_parent_child=True,
                enable_contextual_compression=True,
                metadata_filters={"category": "Financial"}
            )

            self.assertGreater(len(retrieved_chunks), 0)
            top_chunk = retrieved_chunks[0]
            self.assertEqual(top_chunk["file_name"], "acme_q3_filing.pdf")
            self.assertIn("12.3 million", top_chunk["text"])

            # Step 3: Format grounded RAG prompt with exact citations
            sys_inst, user_prompt, citations = PromptGroundingManager.format_grounded_rag_prompt(
                query=follow_up_query,
                rag_chunks=retrieved_chunks,
                personality_prompt="You are a helpful assistant."
            )
            self.assertIn("[acme_q3_filing.pdf, p. 2]", citations)
            self.assertIn("12.3 million", user_prompt)

            # Step 4: Generate simulated grounded answer
            response_gen = OllamaChatManager.generate_chat_response(
                model="llama3.2",
                messages=[
                    {"role": "system", "content": sys_inst},
                    {"role": "user", "content": user_prompt}
                ]
            )
            generated_answer = "".join(list(response_gen)).strip()

            # Step 5: Verify answer faithfulness and detect hallucinations
            verification = HallucinationDetector.verify_answer(
                query=follow_up_query,
                answer=generated_answer,
                retrieved_chunks=retrieved_chunks
            )

            self.assertEqual(verification["status"], "VERIFIED")
            self.assertGreaterEqual(verification["faithfulness_score"], 0.80)
            self.assertEqual(len(verification["unsupported_claims"]), 0)


if __name__ == "__main__":
    unittest.main()
