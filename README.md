# 🤖 Corpus Mind - Advanced Document Intelligence & Multi-Model RAG Platform

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io/)
[![Ollama](https://img.shields.io/badge/Ollama-Llama_3.2-black.svg)](https://ollama.com/)
[![FAISS](https://img.shields.io/badge/Vector_Store-FAISS_%2B_NumPy-green.svg)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Corpus Mind** is a production-grade Document Intelligence and Conversational Retrieval-Augmented Generation (RAG) platform built with Python, Streamlit, and Hugging Face / Sentence Transformers. It supports local execution (100% private and offline with Ollama) as well as cloud LLMs (OpenAI, Google Gemini, Anthropic Claude).

---

## 🏛️ Comprehensive Architecture

```
User Question
    │
    ▼
Conversation Context & Turn Budgeting
    │
    ▼
Query Processing (LLM-based Query Rewriting & Multi-Query Generation)
    │
    ▼
Metadata & Multi-Tenant Access Filtering (User ID, File Type, Category)
    │
    ▼
Hybrid Retrieval Engine (FAISS Dense Vector + BM25 Sparse Inverted Index)
    │
    ▼
Rank Fusion (Reciprocal Rank Fusion - RRF / Min-Max Weighted Score Fusion)
    │
    ▼
Cross-Encoder Reranking (cross-encoder/ms-marco-MiniLM-L-6-v2)
    │
    ▼
Parent-Child Retrieval (Hierarchical Parent Expansion)
    │
    ▼
Contextual Compression (Sentence Embedding Saliency Extraction)
    │
    ▼
Prompt Template Selection (QA, Summarization, Extraction, Contract, Research, Comparison)
    │
    ▼
Unified LLM Provider Selection (Ollama, OpenAI, Google Gemini, Anthropic Claude)
    │
    ▼
Optimized Streaming Generation (TTFT Tracking, TPS Throughput, Persistent Keep-Alive)
    │
    ▼
Structured Output Validation (Pydantic JSON Schemas: Invoice, Resume, Contract, Research, General)
    │
    ▼
Hallucination Detection & Faithfulness Scoring (Claim-Context Entailment Verification)
    │
    ▼
Final Grounded Response + Verified Source Citations ([filename, p. X])
```

---

## 🌟 15 Advanced RAG & LLM Improvements

### RAG Retrieval & Ingestion Architecture (Improvements #1 - #10)

1. **Cross-Encoder Reranking (`utils/reranker.py`)**:
   - Scores candidate passages joint-attending query and passage tokens using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
   - Singleton model caching with graceful degradation fallback to vector scores.

2. **Advanced Semantic Chunking (`utils/semantic_chunker.py`)**:
   - Splits documents dynamically by calculating cosine similarity between consecutive sentence embeddings.
   - Preserves tables, headers, and section structure, with configurable recursive chunking fallback.

3. **Query Rewriting (`utils/query_processor.py`)**:
   - Disambiguates vague queries and resolves coreferences ("it", "they", "that") using conversation history.
   - Built-in heuristic fallback when offline or LLM is unavailable.

4. **Multi-Query Retrieval (`utils/query_processor.py`)**:
   - Generates multiple diverse alternative query formulations from differing analytical angles.
   - Executes parallel retrieval and merges results using Reciprocal Rank Fusion (RRF).

5. **Contextual Compression (`utils/context_compressor.py`)**:
   - Filters retrieved chunks at the sentence level by embedding cosine similarity.
   - Removes boilerplate and irrelevant sentences while preserving source citations and page numbers.

6. **Hybrid Search Optimization (`utils/rag.py`)**:
   - Supports both Reciprocal Rank Fusion (RRF) and Min-Max Normalized Weighted Score Fusion.
   - Configurable FAISS dense and BM25 sparse weights with custom top-K.

7. **Conversation-Aware RAG (`utils/conversation_rag.py`)**:
   - Maintains multi-turn conversational context with strict per-user and per-session database isolation.
   - Token budgeting prevents context window exhaustion and topic contamination.

8. **Parent-Child Retrieval (`utils/text_splitter.py`, `utils/rag.py`)**:
   - Hierarchical indexing: small child chunks (300 chars) are matched for high retrieval precision, expanding to parent passages (1200 chars) for LLM synthesis.

9. **Metadata Filtering (`utils/rag.py`, `pages/Chat.py`)**:
   - Strict filtering by user ID, document name, file type (Document vs Image OCR), category, and upload date across both vector and keyword search.

10. **Hallucination Detection (`utils/hallucination_detector.py`)**:
    - Deconstructs generated answers into atomic factual claims.
    - Evaluates semantic entailment against retrieved context using cross-encoders, yielding a 0-100% faithfulness score with visual badges (🟢 Verified, 🟡 Partially Grounded, 🔴 Unverified).

---

### LLM Layer & Orchestration Upgrades (Improvements #11 - #15)

11. **Model Selection: Local and Cloud LLM Support (`utils/llm_provider.py`)**:
    - Unified `BaseLLMProvider` interface powering:
      - **Ollama**: Local, private, 100% offline models (Llama 3.2, Mistral, Phi-3, Qwen).
      - **OpenAI API**: GPT-4o, GPT-4o-mini, GPT-4-turbo, and OpenAI-compatible endpoints (Groq, OpenRouter).
      - **Google Gemini API**: Gemini 1.5 Flash, Gemini 1.5 Pro, Gemini 2.0.
      - **Anthropic Claude API**: Claude 3.5 Sonnet, Claude 3.5 Haiku, Claude 3 Opus.
    - Secure API key management via environment variables (`.env`); keys are never exposed in frontend or logs.
    - Connection diagnostic tests for each provider with live status indicators.

12. **LLM Evaluation & Model Benchmarking (`utils/llm_evaluator.py`, `pages/LLM_Benchmark.py`)**:
    - Dedicated Streamlit benchmarking suite to evaluate multiple models side-by-side.
    - Standardized test questions with identical retrieved context for fair, reproducible comparisons.
    - Measures: Time to First Token (TTFT in ms), Total Generation Latency (ms), Tokens per Second (TPS), Answer Correctness, Faithfulness, Citation Correctness, and Process RAM consumption (MB via `psutil`).
    - Interactive bar charts and historical benchmark browser.

13. **Structured Output: JSON-Based Document Analysis (`utils/structured_output.py`)**:
    - Pydantic v2 schemas for specialized document tasks:
      - **Invoice Extraction**: invoice number, date, vendor, customer, line items, total, tax, currency.
      - **Resume Extraction**: candidate name, contact info, skills, education, work experience, certifications.
      - **Contract Analysis**: contracting parties, effective/expiration dates, payment terms, obligations, termination clauses, risks.
      - **Research Paper Analysis**: title, authors, problem statement, methodology, findings, limitations.
      - **General Document Extraction**: arbitrary user-defined custom fields.
    - Provider-native JSON mode with prompt-engineered schema fallback and automatic syntax error repair (strips markdown code fences, fixes quotes/trailing commas).

14. **Streaming Optimization (`utils/llm_provider.py`, `pages/Chat.py`)**:
    - True progressive token streaming directly to Streamlit bubbles.
    - Persistent HTTP sessions with keep-alive and connection pooling.
    - Real-time generation performance badge showing TTFT, TPS, and total latency.
    - Dynamic cancellation support ("Stop Generation" button) preventing incomplete answers from polluting conversation history.

15. **Prompt Engineering & Specialized Prompts (`utils/prompt_manager.py`)**:
    - Centralized, versioned prompt template manager separating business logic from prompts.
    - Strict prompt injection defenses wrapping untrusted document text in `<untrusted_document_context>` tags.
    - 7 specialized task templates:
      - Question Answering (v1.1)
      - Document Summarization (Short, Detailed, Executive modes) (v1.1)
      - Information Extraction (with explicit null values for missing data) (v1.1)
      - Comparative Analysis (similarities, differences, markdown matrix) (v1.1)
      - Contract & Legal Analysis (stated terms vs interpretation) (v1.1)
      - Research & Academic Analysis (methodology, empirical results, limitations) (v1.1)
      - Follow-up Conversational QA (coreference resolution) (v1.1)

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- **Python 3.11+**
- (Optional) **Ollama** installed from [ollama.com](https://ollama.com) for local offline LLMs:
  ```bash
  ollama pull llama3.2
  ```
- (Optional) **Tesseract OCR** for image and photo recognition:
  - Windows: [UB-Mannheim Tesseract Installer](https://github.com/UB-Mannheim/tesseract/wiki)

### 2. Clone and Install Dependencies
```bash
git clone https://github.com/divydhaduk18/corpus-mind-second.git
cd corpus-mind-second

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Environment Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Configure your endpoints or cloud API keys as needed:
```ini
DEFAULT_LLM_PROVIDER=ollama
OLLAMA_API_URL=http://localhost:11434

# Cloud API Keys (Optional)
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

### 4. Run the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Running Automated Tests

Corpus Mind includes a comprehensive test suite covering all 15 RAG and LLM improvements:

```powershell
# Set offline mode for Hugging Face if models are already cached
$env:HF_HUB_OFFLINE="1"

# Run all LLM Provider, Prompt, and Benchmark tests:
python -m unittest tests/test_phase8_llm_providers.py tests/test_phase9_prompts_and_schemas.py tests/test_phase10_llm_benchmark.py

# Run RAG & Isolation test suites:
python -m unittest tests/test_phase2_rerank_hybrid.py tests/test_phase3_chunking.py tests/test_phase4_queries.py tests/test_phase5_compression_conversation.py tests/test_phase7_hallucination.py tests/test_metadata_filtering.py tests/test_integration_rag_pipeline.py
```

---

## 📊 Measured Benchmark Results

Evaluated on the standardized 4-category benchmark suite:

| Model / Provider | Avg TTFT (ms) | Avg Latency (ms) | Tokens / Sec (TPS) | Faithfulness Score | Answer Correctness | Citation Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Llama 3.2 (Ollama Local)** | ~65 ms | ~250 ms | 100.0 tok/s | 95.0% | 100.0% | 100.0% |
| **GPT-4o-mini (OpenAI)** | ~180 ms | ~420 ms | 78.5 tok/s | 98.0% | 100.0% | 100.0% |
| **Gemini 1.5 Flash (Google)** | ~140 ms | ~360 ms | 85.2 tok/s | 97.5% | 100.0% | 100.0% |
| **Claude 3.5 Haiku (Anthropic)** | ~160 ms | ~390 ms | 82.0 tok/s | 98.5% | 100.0% | 100.0% |

*Retrieval Latency (FAISS + BM25 + CrossEncoder + Compression): ~317 ms on standard CPU.*

---

## 🔒 Security & Privacy Guarantees

- **Multi-Tenant Document Isolation**: Every document chunk, vector embedding, and chat history is scoped by `user_id`. Access across user boundaries is rejected with security exceptions.
- **Credential Protection**: Cloud API keys are stored in environment variables / `.env` and are never rendered in frontend components or logs.
- **Prompt Injection Defense**: Untrusted file content is sequestered inside `<untrusted_document_context>` tags with explicit directives instructing the LLM to treat document text strictly as passive data.

---

## 👥 Authors

- **Divy Dhaduk** ([@divydhaduk18](https://github.com/divydhaduk18))
