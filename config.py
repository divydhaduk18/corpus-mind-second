import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Auto-configure offline mode for Hugging Face Hub if models are already cached
if "HF_HUB_OFFLINE" not in os.environ:
    hf_cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    if (hf_cache_dir / "models--sentence-transformers--all-MiniLM-L6-v2").exists():
        os.environ["HF_HUB_OFFLINE"] = "1"

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = BASE_DIR / "uploads"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
LOGS_DIR = BASE_DIR / "logs"
CSS_DIR = BASE_DIR / "css"

# Create directories if they do not exist
for directory in [DATA_DIR, UPLOADS_DIR, VECTOR_STORE_DIR, LOGS_DIR, CSS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Database Config
DB_PATH = DATA_DIR / "database.sqlite"

# Ollama API
OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://localhost:11434")
DEFAULT_MODEL = "llama3.2"

# Security Configurations
SESSION_TIMEOUT_MINUTES = int(os.environ.get("SESSION_TIMEOUT_MINUTES", 30))
PASSWORD_SALT_ROUNDS = 100000  # For PBKDF2

# Vector Store Config
# Default selected based on empirical benchmark (8x lower RAM, 7.5x faster encode, 15x faster startup, equal top-1 retrieval)
DEFAULT_EMBEDDING_MODEL = os.environ.get("DEFAULT_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 50))

# Hybrid RAG & Advanced Retrieval Configurations
RAG_SEARCH_MODE = os.environ.get("RAG_SEARCH_MODE", "hybrid")  # Options: "hybrid", "dense", "bm25"
RRF_K = int(os.environ.get("RRF_K", 60))                        # Reciprocal Rank Fusion constant
DEFAULT_TOP_K = int(os.environ.get("DEFAULT_TOP_K", 4))
DEFAULT_SIMILARITY_THRESHOLD = float(os.environ.get("DEFAULT_SIMILARITY_THRESHOLD", 0.15))

# 1. Cross-Encoder Reranking
DEFAULT_RERANKER_MODEL = os.environ.get("DEFAULT_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKING_ENABLED = os.environ.get("RERANKING_ENABLED", "true").lower() in ("true", "1", "yes")

# 2. Semantic Chunking
DEFAULT_CHUNKING_STRATEGY = os.environ.get("DEFAULT_CHUNKING_STRATEGY", "semantic")  # "semantic" or "recursive"
SEMANTIC_SIMILARITY_THRESHOLD = float(os.environ.get("SEMANTIC_SIMILARITY_THRESHOLD", 0.65))
SEMANTIC_MIN_CHUNK_SIZE = int(os.environ.get("SEMANTIC_MIN_CHUNK_SIZE", 150))
SEMANTIC_MAX_CHUNK_SIZE = int(os.environ.get("SEMANTIC_MAX_CHUNK_SIZE", 800))

# 3. Query Rewriting
QUERY_REWRITING_ENABLED = os.environ.get("QUERY_REWRITING_ENABLED", "true").lower() in ("true", "1", "yes")

# 4. Multi-Query Retrieval
MULTI_QUERY_ENABLED = os.environ.get("MULTI_QUERY_ENABLED", "true").lower() in ("true", "1", "yes")
MULTI_QUERY_COUNT = int(os.environ.get("MULTI_QUERY_COUNT", 3))

# 5. Contextual Compression
CONTEXTUAL_COMPRESSION_ENABLED = os.environ.get("CONTEXTUAL_COMPRESSION_ENABLED", "true").lower() in ("true", "1", "yes")
COMPRESSION_SIMILARITY_THRESHOLD = float(os.environ.get("COMPRESSION_SIMILARITY_THRESHOLD", 0.35))

# 6. Hybrid Search Optimization (Weighted & Normalized Score Fusion)
HYBRID_FUSION_METHOD = os.environ.get("HYBRID_FUSION_METHOD", "rrf")  # "rrf" or "weighted"
DENSE_SEARCH_WEIGHT = float(os.environ.get("DENSE_SEARCH_WEIGHT", 0.5))
SPARSE_SEARCH_WEIGHT = float(os.environ.get("SPARSE_SEARCH_WEIGHT", 0.5))

# 7. Conversation-Aware RAG
CONVERSATION_AWARE_ENABLED = os.environ.get("CONVERSATION_AWARE_ENABLED", "true").lower() in ("true", "1", "yes")
MAX_CONVERSATION_HISTORY_TURNS = int(os.environ.get("MAX_CONVERSATION_HISTORY_TURNS", 6))
MAX_CONVERSATION_HISTORY_TOKENS = int(os.environ.get("MAX_CONVERSATION_HISTORY_TOKENS", 1000))

# 8. Parent-Child Hierarchical Retrieval
PARENT_CHILD_ENABLED = os.environ.get("PARENT_CHILD_ENABLED", "true").lower() in ("true", "1", "yes")
PARENT_CHUNK_SIZE = int(os.environ.get("PARENT_CHUNK_SIZE", 1200))
PARENT_CHUNK_OVERLAP = int(os.environ.get("PARENT_CHUNK_OVERLAP", 100))
CHILD_CHUNK_SIZE = int(os.environ.get("CHILD_CHUNK_SIZE", 300))
CHILD_CHUNK_OVERLAP = int(os.environ.get("CHILD_CHUNK_OVERLAP", 40))

# 10. Hallucination Detection & Faithfulness Verification
HALLUCINATION_DETECTION_ENABLED = os.environ.get("HALLUCINATION_DETECTION_ENABLED", "true").lower() in ("true", "1", "yes")
FAITHFULNESS_THRESHOLD = float(os.environ.get("FAITHFULNESS_THRESHOLD", 0.70))

# OCR Configurations
DEFAULT_OCR_ENGINE = os.environ.get("DEFAULT_OCR_ENGINE", "tesseract")  # "tesseract" or "rapidocr"
OCR_LANG = os.environ.get("OCR_LANG", "eng")
OCR_MAX_IMAGE_DIM = int(os.environ.get("OCR_MAX_IMAGE_DIM", 2000))
TESSERACT_CMD_PATH = os.environ.get("TESSERACT_CMD_PATH", r"C:/Program Files/Tesseract-OCR/tesseract.exe")

# Supported Document & Image Formats
SUPPORTED_DOC_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv", ".json"}
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

# User-Scoped Paths to Enforce Multi-Tenant Document Isolation
def get_user_uploads_dir(user_id: int) -> Path:
    """Return user-scoped upload directory and ensure it exists."""
    user_dir = UPLOADS_DIR / f"user_{user_id}"
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir

def get_user_vector_store_dir(user_id: int) -> Path:
    """Return user-scoped vector store directory and ensure it exists."""
    user_dir = VECTOR_STORE_DIR / f"user_{user_id}"
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir

# UI Themes
THEMES = {
    "Dark Mode (Default)": "dark",
    "Light Mode": "light"
}

# Personalities for the Assistant
ASSISTANT_PERSONALITIES = {
    "Professional": (
        "You are a professional, highly intelligent, and objective AI Assistant. "
        "Provide clear, concise, and accurate answers based on the retrieved context."
    ),
    "Creative": (
        "You are a creative, enthusiastic, and brainstorming partner AI. "
        "Explore multiple perspectives, provide engaging examples, and write in an inspiring tone."
    ),
    "Friendly": (
        "You are a warm, helpful, and friendly AI companion. "
        "Speak with empathy, keep explanations easy to understand, and show support."
    ),
    "Coding Assistant": (
        "You are an expert software engineer and programming assistant. "
        "Write clean, well-commented, and efficient code. Explain your design decisions briefly."
    )
}
