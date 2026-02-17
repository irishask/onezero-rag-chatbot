"""
config.py — Central configuration for the ONE ZERO RAG Chatbot.

All tunable parameters (models, thresholds, paths, API settings)
live here so every module imports from one place.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if present (for API keys)
load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_PERSIST_DIR = PROJECT_ROOT / "vectorstore_db"

DOCUMENT_PATHS: list[str] = [
    str(DATA_DIR / "cards.md"),
    str(DATA_DIR / "securities.md"),
]

# ── API Keys ─────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
# ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

# ── Chunking ─────────────────────────────────────────────────────────────────
LARGE_SECTION_THRESHOLD: int = 1500      # chars — sections above this get sub-split
RECURSIVE_CHUNK_SIZE: int = 1000         # target size for sub-chunks
RECURSIVE_CHUNK_OVERLAP: int = 200       # overlap between consecutive sub-chunks

# ── Embedding Models ─────────────────────────────────────────────────────────
EMBEDDING_MODELS: dict[str, dict] = {
    "text-embedding-3-small": {
        "provider": "openai",
        "dimensions": 1536,
    },
    "text-embedding-3-large": {
        "provider": "openai",
        "dimensions": 3072,
    },
    "BAAI/bge-m3": {
        "provider": "huggingface",
        "dimensions": 1024,
    },
}

DEFAULT_EMBEDDING_MODEL: str = "text-embedding-3-small"

# ── Vector Store (ChromaDB) ──────────────────────────────────────────────────
CHROMA_COLLECTION_PREFIX: str = "onezero"   # collection name: "{prefix}_{model_slug}"
DISTANCE_METRIC: str = "cosine"

# ── Retrieval ────────────────────────────────────────────────────────────────
TOP_K: int = 5
RELEVANCE_THRESHOLD: float = 0.35          # max cosine distance; lower = stricter
# ── Reranking (Cross-Encoder) ────────────────────────────────────────────────
RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RETRIEVAL_CANDIDATES: int = 20        # retrieve more candidates, then rerank to top_k

# ── Hybrid Search (BM25 + Vector) ───────────────────────────────────────────
BM25_WEIGHT: float = 0.3              # weight for BM25 score in fusion (0.0 = vector only)
VECTOR_WEIGHT: float = 0.7            # weight for vector score in fusion

# ── Generation (Anthropic Claude) ────────────────────────────────────────────
# LLM_MODEL: str = "claude-sonnet-4-5-20250514"
LLM_MODEL: str = "gpt-4o"
LLM_TEMPERATURE: float = 0.0
LLM_MAX_TOKENS: int = 1024

SYSTEM_PROMPT: str = (
    "You are a helpful ONE ZERO Bank assistant. "
    "Answer the user's question based ONLY on the provided context from the bank's policy documents. "
    "If the context does not contain enough information to answer, say so honestly — do not guess. "
    "Always cite the source document and section heading your answer comes from."
)


