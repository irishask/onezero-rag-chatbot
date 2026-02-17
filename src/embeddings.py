"""
embeddings.py — Embedding model factory for the ONE ZERO RAG Chatbot.

Provides a unified interface for embedding text using different models:
- OpenAI: text-embedding-3-small, text-embedding-3-large (API-based)
- HuggingFace: BAAI/bge-m3 (local, via FlagEmbedding library)

All models expose the same interface via the EmbeddingModel protocol:
    embed_texts(texts: list[str]) -> list[list[float]]

Usage:
    model = get_embedding_model("text-embedding-3-small")
    vectors = model.embed_texts(["How do I withdraw cash?", "What are the fees?"])
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

from config import EMBEDDING_MODELS, OPENAI_API_KEY


# ── Abstract base ────────────────────────────────────────────────────────────

class EmbeddingModel(ABC):
    """Common interface for all embedding models."""

    def __init__(self, model_name: str, dimensions: int) -> None:
        self.model_name = model_name
        self.dimensions = dimensions

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into vectors.

        Parameters
        ----------
        texts : list[str]
            Input texts to embed.

        Returns
        -------
        list[list[float]]
            One vector per input text, each of length self.dimensions.
        """
        ...

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string. Convenience wrapper.

        Parameters
        ----------
        query : str
            Single query text.

        Returns
        -------
        list[float]
            Embedding vector.
        """
        return self.embed_texts([query])[0]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name!r}, dim={self.dimensions})"


# ── OpenAI implementation ────────────────────────────────────────────────────

class OpenAIEmbeddingModel(EmbeddingModel):
    """Embedding model using OpenAI's API (text-embedding-3-small/large).

    Handles batching automatically — OpenAI supports up to 2048 texts per call,
    but we batch at 100 for safety and progress logging.
    """

    BATCH_SIZE: int = 100  # texts per API call

    def __init__(self, model_name: str, dimensions: int) -> None:
        super().__init__(model_name, dimensions)
        from openai import OpenAI

        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set. Check your .env file.")

        self.client = OpenAI(api_key=OPENAI_API_KEY)
        print(f"  ✅ OpenAI embedding model loaded: {model_name} (dim={dimensions})")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts via OpenAI API with batching.

        Parameters
        ----------
        texts : list[str]
            Input texts to embed.

        Returns
        -------
        list[list[float]]
            One vector per input text.
        """
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i : i + self.BATCH_SIZE]
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch,
            )
            # Response objects are ordered by index
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

            if len(texts) > self.BATCH_SIZE:
                print(f"    Embedded batch {i // self.BATCH_SIZE + 1}"
                      f"/{(len(texts) - 1) // self.BATCH_SIZE + 1}")

        return all_embeddings


# ── HuggingFace BGE-M3 implementation ────────────────────────────────────────

class BGEM3EmbeddingModel(EmbeddingModel):
    """Embedding model using BAAI/bge-m3 via FlagEmbedding (local, CPU).

    BGE-M3 supports dense, sparse, and ColBERT embeddings.
    We use dense embeddings only for vector store compatibility.

    First load downloads the model (~2GB). Subsequent loads use cache.
    """

    def __init__(self, model_name: str, dimensions: int) -> None:
        super().__init__(model_name, dimensions)
        from FlagEmbedding import BGEM3FlagModel

        print(f"  Loading BGE-M3 model (first run downloads ~2GB)...")
        t0 = time.time()
        self.model = BGEM3FlagModel(
            model_name,
            use_fp16=False,  # CPU — fp16 not supported
        )
        elapsed = time.time() - t0
        print(f"  ✅ BGE-M3 model loaded: {model_name} (dim={dimensions}) in {elapsed:.1f}s")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using BGE-M3 dense embeddings.

        Parameters
        ----------
        texts : list[str]
            Input texts to embed.

        Returns
        -------
        list[list[float]]
            One vector per input text (dense embeddings only).
        """
        output = self.model.encode(
            texts,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        # output["dense_vecs"] is a numpy array of shape (n_texts, dimensions)
        return output["dense_vecs"].tolist()


# ── Factory ──────────────────────────────────────────────────────────────────

def get_embedding_model(model_name: str) -> EmbeddingModel:
    """Factory: create an embedding model by name.

    Parameters
    ----------
    model_name : str
        Must be a key in config.EMBEDDING_MODELS.
        Supported: "text-embedding-3-small", "text-embedding-3-large", "BAAI/bge-m3"

    Returns
    -------
    EmbeddingModel
        Initialized model ready for embed_texts() / embed_query().

    Raises
    ------
    ValueError
        If model_name is not registered in config.EMBEDDING_MODELS.
    """
    if model_name not in EMBEDDING_MODELS:
        raise ValueError(
            f"Unknown model: {model_name!r}. "
            f"Available: {list(EMBEDDING_MODELS.keys())}"
        )

    spec = EMBEDDING_MODELS[model_name]
    provider = spec["provider"]
    dimensions = spec["dimensions"]

    print(f"Initializing embedding model: {model_name} (provider={provider})")

    if provider == "openai":
        return OpenAIEmbeddingModel(model_name, dimensions)
    elif provider == "huggingface":
        return BGEM3EmbeddingModel(model_name, dimensions)
    else:
        raise ValueError(f"Unknown provider: {provider!r} for model {model_name!r}")