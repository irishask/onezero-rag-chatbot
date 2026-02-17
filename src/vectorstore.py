"""
vectorstore.py — ChromaDB vector store for the ONE ZERO RAG Chatbot.

Handles building, persisting, loading, and querying ChromaDB collections.
Each embedding model gets its own collection so we can compare retrieval
quality across models using the same evaluation queries.

Design decisions:
- Pre-computed embeddings: we embed chunks ourselves (not via ChromaDB's
  built-in embedding function) so we can measure embedding latency separately
  and reuse vectors for evaluation.
- Persistent storage: collections are saved to disk (vectorstore_db/) so
  re-indexing is only needed once per model.
- One collection per model: naming convention "{prefix}_{model_slug}".
"""

from __future__ import annotations

import re
import time

import chromadb

from src.chunking import Chunk
from src.embeddings import EmbeddingModel
from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_PREFIX, DISTANCE_METRIC


# ── Helpers ──────────────────────────────────────────────────────────────────

def _slugify_model_name(model_name: str) -> str:
    """Convert model name to a safe collection name.

    Examples:
        "text-embedding-3-small" → "text_embedding_3_small"
        "BAAI/bge-m3"            → "baai_bge_m3"
    """
    slug = model_name.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")
    return slug


def get_collection_name(model_name: str) -> str:
    """Build the ChromaDB collection name for a given embedding model.

    Parameters
    ----------
    model_name : str
        Embedding model name (e.g. "text-embedding-3-small").

    Returns
    -------
    str
        Collection name (e.g. "onezero_text_embedding_3_small").
    """
    return f"{CHROMA_COLLECTION_PREFIX}_{_slugify_model_name(model_name)}"


def _get_chroma_client() -> chromadb.ClientAPI:
    """Create a persistent ChromaDB client."""
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))


# ── Build ────────────────────────────────────────────────────────────────────

def build_vectorstore(
    chunks: list[Chunk],
    embedding_model: EmbeddingModel,
    force_rebuild: bool = False,
) -> tuple[chromadb.Collection, dict[str, float]]:
    """Build a ChromaDB collection from chunks using the given embedding model.

    If the collection already exists with the same number of documents and
    force_rebuild is False, returns the existing collection (skips re-indexing).

    Parameters
    ----------
    chunks : list[Chunk]
        Chunks to index (from chunking.chunk_sections).
    embedding_model : EmbeddingModel
        Model to use for embedding chunk texts.
    force_rebuild : bool
        If True, delete and rebuild the collection even if it exists.

    Returns
    -------
    tuple[chromadb.Collection, dict[str, float]]
        The ChromaDB collection and a timing dict with:
        - "embedding_time_s": time to embed all chunks
        - "indexing_time_s": time to insert into ChromaDB
        - "total_time_s": total build time
    """
    client = _get_chroma_client()
    collection_name = get_collection_name(embedding_model.model_name)
    timings: dict[str, float] = {}

    # Check if collection already exists with correct size
    existing_collections = [c.name for c in client.list_collections()]
    if collection_name in existing_collections and not force_rebuild:
        collection = client.get_collection(name=collection_name)
        if collection.count() == len(chunks):
            print(f"  ✅ Collection '{collection_name}' already exists "
                  f"with {collection.count()} docs — skipping rebuild.")
            timings["embedding_time_s"] = 0.0
            timings["indexing_time_s"] = 0.0
            timings["total_time_s"] = 0.0
            return collection, timings

    # Delete existing collection if rebuilding
    if collection_name in existing_collections:
        client.delete_collection(name=collection_name)
        print(f"  Deleted existing collection '{collection_name}' for rebuild.")

    t_total_start = time.time()

    # Step 1: Embed all chunks
    print(f"  Embedding {len(chunks)} chunks with {embedding_model.model_name}...")
    texts = [chunk.text for chunk in chunks]

    t_embed_start = time.time()
    embeddings = embedding_model.embed_texts(texts)
    t_embed_end = time.time()
    timings["embedding_time_s"] = t_embed_end - t_embed_start
    print(f"  Embedding done in {timings['embedding_time_s']:.2f}s")

    # Step 2: Create collection and insert
    print(f"  Inserting into ChromaDB collection '{collection_name}'...")
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": DISTANCE_METRIC},
    )

    t_index_start = time.time()

    # ChromaDB requires string IDs and string metadata values
    ids = [f"chunk_{i:04d}" for i in range(len(chunks))]
    metadatas = []
    for chunk in chunks:
        meta = {}
        for key, value in chunk.metadata.items():
            meta[key] = str(value)  # ChromaDB requires string values
        metadatas.append(meta)

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    t_index_end = time.time()
    timings["indexing_time_s"] = t_index_end - t_index_start
    timings["total_time_s"] = time.time() - t_total_start

    print(f"  Indexing done in {timings['indexing_time_s']:.2f}s")
    print(f"  ✅ Collection '{collection_name}' built: "
          f"{collection.count()} docs, total {timings['total_time_s']:.2f}s")

    return collection, timings


# ── Load ─────────────────────────────────────────────────────────────────────

def load_vectorstore(model_name: str) -> chromadb.Collection:
    """Load an existing ChromaDB collection from disk.

    Parameters
    ----------
    model_name : str
        Embedding model name used when building the collection.

    Returns
    -------
    chromadb.Collection
        The loaded collection.

    Raises
    ------
    ValueError
        If the collection does not exist.
    """
    client = _get_chroma_client()
    collection_name = get_collection_name(model_name)

    existing = [c.name for c in client.list_collections()]
    if collection_name not in existing:
        raise ValueError(
            f"Collection '{collection_name}' not found. "
            f"Available: {existing}. Run build_vectorstore first."
        )

    collection = client.get_collection(name=collection_name)
    print(f"  ✅ Loaded collection '{collection_name}' with {collection.count()} docs")
    return collection


# ── Query ────────────────────────────────────────────────────────────────────

def query_vectorstore(
    query: str,
    embedding_model: EmbeddingModel,
    collection: chromadb.Collection,
    top_k: int = 5,
    relevance_threshold: float | None = None,
) -> list[dict]:
    """Query the vector store: embed query → cosine search → return results.

    Parameters
    ----------
    query : str
        User question to search for.
    embedding_model : EmbeddingModel
        Must be the same model used to build this collection.
    collection : chromadb.Collection
        ChromaDB collection to search.
    top_k : int
        Number of results to return.
    relevance_threshold : float | None
        Max cosine distance to accept. Results above this threshold are
        filtered out. None = no filtering (return all top_k).

    Returns
    -------
    list[dict]
        Each result dict contains:
        - "text": chunk text
        - "metadata": chunk metadata dict
        - "distance": cosine distance (lower = more similar)
        - "id": ChromaDB document ID
    """
    query_embedding = embedding_model.embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # Unpack ChromaDB nested list format
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]

    output: list[dict] = []
    for doc, meta, dist, doc_id in zip(documents, metadatas, distances, ids):
        # Apply relevance threshold if set
        if relevance_threshold is not None and dist > relevance_threshold:
            continue
        output.append({
            "text": doc,
            "metadata": meta,
            "distance": dist,
            "id": doc_id,
        })

    return output