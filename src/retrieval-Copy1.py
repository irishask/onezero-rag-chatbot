"""
retrieval.py — Retrieval pipeline for the ONE ZERO RAG Chatbot.

Orchestrates the full retrieval flow:
    user query → embed → vector search → filter → format context for LLM

Separates retrieval (finding relevant chunks) from generation (LLM answer).
This separation enables independent evaluation of retrieval quality
(Hit Rate, MRR) without needing to call the LLM.

Usage:
    results, context = retrieve(query, embedding_model, collection)
    # results: list of dicts with text, metadata, distance (for evaluation)
    # context: formatted string ready to inject into LLM prompt
"""

from __future__ import annotations

import chromadb

from src.embeddings import EmbeddingModel
from src.vectorstore import query_vectorstore
from config import TOP_K, RELEVANCE_THRESHOLD


# ── Context formatting ───────────────────────────────────────────────────────

def format_context_for_llm(results: list[dict]) -> str:
    """Format retrieval results into a context string for the LLM prompt.

    Each chunk is numbered and annotated with its source file and section path,
    so the LLM can cite sources in its answer.

    Parameters
    ----------
    results : list[dict]
        Output from query_vectorstore or retrieve(). Each dict has
        "text", "metadata", "distance".

    Returns
    -------
    str
        Formatted context string. Empty string if no results.
    """
    if not results:
        return ""

    parts: list[str] = []
    for i, result in enumerate(results, 1):
        source = result["metadata"].get("source", "unknown")
        section_path = result["metadata"].get("section_path", "unknown")
        distance = result["distance"]

        header = f"[Source {i}: {source} | {section_path} | distance={distance:.4f}]"
        parts.append(f"{header}\n{result['text']}")

    return "\n\n---\n\n".join(parts)


# ── Main retrieval function ──────────────────────────────────────────────────

def retrieve(
    query: str,
    embedding_model: EmbeddingModel,
    collection: chromadb.Collection,
    top_k: int = TOP_K,
    relevance_threshold: float | None = RELEVANCE_THRESHOLD,
) -> tuple[list[dict], str]:
    """Full retrieval pipeline: query → embed → search → format.

    Parameters
    ----------
    query : str
        User question.
    embedding_model : EmbeddingModel
        Must match the model used to build the collection.
    collection : chromadb.Collection
        ChromaDB collection to search.
    top_k : int
        Number of results to retrieve.
    relevance_threshold : float | None
        Max cosine distance to accept. None = no filtering.

    Returns
    -------
    tuple[list[dict], str]
        - results: list of dicts (text, metadata, distance, id) for evaluation
        - context: formatted string ready for LLM prompt
    """
    results = query_vectorstore(
        query=query,
        embedding_model=embedding_model,
        collection=collection,
        top_k=top_k,
        relevance_threshold=relevance_threshold,
    )

    context = format_context_for_llm(results)

    return results, context


# ── Display helper ───────────────────────────────────────────────────────────

def print_retrieval_results(query: str, results: list[dict]) -> None:
    """Pretty-print retrieval results for debugging / notebook display.

    Parameters
    ----------
    query : str
        The original query.
    results : list[dict]
        Output from retrieve().
    """
    print(f"Query: {query!r}")
    print(f"Retrieved: {len(results)} chunks")
    print("-" * 60)

    if not results:
        print("  No relevant chunks found.")
        return

    for i, r in enumerate(results, 1):
        source = r["metadata"].get("source", "?")
        section = r["metadata"].get("section_path", "?")
        dist = r["distance"]
        preview = r["text"][:150].replace("\n", " ")

        print(f"  [{i}] dist={dist:.4f} | {source} | {section}")
        print(f"      {preview}...")
        print()