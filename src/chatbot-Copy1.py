"""
chatbot.py — High-level chatbot interface for the ONE ZERO RAG Chatbot.

Provides a single ask() function that orchestrates the full pipeline:
    question → retrieve → generate → display answer with sources.

Usage in notebook:
    from src.chatbot import ask
    ask("What is the ATM withdrawal limit?", embedding_model, collection)
"""

from __future__ import annotations

import chromadb

from src.embeddings import EmbeddingModel
from src.retrieval import retrieve
from src.generation import generate_answer
from config import TOP_K, RELEVANCE_THRESHOLD


def ask(
    question: str,
    embedding_model: EmbeddingModel,
    collection: chromadb.Collection,
    top_k: int = TOP_K,
    relevance_threshold: float | None = RELEVANCE_THRESHOLD,
    show_sources: bool = True,
    show_context: bool = False,
) -> str:
    """Ask a question and get an answer from the RAG chatbot.

    Full pipeline: embed question → retrieve chunks → generate answer.
    Prints the answer and optionally the source chunks used.

    Parameters
    ----------
    question : str
        User question.
    embedding_model : EmbeddingModel
        Embedding model (must match collection).
    collection : chromadb.Collection
        ChromaDB collection to search.
    top_k : int
        Number of chunks to retrieve.
    relevance_threshold : float | None
        Max cosine distance to accept. None = no filtering.
    show_sources : bool
        If True, print the source sections used for the answer.
    show_context : bool
        If True, print the full context sent to the LLM (verbose debug).

    Returns
    -------
    str
        The generated answer.
    """
    # Retrieve
    results, context = retrieve(
        query=question,
        embedding_model=embedding_model,
        collection=collection,
        top_k=top_k,
        relevance_threshold=relevance_threshold,
    )

    # Generate
    answer = generate_answer(query=question, context=context)

    # Display
    print(f"Q: {question}")
    print(f"\nA: {answer}")

    if show_sources and results:
        print(f"\n{'─'*50}")
        print(f"Sources ({len(results)} chunks retrieved):")
        for i, r in enumerate(results, 1):
            source = r["metadata"].get("source", "?")
            section = r["metadata"].get("section_path", "?")
            dist = r["distance"]
            print(f"  [{i}] {source} | {section} (dist={dist:.4f})")

    if show_context:
        print(f"\n{'─'*50}")
        print("Full context sent to LLM:")
        print(context)

    return answer