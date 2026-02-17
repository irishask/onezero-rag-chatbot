"""
chatbot.py — High-level chatbot interface for the ONE ZERO RAG Chatbot.

Provides a single ask() function that orchestrates the full pipeline:
    question → hybrid retrieve → rerank → generate → display answer with sources.

Usage in notebook:
    from src.chatbot import ask
    ask("What is the ATM withdrawal limit?", embedding_model, collection, bm25_index=bm25_index)
"""

from __future__ import annotations

import chromadb

from src.embeddings import EmbeddingModel
from src.retrieval import retrieve, BM25Index
from src.generation import generate_answer
from config import TOP_K, RELEVANCE_THRESHOLD


def ask(
    question: str,
    embedding_model: EmbeddingModel,
    collection: chromadb.Collection,
    chunks: list | None = None,
    bm25_index: BM25Index | None = None,
    top_k: int = TOP_K,
    relevance_threshold: float | None = RELEVANCE_THRESHOLD,
    use_hybrid: bool = True,
    use_reranker: bool = True,
    show_sources: bool = True,
    show_context: bool = False,
) -> str:
    """Ask a question and get an answer from the RAG chatbot.

    Full pipeline: hybrid retrieve → rerank → generate answer.
    Prints the answer and optionally the source chunks used.

    Parameters
    ----------
    question : str
        User question.
    embedding_model : EmbeddingModel
        Embedding model (must match collection).
    collection : chromadb.Collection
        ChromaDB collection to search.
    chunks : list[Chunk] | None
        All chunks (needed for BM25 hybrid search).
    bm25_index : BM25Index | None
        Pre-built BM25 index. Avoids rebuilding per query.
    top_k : int
        Number of chunks to retrieve.
    relevance_threshold : float | None
        Max cosine distance to accept. None = no filtering.
    use_hybrid : bool
        If True, combine vector + BM25 search.
    use_reranker : bool
        If True, apply cross-encoder reranking.
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
        chunks=chunks,
        bm25_index=bm25_index,
        top_k=top_k,
        relevance_threshold=relevance_threshold,
        use_hybrid=use_hybrid,
        use_reranker=use_reranker,
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
            dist = r.get("distance", 0.0)
            rerank = r.get("rerank_score", None)

            score_str = f"dist={dist:.4f}"
            if rerank is not None:
                score_str += f", rerank={rerank:.4f}"

            print(f"  [{i}] {source} | {section} ({score_str})")

    if show_context:
        print(f"\n{'─'*50}")
        print("Full context sent to LLM:")
        print(context)

    return answer