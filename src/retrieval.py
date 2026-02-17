"""
retrieval.py — Hybrid retrieval pipeline for the ONE ZERO RAG Chatbot.

Three-stage retrieval:
    1. VECTOR SEARCH: Embed query → cosine similarity in ChromaDB → top-N candidates
    2. BM25 SEARCH: Keyword match on chunk texts → top-N candidates
    3. FUSION: Reciprocal Rank Fusion merges both result lists
    4. RERANKING: Cross-encoder re-scores top candidates for final top-k

This hybrid approach fixes pure vector search weaknesses:
- BM25 catches exact keyword matches ("Apple Pay", "ONE PLUS", "dividends")
- Cross-encoder reranking catches semantic nuances that bi-encoder similarity misses

Usage:
    results, context = retrieve(query, embedding_model, collection, chunks)
"""

from __future__ import annotations

import numpy as np
import chromadb
from rank_bm25 import BM25Okapi

from src.embeddings import EmbeddingModel
from src.vectorstore import query_vectorstore
from src.reranker import get_reranker
from config import (
    TOP_K,
    RELEVANCE_THRESHOLD,
    RETRIEVAL_CANDIDATES,
    BM25_WEIGHT,
    VECTOR_WEIGHT,
)


# ── BM25 Index ───────────────────────────────────────────────────────────────

class BM25Index:
    """BM25 keyword search index over chunk texts.

    Built once from all chunks, then queried per user question.
    Tokenization is simple whitespace + lowercasing — sufficient
    for English bank policy documents.
    """

    def __init__(self, chunks: list) -> None:
        """Build BM25 index from chunks.

        Parameters
        ----------
        chunks : list[Chunk]
            All chunks (same set used for vector store).
        """
        self.chunks = chunks
        self.texts = [c.text for c in chunks]
        # Tokenize: lowercase, split on whitespace
        tokenized = [text.lower().split() for text in self.texts]
        self.bm25 = BM25Okapi(tokenized)
        print(f"  ✅ BM25 index built: {len(chunks)} documents")

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """Search for relevant chunks using BM25 keyword matching.

        Parameters
        ----------
        query : str
            User question.
        top_k : int
            Number of results to return.

        Returns
        -------
        list[dict]
            Results with text, metadata, bm25_score. Sorted by score descending.
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices by score
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # only include non-zero matches
                chunk = self.chunks[idx]
                results.append({
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "bm25_score": float(scores[idx]),
                    "id": f"chunk_{idx:04d}",
                })

        return results


# ── Reciprocal Rank Fusion ───────────────────────────────────────────────────

def _reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    vector_weight: float = VECTOR_WEIGHT,
    bm25_weight: float = BM25_WEIGHT,
    k: int = 60,
) -> list[dict]:
    """Merge vector and BM25 results using weighted Reciprocal Rank Fusion.

    RRF score for each document = sum of weight / (k + rank) across both lists.
    This is robust to score scale differences between vector and BM25.

    Parameters
    ----------
    vector_results : list[dict]
        Results from vector search (must have "id" key).
    bm25_results : list[dict]
        Results from BM25 search (must have "id" key).
    vector_weight : float
        Weight for vector search contribution.
    bm25_weight : float
        Weight for BM25 search contribution.
    k : int
        RRF constant (default 60, standard value).

    Returns
    -------
    list[dict]
        Merged results sorted by fused score (descending).
        Each result gains a "fusion_score" key.
    """
    # Collect all unique documents by ID
    doc_map: dict[str, dict] = {}

    # Score from vector results
    for rank, r in enumerate(vector_results, 1):
        doc_id = r["id"]
        if doc_id not in doc_map:
            doc_map[doc_id] = r.copy()
            doc_map[doc_id]["fusion_score"] = 0.0
        doc_map[doc_id]["fusion_score"] += vector_weight / (k + rank)

    # Score from BM25 results
    for rank, r in enumerate(bm25_results, 1):
        doc_id = r["id"]
        if doc_id not in doc_map:
            doc_map[doc_id] = r.copy()
            doc_map[doc_id]["fusion_score"] = 0.0
            # BM25 results don't have "distance", set a placeholder
            if "distance" not in doc_map[doc_id]:
                doc_map[doc_id]["distance"] = 1.0
        doc_map[doc_id]["fusion_score"] += bm25_weight / (k + rank)

    # Sort by fusion score (descending)
    fused = sorted(doc_map.values(), key=lambda x: x["fusion_score"], reverse=True)
    return fused


# ── Context formatting ───────────────────────────────────────────────────────

def format_context_for_llm(results: list[dict]) -> str:
    """Format retrieval results into a context string for the LLM prompt.

    Each chunk is numbered and annotated with its source file and section path,
    so the LLM can cite sources in its answer.

    Parameters
    ----------
    results : list[dict]
        Retrieved chunks. Each dict has "text", "metadata", "distance".

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
        distance = result.get("distance", 0.0)

        header = f"[Source {i}: {source} | {section_path} | distance={distance:.4f}]"
        parts.append(f"{header}\n{result['text']}")

    return "\n\n---\n\n".join(parts)


# ── Main retrieval function ──────────────────────────────────────────────────

def retrieve(
    query: str,
    embedding_model: EmbeddingModel,
    collection: chromadb.Collection,
    chunks: list | None = None,
    bm25_index: BM25Index | None = None,
    top_k: int = TOP_K,
    n_candidates: int = RETRIEVAL_CANDIDATES,
    relevance_threshold: float | None = RELEVANCE_THRESHOLD,
    use_hybrid: bool = True,
    use_reranker: bool = True,
) -> tuple[list[dict], str]:
    """Full hybrid retrieval pipeline: vector + BM25 → fusion → rerank → format.

    Parameters
    ----------
    query : str
        User question.
    embedding_model : EmbeddingModel
        Must match the model used to build the collection.
    collection : chromadb.Collection
        ChromaDB collection to search.
    chunks : list[Chunk] | None
        All chunks (needed for BM25). If None and use_hybrid=True,
        falls back to vector-only search.
    bm25_index : BM25Index | None
        Pre-built BM25 index. If None and chunks provided, builds one.
        Pass a pre-built index to avoid rebuilding per query.
    top_k : int
        Number of final results to return.
    n_candidates : int
        Number of candidates to retrieve before reranking.
    relevance_threshold : float | None
        Max cosine distance to accept. None = no filtering.
    use_hybrid : bool
        If True, combine vector + BM25 search. If False, vector only.
    use_reranker : bool
        If True, apply cross-encoder reranking to candidates.

    Returns
    -------
    tuple[list[dict], str]
        - results: list of dicts (text, metadata, distance, id) for evaluation
        - context: formatted string ready for LLM prompt
    """
    # Stage 1: Vector search (retrieve more candidates for reranking)
    candidate_count = n_candidates if use_reranker else top_k
    vector_results = query_vectorstore(
        query=query,
        embedding_model=embedding_model,
        collection=collection,
        top_k=candidate_count,
        relevance_threshold=None,  # no filtering before reranking
    )

    # Stage 2: BM25 search (if hybrid enabled and chunks available)
    if use_hybrid and (bm25_index is not None or chunks is not None):
        if bm25_index is None and chunks is not None:
            bm25_index = BM25Index(chunks)
        bm25_results = bm25_index.search(query, top_k=candidate_count)

        # Stage 3: Reciprocal Rank Fusion
        candidates = _reciprocal_rank_fusion(vector_results, bm25_results)
    else:
        candidates = vector_results

    # Stage 4: Cross-encoder reranking
    if use_reranker:
        reranker = get_reranker()
        results = reranker.rerank(query, candidates, top_k=top_k)
    else:
        results = candidates[:top_k]

    # Apply relevance threshold ONLY when reranker is not used
    # (reranker already filters by quality — double-filtering causes false drops)
    if relevance_threshold is not None and not use_reranker:
        results = [
            r for r in results
            if r.get("distance", 0.0) <= relevance_threshold
        ]

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
        dist = r.get("distance", 0.0)
        rerank = r.get("rerank_score", None)
        fusion = r.get("fusion_score", None)
        preview = r["text"][:150].replace("\n", " ")

        scores_str = f"dist={dist:.4f}"
        if rerank is not None:
            scores_str += f", rerank={rerank:.4f}"
        if fusion is not None:
            scores_str += f", fusion={fusion:.4f}"

        print(f"  [{i}] {scores_str} | {source} | {section}")
        print(f"      {preview}...")
        print()