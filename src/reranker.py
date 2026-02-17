"""
reranker.py — Cross-encoder reranking for the ONE ZERO RAG Chatbot.

After initial retrieval returns top-N candidates (by vector similarity),
a cross-encoder re-scores each (query, chunk) pair jointly. Cross-encoders
are far more accurate than bi-encoders because they attend to both texts
simultaneously, catching semantic nuances that embedding similarity misses.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 — small, fast on CPU, proven
on information retrieval benchmarks.

Usage:
    reranker = get_reranker()
    reranked = reranker.rerank(query, candidates, top_k=5)
"""

from __future__ import annotations

import time
from sentence_transformers import CrossEncoder

from config import RERANKER_MODEL


# ── Reranker class ───────────────────────────────────────────────────────────

class Reranker:
    """Cross-encoder reranker for retrieval results.

    Scores each (query, document) pair jointly and returns
    the top-k results sorted by cross-encoder score (descending).
    """

    def __init__(self, model_name: str = RERANKER_MODEL) -> None:
        """Load the cross-encoder model.

        Parameters
        ----------
        model_name : str
            HuggingFace model name for the cross-encoder.
        """
        self.model_name = model_name
        print(f"  Loading cross-encoder reranker: {model_name}...")
        t0 = time.time()
        self.model = CrossEncoder(model_name)
        elapsed = time.time() - t0
        print(f"  ✅ Reranker loaded in {elapsed:.1f}s")

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """Re-score and re-rank retrieval candidates using the cross-encoder.

        Parameters
        ----------
        query : str
            User question.
        candidates : list[dict]
            Retrieval results from vector search. Each dict must have
            a "text" key with the chunk text.
        top_k : int
            Number of top results to return after reranking.

        Returns
        -------
        list[dict]
            Top-k candidates sorted by cross-encoder score (highest first).
            Each dict gains a "rerank_score" key.
        """
        if not candidates:
            return []

        # Build (query, document) pairs for the cross-encoder
        pairs = [(query, c["text"]) for c in candidates]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Attach scores to candidates
        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)

        # Sort by cross-encoder score (descending = most relevant first)
        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

        return ranked[:top_k]


# ── Factory (singleton) ─────────────────────────────────────────────────────

_reranker: Reranker | None = None


def get_reranker(model_name: str = RERANKER_MODEL) -> Reranker:
    """Get or create the singleton reranker instance.

    Parameters
    ----------
    model_name : str
        HuggingFace cross-encoder model name.

    Returns
    -------
    Reranker
        Initialized reranker ready for rerank().
    """
    global _reranker
    if _reranker is None or _reranker.model_name != model_name:
        _reranker = Reranker(model_name)
    return _reranker