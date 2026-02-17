"""
evaluation.py — Evaluation framework for the ONE ZERO RAG Chatbot.

Four components:
1. EVAL DATASET — 20 Q&A pairs covering both documents, edge cases, Hebrew terms.
2. RETRIEVAL METRICS — Hit Rate @k, MRR, Context Precision, avg distance.
3. GENERATION METRICS — LLM-as-judge (GPT-4o) for faithfulness, relevance, correctness.
4. EMBEDDING COMPARISON — run all models, collect timings + metrics into one report.

Usage in notebook:
    from src.evaluation import (
        EVAL_DATASET, evaluate_retrieval, evaluate_generation,
        compare_embeddings,
    )
    comparison_df = compare_embeddings(chunks, model_names)
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import chromadb

from src.embeddings import EmbeddingModel, get_embedding_model
from src.retrieval import retrieve
from src.generation import generate_answer
from src.vectorstore import build_vectorstore
from config import TOP_K, RELEVANCE_THRESHOLD, OPENAI_API_KEY, EMBEDDING_MODELS


# ══════════════════════════════════════════════════════════════════════════════
# 1. EVALUATION DATASET
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvalItem:
    """One evaluation question with expected answer and source metadata."""
    question: str
    expected_answer: str
    source: str               # expected source file (cards.md or securities.md)
    expected_section: str     # expected H3 heading (used for Hit Rate matching)
    category: str             # question category for analysis


EVAL_DATASET: list[EvalItem] = [
    # ── Cards: Cash Withdrawals ──────────────────────────────────────────
    EvalItem(
        question="What is the daily ATM withdrawal limit with a credit card in Israel?",
        expected_answer="Up to 10,000 ILS per day, with a single-stroke limit of 8,000 ILS",
        source="cards.md",
        expected_section="Cash Withdrawals limit using ONE ZERO Credit Card",
        category="card_limits",
    ),
    EvalItem(
        question="How much can I withdraw daily from ATMs in Israel with a debit card?",
        expected_answer="Up to 5,000 ILS per day from ATMs; single transaction purchase limit is 3,000 ILS",
        source="cards.md",
        expected_section="Cash Withdrawals limit using ONE ZERO Debit Card",
        category="card_limits",
    ),
    EvalItem(
        question="Are there fees for withdrawing cash from ATMs in Israel?",
        expected_answer="Cash withdrawal from bank-owned ATMs is free. Private ATM companies may charge their own fees.",
        source="cards.md",
        expected_section="Cash withdrawal fees from ATMs in Israel using ONE ZERO credit card or debit card",
        category="fees",
    ),
    # ── Cards: Traveling Abroad ──────────────────────────────────────────
    EvalItem(
        question="What should I do before traveling abroad with my ONE ZERO card?",
        expected_answer="Update Isracard about destination country, update mobile number for SMS confirmations, remember PIN, consider upgrading to ONE/ONE PLUS plan for fee benefits",
        source="cards.md",
        expected_section="What do you need to know before traveling abroad?",
        category="travel",
    ),
    EvalItem(
        question="How can I get card assistance when I am abroad?",
        expected_answer="Contact Isracard customer service via WhatsApp (+972548947598), phone (+97236364666), their app, or request a private banker",
        source="cards.md",
        expected_section="Card Assistance abroad",
        category="travel",
    ),
    # ── Cards: Foreign Currency Fees ─────────────────────────────────────
    EvalItem(
        question="What are the foreign currency exchange fees by subscription plan?",
        expected_answer="ONE and ONE PLUS plans: no conversion fees on card transactions. ZERO plan: 2% commission on foreign currency card transactions.",
        source="cards.md",
        expected_section="Foreign Currency fees by subscription plans",
        category="fees",
    ),
    # ── Cards: Digital Wallets ───────────────────────────────────────────
    EvalItem(
        question="How do I set up Apple Pay with my ONE ZERO card?",
        expected_answer="Through the Cards page in the app -> Add to Wallet -> select device, or via iPhone Wallet app",
        source="cards.md",
        expected_section="Setting Up Apple Pay on Apple Devices",
        category="digital_wallets",
    ),
    # ── Cards: Billing ───────────────────────────────────────────────────
    EvalItem(
        question="What billing days are available for credit card charges?",
        expected_answer="The 2nd, 10th, 15th, or 20th of the month",
        source="cards.md",
        expected_section="Choosing a Billing Day",
        category="billing",
    ),
    # ── Cards: Line of Credit ────────────────────────────────────────────
    EvalItem(
        question="Can I increase my credit card line of credit?",
        expected_answer="Each request is reviewed individually per the bank's credit policy. The bank is not obligated to increase the line of credit.",
        source="cards.md",
        expected_section="Increasing Line of Credit",
        category="credit",
    ),
    # ── Cards: Card Actions ──────────────────────────────────────────────
    EvalItem(
        question="What is the difference between freezing and cancelling a card?",
        expected_answer="A frozen card can be unfrozen at any time. A cancelled card cannot be reactivated.",
        source="cards.md",
        expected_section="Freezing and unfreezing lost and found cards",
        category="card_actions",
    ),
    # ── Securities: Fees by Plan ─────────────────────────────────────────
    EvalItem(
        question="What are the securities trading fees for the ZERO subscription plan?",
        expected_answer="Foreign securities: 0.3% or minimum 24 USD per transaction, max 1,500 USD",
        source="securities.md",
        expected_section="Securities Fees of ZERO Subscription Plan",
        category="securities_fees",
    ),
    EvalItem(
        question="What are the trading fees for foreign securities on the ONE plan?",
        expected_answer="0.1% of transaction amount or minimum 8 USD, maximum 500 USD per transaction",
        source="securities.md",
        expected_section="Securities Fees of ONE Subscription Plan",
        category="securities_fees",
    ),
    EvalItem(
        question="How many free trades do ONE PLUS subscribers get?",
        expected_answer="Up to 10 buy or sell trades in foreign and Israeli securities combined without additional fees",
        source="securities.md",
        expected_section="Securities Fees of ONE PLUS Subscription Plan",
        category="securities_fees",
    ),
    # ── Securities: Trading Hours ────────────────────────────────────────
    EvalItem(
        question="What are the mutual fund trading hours at TASE?",
        expected_answer="Sunday 09:30-15:00, Monday-Thursday 09:30-15:30",
        source="securities.md",
        expected_section="Mutual funds trading hours at TASE",
        category="trading_hours",
    ),
    # ── Securities: Hebrew term ──────────────────────────────────────────
    EvalItem(
        question='What happens if I send a mutual fund order after the "שעה ייעודה"?',
        expected_answer="The order will be sent for the next trading day of the fund",
        source="securities.md",
        expected_section='Sending a Mutual Fund Order After "שעה ייעודה"',
        category="hebrew_term",
    ),
    # ── Securities: Corporate Actions ────────────────────────────────────
    EvalItem(
        question="How are dividends handled for stocks I hold?",
        expected_answer="Dividend payments are calculated based on holding multiplied by published rate. Can receive in cash or company shares.",
        source="securities.md",
        expected_section="Dividends",
        category="corporate_actions",
    ),
    # ── Securities: Stop-Limit ───────────────────────────────────────────
    EvalItem(
        question="What is a stop-limit instruction and how does it work?",
        expected_answer="A conditional trade combining stop and limit features. Set a stop price (trigger) and limit price (execution boundary). Used to mitigate risk.",
        source="securities.md",
        expected_section="Stop-limit Instruction Description",
        category="order_types",
    ),
    # ── Securities: Settlement ───────────────────────────────────────────
    EvalItem(
        question="How long does securities settlement take in Israel and the US?",
        expected_answer="T+1 cycle — proceeds appear one business day after execution. Israeli mutual funds with foreign exposure may take longer.",
        source="securities.md",
        expected_section="Settlement Process and Support for Foreign and Israeli Securities and Account Credits",
        category="settlement",
    ),
    # ── Securities: Operational Hours ────────────────────────────────────
    EvalItem(
        question="What are the operating hours of the securities team?",
        expected_answer="Sunday 09:30-16:00, Monday-Thursday 09:30-23:00",
        source="securities.md",
        expected_section="Securities Team Operational Hours",
        category="trading_hours",
    ),
    # ── Cross-document / edge case ───────────────────────────────────────
    EvalItem(
        question="What are the benefits of the ONE PLUS subscription plan for both cards and securities?",
        expected_answer="Cards: no foreign currency fees, no ATM fees. Securities: up to 10 free trades per month.",
        source="cards.md",  # primary source, but answer spans both docs
        expected_section="Credit card and Debit Card Benefits for Different Subscription Plans",
        category="cross_document",
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
# 2. RETRIEVAL METRICS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RetrievalResult:
    """Metrics for a single eval question."""
    question: str
    category: str
    expected_section: str
    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool
    reciprocal_rank: float    # 1/rank of first correct hit, 0 if not found
    best_distance: float | None  # distance of best correct hit
    context_precision: float  # fraction of retrieved chunks from the correct source file
    retrieval_time_s: float


def _is_section_match(retrieved_h3: str, expected_section: str) -> bool:
    """Check if a retrieved chunk's H3 matches the expected section.

    Uses substring matching to handle slight variations in heading text.

    Parameters
    ----------
    retrieved_h3 : str
        H3 heading from retrieved chunk metadata.
    expected_section : str
        Expected H3 heading from eval dataset.

    Returns
    -------
    bool
        True if the headings match (case-insensitive substring).
    """
    retrieved_lower = retrieved_h3.lower().strip()
    expected_lower = expected_section.lower().strip()
    # Exact or substring match in either direction
    return expected_lower in retrieved_lower or retrieved_lower in expected_lower


def evaluate_retrieval(
    embedding_model: EmbeddingModel,
    collection: chromadb.Collection,
    eval_dataset: list[EvalItem] = EVAL_DATASET,
    top_k: int = TOP_K,
) -> list[RetrievalResult]:
    """Evaluate retrieval quality for all eval questions.

    Runs each question through the retrieval pipeline and measures
    whether the expected section appears in the top-k results.

    Parameters
    ----------
    embedding_model : EmbeddingModel
        Model to use for query embedding (must match collection).
    collection : chromadb.Collection
        ChromaDB collection to search.
    eval_dataset : list[EvalItem]
        Questions to evaluate.
    top_k : int
        Number of results to retrieve per query.

    Returns
    -------
    list[RetrievalResult]
        One result per eval question with Hit Rate and MRR data.
    """
    results: list[RetrievalResult] = []

    for item in eval_dataset:
        t0 = time.time()
        retrieved, _ = retrieve(
            query=item.question,
            embedding_model=embedding_model,
            collection=collection,
            top_k=top_k,
            relevance_threshold=None,  # no filtering — we want to measure raw retrieval
        )
        elapsed = time.time() - t0

        # Find rank of first correct hit
        hit_at_1 = False
        hit_at_3 = False
        hit_at_5 = False
        reciprocal_rank = 0.0
        best_distance: float | None = None

        for rank, r in enumerate(retrieved, 1):
            h3 = r["metadata"].get("h3", "")
            if _is_section_match(h3, item.expected_section):
                if rank == 1:
                    hit_at_1 = True
                if rank <= 3:
                    hit_at_3 = True
                if rank <= 5:
                    hit_at_5 = True
                if reciprocal_rank == 0.0:  # first correct hit
                    reciprocal_rank = 1.0 / rank
                    best_distance = r["distance"]
                break  # only need first correct hit for MRR

        # Context Precision: what fraction of retrieved chunks come from
        # the correct source file? Measures how much noise is in the results.
        if retrieved:
            relevant_count = sum(
                1 for r in retrieved
                if r["metadata"].get("source", "") == item.source
            )
            context_precision = relevant_count / len(retrieved)
        else:
            context_precision = 0.0

        results.append(RetrievalResult(
            question=item.question,
            category=item.category,
            expected_section=item.expected_section,
            hit_at_1=hit_at_1,
            hit_at_3=hit_at_3,
            hit_at_5=hit_at_5,
            reciprocal_rank=reciprocal_rank,
            best_distance=best_distance,
            context_precision=context_precision,
            retrieval_time_s=elapsed,
        ))

    return results


def compute_retrieval_summary(results: list[RetrievalResult]) -> dict[str, float]:
    """Compute aggregate retrieval metrics from individual results.

    Parameters
    ----------
    results : list[RetrievalResult]
        Output from evaluate_retrieval().

    Returns
    -------
    dict[str, float]
        Aggregate metrics: hit_rate_at_1/3/5, mrr, avg_distance, avg_latency_ms.
    """
    n = len(results)
    if n == 0:
        return {}

    distances = [r.best_distance for r in results if r.best_distance is not None]

    return {
        "hit_rate_at_1": sum(r.hit_at_1 for r in results) / n,
        "hit_rate_at_3": sum(r.hit_at_3 for r in results) / n,
        "hit_rate_at_5": sum(r.hit_at_5 for r in results) / n,
        "mrr": sum(r.reciprocal_rank for r in results) / n,
        "context_precision": sum(r.context_precision for r in results) / n,
        "avg_distance": sum(distances) / len(distances) if distances else None,
        "avg_latency_ms": sum(r.retrieval_time_s for r in results) / n * 1000,
        "n_questions": n,
    }


def print_retrieval_summary(
    model_name: str,
    summary: dict[str, float],
) -> None:
    """Pretty-print retrieval metrics for one model."""
    print(f"\n{'='*50}")
    print(f"RETRIEVAL METRICS: {model_name}")
    print(f"{'='*50}")
    print(f"  Hit Rate @1:       {summary['hit_rate_at_1']:.1%}")
    print(f"  Hit Rate @3:       {summary['hit_rate_at_3']:.1%}")
    print(f"  Hit Rate @5:       {summary['hit_rate_at_5']:.1%}")
    print(f"  MRR:               {summary['mrr']:.3f}")
    print(f"  Context Precision: {summary['context_precision']:.1%}")
    if summary['avg_distance'] is not None:
        print(f"  Avg Distance:      {summary['avg_distance']:.4f}")
    print(f"  Avg Latency:       {summary['avg_latency_ms']:.1f} ms")
    print(f"  Questions:         {summary['n_questions']}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# 3. GENERATION METRICS (LLM-as-judge)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GenerationResult:
    """Evaluation result for a single generated answer."""
    question: str
    generated_answer: str
    expected_answer: str
    faithfulness_score: int    # 1-5: is answer grounded in context only?
    relevance_score: int       # 1-5: does answer address the question?
    correctness_score: int     # 1-5: is answer factually correct vs expected?
    faithfulness_reason: str
    relevance_reason: str
    correctness_reason: str


def _llm_judge(
    question: str,
    context: str,
    generated_answer: str,
    expected_answer: str,
) -> dict:
    """Use GPT-4o as a judge to evaluate answer quality.

    Parameters
    ----------
    question : str
        Original user question.
    context : str
        Retrieved context that was provided to the LLM.
    generated_answer : str
        The answer the chatbot generated.
    expected_answer : str
        The expected/reference answer.

    Returns
    -------
    dict
        Keys: faithfulness_score, relevance_score, faithfulness_reason,
        relevance_reason (all strings/ints).
    """
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)

    judge_prompt = f"""You are evaluating a RAG chatbot's answer about bank policies.

Question: {question}
Expected Answer: {expected_answer}
Generated Answer: {generated_answer}
Context Provided: {context[:2000]}

Evaluate on three criteria (score 1-5 each):

1. FAITHFULNESS: Does the generated answer use ONLY information from the provided context? No hallucinated facts? (5=fully grounded, 1=hallucinated)
2. RELEVANCE: Does the generated answer directly address the question asked? (5=perfectly relevant, 1=completely off-topic)
3. CORRECTNESS: Is the generated answer factually correct compared to the expected answer? Does it contain the key facts? (5=fully correct, 1=completely wrong)

Respond in EXACTLY this format (no other text):
FAITHFULNESS_SCORE: <1-5>
FAITHFULNESS_REASON: <one sentence>
RELEVANCE_SCORE: <1-5>
RELEVANCE_REASON: <one sentence>
CORRECTNESS_SCORE: <1-5>
CORRECTNESS_REASON: <one sentence>"""

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.0,
        max_tokens=300,
        messages=[
            {"role": "system", "content": "You are a strict evaluator. Score accurately."},
            {"role": "user", "content": judge_prompt},
        ],
    )

    text = response.choices[0].message.content.strip()

    # Parse response
    result = {
        "faithfulness_score": 3,
        "relevance_score": 3,
        "correctness_score": 3,
        "faithfulness_reason": "Could not parse",
        "relevance_reason": "Could not parse",
        "correctness_reason": "Could not parse",
    }

    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("FAITHFULNESS_SCORE:"):
            try:
                result["faithfulness_score"] = int(line.split(":")[1].strip())
            except ValueError:
                pass
        elif line.startswith("FAITHFULNESS_REASON:"):
            result["faithfulness_reason"] = line.split(":", 1)[1].strip()
        elif line.startswith("RELEVANCE_SCORE:"):
            try:
                result["relevance_score"] = int(line.split(":")[1].strip())
            except ValueError:
                pass
        elif line.startswith("RELEVANCE_REASON:"):
            result["relevance_reason"] = line.split(":", 1)[1].strip()
        elif line.startswith("CORRECTNESS_SCORE:"):
            try:
                result["correctness_score"] = int(line.split(":")[1].strip())
            except ValueError:
                pass
        elif line.startswith("CORRECTNESS_REASON:"):
            result["correctness_reason"] = line.split(":", 1)[1].strip()

    return result


def evaluate_generation(
    embedding_model: EmbeddingModel,
    collection: chromadb.Collection,
    eval_dataset: list[EvalItem] = EVAL_DATASET,
    top_k: int = TOP_K,
    relevance_threshold: float | None = RELEVANCE_THRESHOLD,
) -> list[GenerationResult]:
    """Evaluate end-to-end generation quality using LLM-as-judge.

    For each eval question: retrieve context → generate answer → judge quality.

    Parameters
    ----------
    embedding_model : EmbeddingModel
        Embedding model (must match collection).
    collection : chromadb.Collection
        ChromaDB collection.
    eval_dataset : list[EvalItem]
        Questions to evaluate.
    top_k : int
        Number of chunks to retrieve.
    relevance_threshold : float | None
        Distance threshold for retrieval filtering.

    Returns
    -------
    list[GenerationResult]
        One result per question with scores and reasons.
    """
    results: list[GenerationResult] = []

    for i, item in enumerate(eval_dataset):
        print(f"  Evaluating [{i+1}/{len(eval_dataset)}]: {item.question[:60]}...")

        # Retrieve
        _, context = retrieve(
            query=item.question,
            embedding_model=embedding_model,
            collection=collection,
            top_k=top_k,
            relevance_threshold=relevance_threshold,
        )

        # Generate
        answer = generate_answer(query=item.question, context=context)

        # Judge
        scores = _llm_judge(
            question=item.question,
            context=context,
            generated_answer=answer,
            expected_answer=item.expected_answer,
        )

        results.append(GenerationResult(
            question=item.question,
            generated_answer=answer,
            expected_answer=item.expected_answer,
            faithfulness_score=scores["faithfulness_score"],
            relevance_score=scores["relevance_score"],
            correctness_score=scores["correctness_score"],
            faithfulness_reason=scores["faithfulness_reason"],
            relevance_reason=scores["relevance_reason"],
            correctness_reason=scores["correctness_reason"],
        ))

    return results


def compute_generation_summary(results: list[GenerationResult]) -> dict[str, float]:
    """Compute aggregate generation metrics.

    Parameters
    ----------
    results : list[GenerationResult]
        Output from evaluate_generation().

    Returns
    -------
    dict[str, float]
        Aggregate metrics: avg_faithfulness, avg_relevance.
    """
    n = len(results)
    if n == 0:
        return {}

    return {
        "avg_faithfulness": sum(r.faithfulness_score for r in results) / n,
        "avg_relevance": sum(r.relevance_score for r in results) / n,
        "avg_correctness": sum(r.correctness_score for r in results) / n,
        "n_questions": n,
    }


def print_generation_summary(summary: dict[str, float]) -> None:
    """Pretty-print generation metrics."""
    print(f"\n{'='*50}")
    print(f"GENERATION METRICS (LLM-as-judge)")
    print(f"{'='*50}")
    print(f"  Avg Faithfulness: {summary['avg_faithfulness']:.2f} / 5.0")
    print(f"  Avg Relevance:    {summary['avg_relevance']:.2f} / 5.0")
    print(f"  Avg Correctness:  {summary['avg_correctness']:.2f} / 5.0")
    print(f"  Questions:        {summary['n_questions']}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# 4. CROSS-MODEL EMBEDDING COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def compare_embeddings(
    chunks: list,
    model_names: list[str] | None = None,
    eval_dataset: list[EvalItem] = EVAL_DATASET,
    top_k: int = TOP_K,
    force_rebuild: bool = False,
) -> dict[str, dict]:
    """Run retrieval evaluation across multiple embedding models and collect
    build timings + retrieval metrics into one comparison report.

    Parameters
    ----------
    chunks : list[Chunk]
        Chunks to index (same for all models).
    model_names : list[str] | None
        Embedding model names to compare. Defaults to all in config.
    eval_dataset : list[EvalItem]
        Questions to evaluate.
    top_k : int
        Number of results to retrieve per query.
    force_rebuild : bool
        If True, rebuild vector stores even if they exist.

    Returns
    -------
    dict[str, dict]
        Keyed by model name. Each value contains:
        - All retrieval summary metrics (hit_rate_at_1/3/5, mrr, context_precision, ...)
        - Build timings (embedding_time_s, indexing_time_s, total_build_time_s)
        - Per-question results (retrieval_results)
    """
    if model_names is None:
        model_names = list(EMBEDDING_MODELS.keys())

    comparison: dict[str, dict] = {}

    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"EVALUATING MODEL: {model_name}")
        print(f"{'='*60}")

        # Build / load vector store
        embedding_model = get_embedding_model(model_name)
        collection, timings = build_vectorstore(
            chunks, embedding_model, force_rebuild=force_rebuild
        )

        # Run retrieval evaluation
        retrieval_results = evaluate_retrieval(
            embedding_model, collection, eval_dataset, top_k
        )
        summary = compute_retrieval_summary(retrieval_results)

        # Merge timings into summary
        summary["embedding_time_s"] = timings.get("embedding_time_s", 0.0)
        summary["indexing_time_s"] = timings.get("indexing_time_s", 0.0)
        summary["total_build_time_s"] = timings.get("total_time_s", 0.0)

        # Store per-question results for detailed analysis
        summary["retrieval_results"] = retrieval_results

        print_retrieval_summary(model_name, summary)
        comparison[model_name] = summary

    return comparison