"""
visualization.py — Chart functions for the ONE ZERO RAG Chatbot evaluation.

All plotting logic lives here so the notebook stays clean (call-only).
Uses matplotlib for static charts suitable for the evaluation report.

Usage in notebook:
    from src.visualization import (
        plot_retrieval_comparison,
        plot_generation_comparison,
        plot_build_timings,
        comparison_to_dataframe,
    )
    df = comparison_to_dataframe(comparison)
    plot_retrieval_comparison(comparison)
"""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt


# ── Data conversion ──────────────────────────────────────────────────────────

def comparison_to_dataframe(comparison: dict[str, dict]) -> pd.DataFrame:
    """Convert compare_embeddings() output to a clean DataFrame.

    Parameters
    ----------
    comparison : dict[str, dict]
        Output from evaluation.compare_embeddings().
        Keyed by model name, values contain metric dicts.

    Returns
    -------
    pd.DataFrame
        One row per model, columns are metrics.
    """
    rows = []
    for model_name, metrics in comparison.items():
        row = {"model": model_name}
        for key, value in metrics.items():
            if key == "retrieval_results":
                continue  # skip per-question details
            row[key] = value
        rows.append(row)

    df = pd.DataFrame(rows).set_index("model")
    return df


def generation_results_to_dataframe(gen_results: list) -> pd.DataFrame:
    """Convert generation evaluation results to a DataFrame.

    Parameters
    ----------
    gen_results : list[GenerationResult]
        Output from evaluation.evaluate_generation().

    Returns
    -------
    pd.DataFrame
        One row per question with all scores.
    """
    rows = []
    for r in gen_results:
        rows.append({
            "question": r.question[:80],
            "faithfulness": r.faithfulness_score,
            "relevance": r.relevance_score,
            "correctness": r.correctness_score,
            "faithfulness_reason": r.faithfulness_reason,
            "relevance_reason": r.relevance_reason,
            "correctness_reason": r.correctness_reason,
        })
    return pd.DataFrame(rows)


# ── Retrieval comparison charts ──────────────────────────────────────────────

def plot_retrieval_comparison(comparison: dict[str, dict]) -> plt.Figure:
    """Bar chart comparing retrieval metrics across embedding models.

    Shows Hit Rate @1/@3/@5, MRR, and Context Precision side by side.

    Parameters
    ----------
    comparison : dict[str, dict]
        Output from evaluation.compare_embeddings().

    Returns
    -------
    plt.Figure
        The matplotlib figure (displayed automatically in notebook).
    """
    models = list(comparison.keys())
    # Shorten model names for display
    short_names = [m.split("/")[-1] for m in models]

    metrics = ["hit_rate_at_1", "hit_rate_at_3", "hit_rate_at_5", "mrr", "context_precision"]
    labels = ["Hit@1", "Hit@3", "Hit@5", "MRR", "Ctx Precision"]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = range(len(metrics))
    width = 0.25
    offsets = [-width, 0, width]
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    for i, (model, short) in enumerate(zip(models, short_names)):
        values = [comparison[model].get(m, 0) for m in metrics]
        bars = ax.bar(
            [xi + offsets[i] for xi in x],
            values,
            width=width,
            label=short,
            color=colors[i % len(colors)],
        )
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.0%}" if val <= 1 else f"{val:.2f}",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_ylabel("Score")
    ax.set_title("Retrieval Quality — Embedding Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.legend(title="Embedding Model")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()
    return fig


def plot_build_timings(comparison: dict[str, dict]) -> plt.Figure:
    """Bar chart comparing build timings (embedding + indexing) across models.

    Parameters
    ----------
    comparison : dict[str, dict]
        Output from evaluation.compare_embeddings().

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    models = list(comparison.keys())
    short_names = [m.split("/")[-1] for m in models]

    embed_times = [comparison[m].get("embedding_time_s", 0) for m in models]
    index_times = [comparison[m].get("indexing_time_s", 0) for m in models]
    latencies = [comparison[m].get("avg_latency_ms", 0) for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: build times (stacked)
    ax = axes[0]
    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    bars_embed = ax.bar(short_names, embed_times, label="Embedding", color=colors)
    bars_index = ax.bar(
        short_names, index_times, bottom=embed_times,
        label="Indexing", color=[c + "80" for c in ["#2196F3", "#FF9800", "#4CAF50"]],
        alpha=0.6,
    )
    for bar, et, it in zip(bars_embed, embed_times, index_times):
        total = et + it
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            total + 0.2,
            f"{total:.1f}s",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Vector Store Build Time")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Right: avg query latency
    ax = axes[1]
    bars = ax.bar(short_names, latencies, color=colors)
    for bar, val in zip(bars, latencies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.5,
            f"{val:.1f}ms",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Average Query Latency")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()
    return fig


# ── Generation quality charts ────────────────────────────────────────────────

def plot_generation_scores(gen_results: list) -> plt.Figure:
    """Bar chart of per-question generation scores (faithfulness, relevance, correctness).

    Parameters
    ----------
    gen_results : list[GenerationResult]
        Output from evaluation.evaluate_generation().

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    df = generation_results_to_dataframe(gen_results)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, metric, color, title in zip(
        axes,
        ["faithfulness", "relevance", "correctness"],
        ["#2196F3", "#FF9800", "#4CAF50"],
        ["Faithfulness", "Relevance", "Correctness"],
    ):
        values = df[metric].values
        ax.barh(range(len(values)), values, color=color, alpha=0.8)
        ax.set_xlim(0, 5.5)
        ax.set_yticks(range(len(values)))
        ax.set_yticklabels([q[:40] + "..." for q in df["question"]], fontsize=7)
        ax.set_xlabel("Score (1-5)")
        ax.set_title(title)
        ax.axvline(x=df[metric].mean(), color="red", linestyle="--", alpha=0.7,
                    label=f"Mean: {df[metric].mean():.2f}")
        ax.legend(fontsize=8)
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.show()
    return fig


def plot_generation_summary(summary: dict[str, float]) -> plt.Figure:
    """Simple bar chart of average generation scores.

    Parameters
    ----------
    summary : dict[str, float]
        Output from evaluation.compute_generation_summary().

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    metrics = ["avg_faithfulness", "avg_relevance", "avg_correctness"]
    labels = ["Faithfulness", "Relevance", "Correctness"]
    values = [summary[m] for m in metrics]
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=colors)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.05,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_ylim(0, 5.5)
    ax.set_ylabel("Score (1-5)")
    ax.set_title("Generation Quality — LLM-as-Judge (GPT-4o)")
    ax.axhline(y=5.0, color="green", linestyle="--", alpha=0.3, label="Perfect score")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()
    return fig