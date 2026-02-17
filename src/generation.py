"""
generation.py — LLM answer generation for the ONE ZERO RAG Chatbot.

Sends retrieved context + user question to OpenAI GPT-4o and returns
an answer grounded in the provided bank policy documents.

The LLM is instructed to:
- Answer ONLY from the provided context (no hallucination)
- Cite source document and section for every claim
- Admit when the context is insufficient

Usage:
    answer = generate_answer(query, context)
"""

from __future__ import annotations

from openai import OpenAI

from config import OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, SYSTEM_PROMPT


# ── Client (module-level singleton) ──────────────────────────────────────────

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Lazy-initialize the OpenAI client."""
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set. Check your .env file.")
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


# ── Generation ───────────────────────────────────────────────────────────────

def generate_answer(
    query: str,
    context: str,
    model: str = LLM_MODEL,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    """Generate an answer using OpenAI GPT-4o with retrieved context.

    Parameters
    ----------
    query : str
        User question.
    context : str
        Formatted context string from retrieval.format_context_for_llm().
        If empty, the LLM is told no relevant documents were found.
    model : str
        OpenAI model name (default from config: gpt-4o).
    temperature : float
        Sampling temperature (0.0 = deterministic).
    max_tokens : int
        Max tokens in the response.
    system_prompt : str
        System instruction for the LLM.

    Returns
    -------
    str
        The generated answer.
    """
    client = _get_client()

    if context:
        user_message = (
            f"Context from bank policy documents:\n\n"
            f"{context}\n\n"
            f"---\n\n"
            f"Question: {query}\n\n"
            f"Answer the question based ONLY on the context above. "
            f"Cite the source document and section for each claim."
        )
    else:
        user_message = (
            f"Question: {query}\n\n"
            f"No relevant documents were found in the bank's policy database. "
            f"Please let the user know you couldn't find relevant information "
            f"and suggest they contact a bank representative."
        )

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )

    return response.choices[0].message.content