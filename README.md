# ONE ZERO Bank — RAG Chatbot

A Retrieval-Augmented Generation chatbot that answers questions from ONE ZERO Bank's policy documents on cards and securities.

## Quick Start

```bash
# 1. Clone / unzip the project
cd onezero-rag-chatbot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your OpenAI API key
cp .env.example .env
# Edit .env → paste your key: OPENAI_API_KEY=sk-proj-...

# 4. Place data files
# Put cards.md and securities.md in data/

# 5. Run the notebook
jupyter notebook main.ipynb
```

## Architecture

```
User Question
     │
     ▼
┌──────────────────────────────────────────────────────┐
│  1. PARSE         document_loader.py                 │
│     Markdown → 228 H3 sections with H2 parent context│
├──────────────────────────────────────────────────────┤
│  2. CHUNK         chunking.py                        │
│     Section-aware: small sections stay whole,         │
│     large (>1500 chars) sub-split with overlap        │
│     → 266 chunks                                      │
├──────────────────────────────────────────────────────┤
│  3. EMBED         embeddings.py                      │
│     3 models compared:                                │
│     • text-embedding-3-small (OpenAI, 1536d)          │
│     • text-embedding-3-large (OpenAI, 3072d)          │
│     • BAAI/bge-m3 (local HuggingFace, 1024d)         │
├──────────────────────────────────────────────────────┤
│  4. STORE         vectorstore.py                     │
│     ChromaDB local persistent, 1 collection per model │
├──────────────────────────────────────────────────────┤
│  5. RETRIEVE      retrieval.py                       │
│     Hybrid: Vector search + BM25 keyword search       │
│     Reciprocal Rank Fusion → Cross-encoder reranking  │
├──────────────────────────────────────────────────────┤
│  6. GENERATE      generation.py                      │
│     GPT-4o — answer from context only, cite sources   │
├──────────────────────────────────────────────────────┤
│  7. EVALUATE      evaluation.py                      │
│     20 test Q&A pairs × 3 embedding models            │
│     Retrieval: Hit Rate, MRR, Context Precision       │
│     Generation: Faithfulness, Relevance, Correctness  │
└──────────────────────────────────────────────────────┘
```

## Key Design Decisions

### Section-Aware Chunking (not naive splitting)

Bank policy documents are organized by topic. Each H3 section is a natural semantic unit (median ~580 chars). Splitting mid-fee-table or mid-procedure destroys meaning. Every chunk is prefixed with its H2 parent heading for hierarchical context:

```
## Cash Withdrawals
### Cash Withdrawals limit using ONE ZERO Credit Card
The daily limit for cash withdrawal...
```

Only 16 of 228 sections exceeded the 1500-char threshold and required sub-splitting with `RecursiveCharacterTextSplitter`.

### Hybrid Retrieval: Vector + BM25 + Reranking

Pure vector search struggles with exact terms ("Apple Pay", "ONE PLUS", "dividends"). The three-stage retrieval pipeline addresses this:

1. **Vector search** (ChromaDB cosine similarity) — captures semantic meaning
2. **BM25 keyword search** — catches exact term matches
3. **Reciprocal Rank Fusion** — merges both result lists robustly
4. **Cross-encoder reranking** (`ms-marco-MiniLM-L-6-v2`) — re-scores (query, chunk) pairs jointly for final ranking

This improved correctness from 4.20 to 4.45 and relevance from 4.70 to 4.85 on our evaluation set.

### Three Embedding Models Compared

| Model | Hit@1 | MRR | Build Time |
|-------|-------|-----|------------|
| text-embedding-3-small | 90% | 0.950 | ~7s |
| text-embedding-3-large | 85% | 0.925 | ~7s |
| BAAI/bge-m3 | **95%** | **0.975** | ~687s |

All three achieve 100% Hit@3 and Hit@5, validating the chunking strategy. BGE-M3 wins on retrieval quality; text-embedding-3-small offers the best speed/quality tradeoff. text-embedding-3-large does not outperform small on this dataset — with only 266 short chunks, the smaller model generalizes better.

### GPT-4o for Generation

Strongest available model for instruction-following, source citation, and admitting uncertainty. System prompt enforces: answer from context only, cite document and section, say "I don't know" when context is insufficient.

## Evaluation Results

### Retrieval (vector-only, 20 queries × 3 models)

| Metric | small | large | bge-m3 |
|--------|-------|-------|--------|
| Hit Rate @1 | 90% | 85% | 95% |
| Hit Rate @3 | 100% | 100% | 100% |
| Hit Rate @5 | 100% | 100% | 100% |
| MRR | 0.950 | 0.925 | 0.975 |
| Context Precision | 95% | 96% | 95% |
| Avg Latency | 434ms | 574ms | 307ms |

### Generation (LLM-as-judge, GPT-4o, 20 queries)

| Metric | Vector Only | Hybrid + Reranking |
|--------|------------|-------------------|
| Faithfulness | 5.00 / 5.0 | 4.90 / 5.0 |
| Relevance | 4.70 / 5.0 | **4.85 / 5.0** |
| Correctness | 4.20 / 5.0 | **4.45 / 5.0** |

Hybrid retrieval improved both relevance and correctness. Faithfulness remained near-perfect — the chatbot does not hallucinate.

## Project Structure

```
onezero-rag-chatbot/
├── config.py                    # All constants and configuration
├── .env                         # API key (gitignored)
├── .env.example                 # Template
├── .gitignore
├── requirements.txt
├── README.md
├── data/
│   ├── cards.md                 # Bank policy — cards
│   └── securities.md            # Bank policy — securities
├── src/
│   ├── __init__.py
│   ├── document_loader.py       # Markdown parser → sections
│   ├── chunking.py              # Section-aware chunking
│   ├── embeddings.py            # Embedding model factory
│   ├── vectorstore.py           # ChromaDB build/load/query
│   ├── retrieval.py             # Hybrid retrieval pipeline
│   ├── reranker.py              # Cross-encoder reranking
│   ├── generation.py            # GPT-4o answer generation
│   ├── chatbot.py               # High-level ask() interface
│   ├── evaluation.py            # Eval dataset + metrics
│   └── visualization.py         # Chart functions
├── main.ipynb                   # Jupyter notebook — main entry point
└── vectorstore_db/              # ChromaDB storage (gitignored)
```

## Assumptions

- OpenAI API key with access to `text-embedding-3-small`, `text-embedding-3-large`, and `gpt-4o`
- Documents are in English with occasional Hebrew terms
- ChromaDB local storage is sufficient (no Docker required)
- BAAI/bge-m3 runs on CPU (~11 minutes for 266 chunks; cached after first run)
- Cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) runs on CPU (~12s first load, then fast)

## Dependencies

- Python 3.12
- openai, chromadb, FlagEmbedding, sentence-transformers, rank-bm25
- langchain-text-splitters, python-dotenv
- pandas, matplotlib, jupyter