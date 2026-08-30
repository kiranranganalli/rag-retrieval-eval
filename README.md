# RAG Retrieval Eval

A retrieval system built to measure — not just demo — how well different
retrieval techniques find the right answer in a real document. Each stage
adds one technique and gets scored on Recall@5, Recall@20, MRR, and latency,
so the comparison across stages is the actual point of the project.

**Test corpus:** a 69-chunk internal business operations document (Copperleaf
Kitchen & Roastery), scored against 10 hand-written gold questions covering
financials, hours, menu pricing, wholesale terms, and events.

See [FINDINGS.md](FINDINGS.md) for the full writeup, including a
citation-faithfulness bug found after the retrieval numbers looked perfect.

## Status — all 5 stages complete

- [x] Schema built (Postgres 17, database `evals`, schema `rag_eval`)
- [x] Ingestion pipeline (`ingest.py`)
- [x] Real document loaded (69 chunks), 10 gold questions written
- [x] Stage 1: baseline dense search (`eval_baseline.py`)
- [x] Stage 2: hybrid search — BM25-style + vectors, RRF fused (`eval_hybrid.py`)
- [x] Stage 3: contextual chunks — LLM-written context note per chunk, re-embedded
      (`generate_context.py`, `eval_contextual.py`)
- [x] Stage 4: cross-encoder reranker (`eval_reranked.py`)
- [x] Stage 5: cited answers — LLM answers using only retrieved chunks, citing
      chunk id(s) (`generate_answers.py`)

## Results across all 4 retrieval stages

<img width="1350" height="825" alt="retrieval_comparison_chart" src="https://github.com/user-attachments/assets/e5ca39eb-a4ef-4ef6-9ec8-3f267cd250d6" />

## Real Q&A

Real output from `generate_answers.py`, running the full pipeline (hybrid
search → rerank → cited answer) against the Copperleaf document:

> **Q: What is the 2026 revenue target?**
> A: The 2026 revenue target is $4.85 million [Chunk 5].

> **Q: What is in the Smoked Trout Toast?**
> A: The Smoked Trout Toast includes house-smoked trout, dill creme fraiche, pickled shallot, and rye [Chunk 18].

> **Q: How often is free grinder calibration offered?**
> A: Free grinder calibration is offered once per quarter [Chunk 32].

> **Q: Which location can host private events?**
> A: The location that can host private events is Harrow Street, specifically in The Cellar Room. Riverbend cannot host events [Chunk 42], [Chunk 63].
>
> *(Note: retrieval correctly surfaced the true source, Chunk 33, in its top 5 results — but the LLM cited different chunks instead. See [FINDINGS.md](FINDINGS.md) for the full citation-faithfulness investigation.)*


| Stage | Recall@5 | Recall@20 | MRR | Avg latency |
|---|---|---|---|---|
| 1. Baseline (dense only) | 0.90 | 1.00 | 0.56 | 266ms |
| 2. Hybrid (BM25 + vectors, RRF) | 0.90 | 1.00 | 0.65 | 379ms |
| 3. Contextual + hybrid | 0.90 | 1.00 | 0.64 | 263ms |
| **4. Reranked** | **1.00** | 1.00 | **0.72** | 1095ms |

Reranking is the clear winner — the only stage to hit perfect Recall@5, and
the best MRR overall, at roughly 3-4x the latency of the earlier stages.

## What the numbers actually mean

**Stage 2** improved ranking quality (MRR 0.56 → 0.65) — answers land closer
to the top — at the cost of extra latency from running two searches per
query. It did not fix the one exact-match keyword question I expected it to.

**Stage 3** didn't move the aggregate MRR, but it changed *which* questions
succeeded. It fixed the exact-match question stages 1-2 consistently missed,
but broke a different question that had been perfect until then. Root cause:
the broken question's answer was a single sentence buried inside a chunk that
was 90% a pricing table. The LLM's auto-generated context note summarized the
pricing table (the bulk of the content) and de-emphasized the one-line fact
the question needed. **Lesson: contextual summaries help when a chunk covers
one topic, but can hurt when a chunk mixes two unrelated facts — the summary
tends to represent whichever one takes up more space.**

**Stage 4** was the biggest, most consistent win — first stage to hit perfect
Recall@5, at roughly 3-4x the latency of the baseline, since it runs a
second, heavier scoring pass over candidates.

**Stage 5** revealed something the retrieval metrics alone didn't catch: for
2 of 10 questions, the correct chunk *was* in the top 5 shown to the LLM, but
the LLM cited a different, incorrect chunk while still giving the right
answer. 100% Recall@5 does not guarantee a trustworthy citation — see
[FINDINGS.md](FINDINGS.md) for the full breakdown.

## Setup

1. Postgres 17 with pgvector (Homebrew):
   ```
   brew install postgresql@17 pgvector
   brew services start postgresql@17
   ```

2. Schema, in its own `rag_eval` schema inside the `evals` database:
   ```
   psql -d evals -c "CREATE EXTENSION IF NOT EXISTS vector;"
   psql -d evals -c "CREATE SCHEMA IF NOT EXISTS rag_eval;"
   psql -d evals -f schema.sql
   ```

3. Create a virtual environment and install deps (recommended — keeps this
   project's packages isolated from system-wide tools like dbt):
   ```
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   pip install sentence-transformers
   ```

4. Env vars (each new terminal session, or add to `.zshrc`):
   ```
   export OPENAI_API_KEY=sk-...
   export DATABASE_URL=postgresql://kiran@localhost:5432/evals
   export USE_TF=0
   export USE_TORCH=1
   ```

## How to run it

1. Add `.txt` files to `./documents/`, then ingest:
   ```
   python ingest.py
   ```

2. Write gold questions:
   ```sql
   INSERT INTO rag_eval.gold_questions (question, answer_chunk_ids)
   VALUES ('What was mentioned about supply chain risk?', ARRAY[14, 15]);
   ```

3. Run each stage in order (each writes its own row to `retrieval_metrics`):
   ```
   python eval_baseline.py       # stage 1: dense only
   python eval_hybrid.py         # stage 2: hybrid (BM25 + vectors, RRF)
   python generate_context.py    # builds contextual chunks (run once)
   python eval_contextual.py     # stage 3: contextual + hybrid
   python eval_reranked.py       # stage 4: + cross-encoder reranker
   python generate_answers.py    # stage 5: cited final answers
   ```

## Known limitations / what I'd build next

1. **Citation-verification pass** — check that a cited chunk actually
   contains the claimed fact (LLM-as-judge, same pattern as my
   [agent-eval-system](https://github.com/kiranranganalli/agent-eval-system) project)
2. **Larger gold set** — 10 questions is a proof of concept, not a robust benchmark
3. **Re-chunk the mixed pricing/policy chunk** that caused the Stage 3 regression
4. **Cost/latency tradeoff analysis** — is reranking worth 3-4x latency on
   every query, or should it apply selectively?

## Stack

Postgres 17 + pgvector (dense search), Postgres full-text search (BM25-style
lexical search), `text-embedding-3-small` (OpenAI) for embeddings,
`gpt-4o-mini` for context generation and final answers,
`cross-encoder/ms-marco-MiniLM-L-6-v2` (self-hosted, free) for reranking. All
retrieval runs and metrics are logged to Postgres for full reproducibility
and stage-by-stage comparison.
