# RAG Retrieval Eval

A retrieval system built to measure — not just demo — how well different
retrieval techniques find the right answer in a real document. Each stage adds
one technique and gets scored on Recall@5, Recall@20, MRR, and latency, so the
comparison across stages is the actual point of the project.

Test corpus: a 69-chunk internal business operations document (Copperleaf
Kitchen & Roastery), scored against 10 hand-written gold questions covering
financials, hours, menu pricing, wholesale terms, and events.

## Status

- [x] Schema built (Postgres 17, database `evals`, schema `rag_eval`)
- [x] Ingestion pipeline (`ingest.py`)
- [x] Real document loaded (69 chunks), 10 gold questions written
- [x] Stage 1: baseline dense search (`eval_baseline.py`)
- [x] Stage 2: hybrid search, BM25-style + vectors, RRF fused (`eval_hybrid.py`)
- [x] Stage 3: contextual chunks — LLM-written context note per chunk, re-embedded
      (`generate_context.py`, `eval_contextual.py`)
- [ ] Stage 4: cross-encoder reranker
- [ ] Stage 5: metadata filtering + citations

## Results so far

| Stage | Recall@5 | Recall@20 | MRR | Avg latency |
|---|---|---|---|---|
| 1. Baseline (dense only) | 0.90 | 1.00 | 0.56 | 266ms |
| 2. Hybrid (BM25 + vectors, RRF) | 0.90 | 1.00 | 0.65 | 379ms |
| 3. Contextual + hybrid | 0.90 | 1.00 | 0.64 | 263ms |

### What the numbers actually mean
- **Stage 2** improved ranking quality (MRR 0.56 to 0.65) — answers land closer
  to the top — at the cost of extra latency from running two searches per query.
- **Stage 3** didn't move the aggregate MRR, but it changed *which* questions
  succeeded. It fixed a question that stages 1-2 consistently missed (an exact
  keyword/number-heavy delivery-terms question), but broke a different question
  that had been perfect until then.
- **Root cause of the Stage 3 regression:** the broken question's answer was a
  single sentence buried inside a chunk that was 90% a pricing table. The LLM's
  auto-generated context note summarized the pricing table (the bulk of the
  content) and effectively de-emphasized the one-line fact the question needed.
  Lesson: contextual summaries help when a chunk covers one topic, but can hurt
  when a chunk accidentally mixes two unrelated facts, since the summary tends
  to represent whichever one takes up more space. Fix would be re-chunking so
  that fact and the pricing table aren't in the same chunk.

## Setup

1. Postgres 17 with pgvector (Homebrew):
   brew install postgresql@17 pgvector
   brew services start postgresql@17

2. Schema, in its own rag_eval schema inside the evals database:
   psql -d evals -c "CREATE EXTENSION IF NOT EXISTS vector;"
   psql -d evals -c "CREATE SCHEMA IF NOT EXISTS rag_eval;"
   psql -d evals -f schema.sql

3. Python deps:
   pip install -r requirements.txt

4. Env vars (each new terminal session, or add to .zshrc):
   export OPENAI_API_KEY=sk-...
   export DATABASE_URL=postgresql://kiran@localhost:5432/evals

## How to run it

1. Add .txt files to ./documents/, then ingest:
   python ingest.py

2. Write gold questions:
   INSERT INTO rag_eval.gold_questions (question, answer_chunk_ids)
   VALUES ('What was mentioned about supply chain risk?', ARRAY[14, 15]);

3. Run each stage in order (each writes its own row to retrieval_metrics):
   python eval_baseline.py       # stage 1: dense only
   python eval_hybrid.py         # stage 2: hybrid (BM25 + vectors, RRF)
   python generate_context.py    # builds contextual chunks (run once)
   python eval_contextual.py     # stage 3: contextual + hybrid

## What's next
- Stage 4: cross-encoder reranker (bge-reranker-v2-m3) over the top candidates
- Stage 5: metadata filtering + citations in the final answer
- Investigate re-chunking so single-fact sentences aren't buried inside large
  tables, which should fix the Stage 3 regression without losing its gains
