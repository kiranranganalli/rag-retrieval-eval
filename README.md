# RAG Retrieval Eval

A retrieval-augmented search system, built to measure — not just demo — how well
different retrieval techniques find the right answer in a document set.

Goal: chunk documents into a searchable knowledge base, then run 5 pipeline
stages (dense search → hybrid → contextual chunks → reranker → citations),
scoring each stage on Recall@5, Recall@20, MRR, and latency. The chart of all
5 stages side by side is the actual portfolio piece.

## Status
- [x] Schema built (Postgres 17, database `evals`, schema `rag_eval`)
- [x] Ingestion pipeline working (`ingest.py`)
- [x] Baseline eval pipeline working (`eval_baseline.py`)
- [x] Stage 1 (baseline dense search) validated on a 1-sentence test doc
- [ ] Load a real, multi-page document
- [ ] Write ~10-15 real gold questions
- [ ] Rerun stage 1 on real data for a meaningful baseline number
- [ ] Stage 2: BM25 + RRF fusion
- [ ] Stage 3: contextual chunk headers
- [ ] Stage 4: cross-encoder reranker
- [ ] Stage 5: metadata filtering + citations

## Setup (already done — kept here for reference)

1. Postgres 17 with pgvector installed via Homebrew:
   ```
   brew install postgresql@17 pgvector
   brew services start postgresql@17
   ```

2. Schema created in the `evals` database, under its own `rag_eval` schema
   (kept separate from the agent-eval-system tables):
   ```
   psql -d evals -c "CREATE EXTENSION IF NOT EXISTS vector;"
   psql -d evals -c "CREATE SCHEMA IF NOT EXISTS rag_eval;"
   psql -d evals -f schema.sql
   ```

3. Python deps:
   ```
   pip install -r requirements.txt
   ```

4. Env vars (set these each new terminal session, or add to `.zshrc`):
   ```
   export OPENAI_API_KEY=sk-...
   export DATABASE_URL=postgresql://kiran@localhost:5432/evals
   ```

## How to run it

1. **Add documents.** Drop `.txt` files into `./documents/`.

2. **Ingest:**
   ```
   python ingest.py
   ```

3. **Write gold questions** — for each, find the chunk id(s) that answer it and insert:
   ```sql
   INSERT INTO rag_eval.gold_questions (question, answer_chunk_ids)
   VALUES ('What was mentioned about supply chain risk?', ARRAY[14, 15]);
   ```

4. **Run baseline eval:**
   ```
   python eval_baseline.py
   ```

You'll get Recall@5, Recall@20, MRR, and latency — one row in `retrieval_metrics`
per stage you run.

## What's next (separate days, not today)
- Stage 2: add BM25 arm (tsvector, already in schema) + RRF fusion
- Stage 3: contextual chunk headers (Claude Haiku + prompt caching)
- Stage 4: cross-encoder reranker (bge-reranker-v2-m3)
- Stage 5: metadata filtering + citations in final answer

Each stage writes its own row to `retrieval_metrics` with a different `stage` name,
so at the end you chart all 5 stages side by side — that chart is the portfolio piece.
