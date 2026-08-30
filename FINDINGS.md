# RAG Retrieval Eval: Results & Findings

A retrieval-augmented search system built to *measure*, not just demo, how much
each retrieval technique improves answer quality — tested on a real 69-chunk
business document (Copperleaf Kitchen & Roastery internal operations profile).

## The approach
Built the evaluation harness before the pipeline. Ten gold questions were
written by hand against the real document, each mapped to the chunk(s) that
answer them. Every pipeline stage is scored against the same gold set on:

- **Recall@5** — did the correct chunk appear in the top 5 results?
- **Recall@20** — same, but top 20 (looser test)
- **MRR** — how close to rank #1 did the correct chunk land?
- **Latency** — how long each query took

## Results across all 4 retrieval stages

| Stage | Recall@5 | Recall@20 | MRR | Latency |
|---|---|---|---|---|
| 1. Baseline (dense vector search only) | 0.90 | 1.00 | 0.56 | 266ms |
| 2. Hybrid (+ BM25 keyword search, RRF fusion) | 0.90 | 1.00 | 0.65 | 379ms |
| 3. Contextual (+ LLM-written context notes per chunk) | 0.90 | 1.00 | 0.64 | 263ms |
| 4. Reranked (+ cross-encoder reranker) | **1.00** | 1.00 | **0.72** | 1095ms |

Reranking was the clear winner — it's the only stage that hit perfect recall,
and it delivered the best ranking quality overall, at roughly 3-4x the latency
cost of the earlier stages.

## What each stage actually did, and what I learned

**Stage 2 (Hybrid search)** improved MRR (0.56 → 0.65) by blending keyword
matching alongside vector search, but did *not* fix the one exact-match
question in my gold set (delivery days + mileage) that I expected it to fix.
Useful reminder: hybrid search helps ranking broadly, but isn't a guaranteed
fix for every keyword-heavy failure.

**Stage 3 (Contextual chunks)** fixed that exact question — but broke a
*different* one that had been perfect in every prior stage. Investigating why:
the chunk in question mixed two unrelated facts (a one-line location policy,
plus a large 7-row pricing table). The LLM-generated context note summarized
the chunk based on its dominant content (the pricing table), which diluted the
one-line fact the question was actually asking about. **Lesson: contextual
summaries work best on chunks that cover one topic — a chunk mixing two
unrelated facts can have its context note skew toward whichever one takes up
more space.**

**Stage 4 (Reranking)** was the biggest, most consistent win — first stage to
hit perfect Recall@5, at the cost of being the slowest by far (roughly 3-4x
the baseline latency, since it does a second, heavier scoring pass over
candidates).

## A finding beyond the metrics: citation-level faithfulness

After reranking, I added a final step: have an LLM answer each question using
only the top 5 retrieved chunks, and cite which chunk it used. All 10 answers
were factually correct. But checking the *citations* against the actual chunk
content revealed something the retrieval metrics didn't catch:

- For 2 of 10 questions, retrieval correctly surfaced the true source chunk
  in the top 5 (confirmed by inspecting `retrieval_runs` directly) — but the
  LLM cited a *different*, incorrect chunk instead, even though its answer
  text was accurate.

This means **100% Recall@5 does not guarantee a trustworthy citation.**
Retrieval did its job; the answer-generation step is a separate, additional
place faithfulness can break. A natural next step — not yet built — would be
a citation-verification pass: checking that a cited chunk actually contains
the claimed fact, using the same LLM-as-judge pattern from my agent
evaluation project ([agent-eval-system](https://github.com/kiranranganalli/agent-eval-system)).

## Stack
Postgres 17 + pgvector (dense search), Postgres full-text search (BM25-style
lexical search), `text-embedding-3-small` (OpenAI) for embeddings,
`gpt-4o-mini` for context generation and final answers, `cross-encoder/ms-marco-MiniLM-L-6-v2`
(self-hosted, free) for reranking. All retrieval runs and metrics are logged
to Postgres for full reproducibility and stage-by-stage comparison.

## What I'd build next
1. Citation-verification pass (LLM-as-judge checking cited chunk vs. claimed fact)
2. Larger gold set (10 questions is a proof of concept, not a robust benchmark)
3. Fix the chunk-33 boundary issue directly (split the mixed pricing/policy chunk)
4. Cost/latency tradeoff analysis — is reranking worth 3-4x latency for every
   query, or should it be applied selectively?
