"""
Stage 2: hybrid search (BM25-style keyword + dense vector), merged with RRF.
Run this AFTER eval_baseline.py exists and gold_questions are populated.

Usage:
    python eval_hybrid.py
"""

import os
import time
import psycopg
from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"
TOP_K = 20
CANDIDATE_K = 50   # pull more candidates from each arm before fusing
STAGE_NAME = "hybrid_rrf"
RRF_K = 60         # standard RRF constant

client = OpenAI()


def embed_query(text: str) -> list[float]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding


def dense_search(conn, query_embedding, k=CANDIDATE_K):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding, k),
        )
        return [row[0] for row in cur.fetchall()]


def lexical_search(conn, question: str, k=CANDIDATE_K):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id FROM chunks
            WHERE ts @@ plainto_tsquery('english', %s)
            ORDER BY ts_rank_cd(ts, plainto_tsquery('english', %s)) DESC
            LIMIT %s
            """,
            (question, question, k),
        )
        return [row[0] for row in cur.fetchall()]


def rrf_fuse(rank_lists: list[list[int]], k: int = RRF_K, top_k: int = TOP_K):
    scores = {}
    for ranked in rank_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_id for chunk_id, _ in fused[:top_k]]


def recall_at(retrieved: list[int], correct: list[int], k: int) -> float:
    top_k = set(retrieved[:k])
    return 1.0 if any(c in top_k for c in correct) else 0.0


def mrr(retrieved: list[int], correct: list[int]) -> float:
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in correct:
            return 1.0 / rank
    return 0.0


def main():
    conn = psycopg.connect(os.environ["DATABASE_URL"])
    conn.execute("SET search_path TO rag_eval, public;")

    with conn.cursor() as cur:
        cur.execute("SELECT id, question, answer_chunk_ids FROM gold_questions ORDER BY id;")
        gold = cur.fetchall()

    if not gold:
        print("No gold questions yet.")
        return

    recall_5_scores, recall_20_scores, mrr_scores, latencies = [], [], [], []

    for qid, question, correct_ids in gold:
        start = time.time()
        q_emb = embed_query(question)
        dense_ids = dense_search(conn, q_emb)
        lexical_ids = lexical_search(conn, question)
        retrieved = rrf_fuse([dense_ids, lexical_ids])
        latency_ms = (time.time() - start) * 1000

        r5 = recall_at(retrieved, correct_ids, 5)
        r20 = recall_at(retrieved, correct_ids, 20)
        m = mrr(retrieved, correct_ids)

        recall_5_scores.append(r5)
        recall_20_scores.append(r20)
        mrr_scores.append(m)
        latencies.append(latency_ms)

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO retrieval_runs (stage, question_id, retrieved_chunk_ids, latency_ms)
                VALUES (%s, %s, %s, %s)
                """,
                (STAGE_NAME, qid, retrieved, int(latency_ms)),
            )
        conn.commit()

        print(f"Q{qid}: recall@5={r5} recall@20={r20} mrr={m:.2f} ({latency_ms:.0f}ms)")

    avg = lambda xs: sum(xs) / len(xs)
    r5_avg, r20_avg, mrr_avg, lat_avg = avg(recall_5_scores), avg(recall_20_scores), avg(mrr_scores), avg(latencies)

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO retrieval_metrics (stage, recall_at_5, recall_at_20, mrr, avg_latency_ms)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (STAGE_NAME, r5_avg, r20_avg, mrr_avg, lat_avg),
        )
    conn.commit()
    conn.close()

    print("\n--- Hybrid (BM25 + vectors, RRF fused) results ---")
    print(f"Recall@5:  {r5_avg:.2f}")
    print(f"Recall@20: {r20_avg:.2f}")
    print(f"MRR:       {mrr_avg:.2f}")
    print(f"Avg latency: {lat_avg:.0f}ms")


if __name__ == "__main__":
    main()
