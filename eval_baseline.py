"""
Stage 1: baseline dense-only retrieval, scored against your gold questions.
Run this AFTER ingest.py and AFTER you've added rows to gold_questions.

Usage:
    python eval_baseline.py
"""

import os
import time
import psycopg
from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"
TOP_K = 20
STAGE_NAME = "baseline_dense"

client = OpenAI()


def embed_query(text: str) -> list[float]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding


def retrieve(conn, query_embedding, k=TOP_K):
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


def recall_at(retrieved: list[int], correct: list[int], k: int) -> float:
    top_k = set(retrieved[:k])
    hit = any(c in top_k for c in correct)
    return 1.0 if hit else 0.0


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
        print("No gold questions yet. Add rows to gold_questions and rerun.")
        return

    recall_5_scores, recall_20_scores, mrr_scores, latencies = [], [], [], []

    for qid, question, correct_ids in gold:
        start = time.time()
        q_emb = embed_query(question)
        retrieved = retrieve(conn, q_emb, k=TOP_K)
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

    print("\n--- Baseline results ---")
    print(f"Recall@5:  {r5_avg:.2f}")
    print(f"Recall@20: {r20_avg:.2f}")
    print(f"MRR:       {mrr_avg:.2f}")
    print(f"Avg latency: {lat_avg:.0f}ms")


if __name__ == "__main__":
    main()
