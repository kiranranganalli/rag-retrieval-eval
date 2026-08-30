"""
Stage 4: hybrid search (dense + lexical, RRF fused) + cross-encoder reranker.
Takes the fused hybrid results, re-scores them with a smarter model, resorts.

Usage:
    python eval_reranked.py
"""

import os
import time
import psycopg
from openai import OpenAI
from sentence_transformers import CrossEncoder

EMBED_MODEL = "text-embedding-3-small"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CANDIDATE_K = 50
TOP_K = 20
STAGE_NAME = "reranked"
RRF_K = 60

client = OpenAI()
reranker = CrossEncoder(RERANK_MODEL)


def embed_query(text: str) -> list[float]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding


def dense_search(conn, query_embedding, k=CANDIDATE_K):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s",
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


def rrf_fuse(rank_lists, k=RRF_K, top_k=CANDIDATE_K):
    scores = {}
    for ranked in rank_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_id for chunk_id, _ in fused[:top_k]]


def get_texts(conn, chunk_ids):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, text FROM chunks WHERE id = ANY(%s)",
            (chunk_ids,),
        )
        return dict(cur.fetchall())


def rerank(question, candidate_ids, texts_by_id, top_k=TOP_K):
    pairs = [(question, texts_by_id[cid]) for cid in candidate_ids]
    scores = reranker.predict(pairs)
    scored = sorted(zip(candidate_ids, scores), key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in scored[:top_k]]


def recall_at(retrieved, correct, k):
    return 1.0 if any(c in set(retrieved[:k]) for c in correct) else 0.0


def mrr(retrieved, correct):
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

    recall_5_scores, recall_20_scores, mrr_scores, latencies = [], [], [], []

    for qid, question, correct_ids in gold:
        start = time.time()
        q_emb = embed_query(question)
        dense_ids = dense_search(conn, q_emb)
        lexical_ids = lexical_search(conn, question)
        candidates = rrf_fuse([dense_ids, lexical_ids])
        texts_by_id = get_texts(conn, candidates)
        retrieved = rerank(question, candidates, texts_by_id)
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

    print("\n--- Reranked results ---")
    print(f"Recall@5:  {r5_avg:.2f}")
    print(f"Recall@20: {r20_avg:.2f}")
    print(f"MRR:       {mrr_avg:.2f}")
    print(f"Avg latency: {lat_avg:.0f}ms")


if __name__ == "__main__":
    main()
