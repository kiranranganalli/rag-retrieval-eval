"""
Stage 5: take the best pipeline (reranked retrieval), then ask an LLM
to answer each gold question using ONLY the retrieved chunks, citing
which chunk id(s) it used. This is the final, user-facing step.

Usage:
    python generate_answers.py
"""

import os
import psycopg
from openai import OpenAI
from sentence_transformers import CrossEncoder

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CANDIDATE_K = 50
TOP_K = 5   # only pass the top 5 reranked chunks to the LLM as context

client = OpenAI()
reranker = CrossEncoder(RERANK_MODEL)


def embed_query(text):
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding


def dense_search(conn, query_embedding, k=CANDIDATE_K):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s",
            (query_embedding, k),
        )
        return [row[0] for row in cur.fetchall()]


def lexical_search(conn, question, k=CANDIDATE_K):
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


def rrf_fuse(rank_lists, k=60, top_k=CANDIDATE_K):
    scores = {}
    for ranked in rank_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_id for chunk_id, _ in fused[:top_k]]


def get_texts(conn, chunk_ids):
    with conn.cursor() as cur:
        cur.execute("SELECT id, text FROM chunks WHERE id = ANY(%s)", (chunk_ids,))
        return dict(cur.fetchall())


def rerank(question, candidate_ids, texts_by_id, top_k=TOP_K):
    pairs = [(question, texts_by_id[cid]) for cid in candidate_ids]
    scores = reranker.predict(pairs)
    scored = sorted(zip(candidate_ids, scores), key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in scored[:top_k]]


def generate_answer(question, top_chunks_with_ids):
    context = "\n\n".join(f"[Chunk {cid}]\n{text}" for cid, text in top_chunks_with_ids)
    prompt = f"""Answer the question using ONLY the chunks below. Cite the chunk
id(s) you used in square brackets, like [Chunk 18]. If the chunks don't
contain the answer, say so plainly instead of guessing.

{context}

Question: {question}
Answer:"""
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
    )
    return resp.choices[0].message.content.strip()


def main():
    conn = psycopg.connect(os.environ["DATABASE_URL"])
    conn.execute("SET search_path TO rag_eval, public;")

    with conn.cursor() as cur:
        cur.execute("SELECT id, question FROM gold_questions ORDER BY id;")
        gold = cur.fetchall()

    for qid, question in gold:
        q_emb = embed_query(question)
        dense_ids = dense_search(conn, q_emb)
        lexical_ids = lexical_search(conn, question)
        candidates = rrf_fuse([dense_ids, lexical_ids])
        texts_by_id = get_texts(conn, candidates)
        top_ids = rerank(question, candidates, texts_by_id)
        top_chunks = [(cid, texts_by_id[cid]) for cid in top_ids]

        answer = generate_answer(question, top_chunks)

        print(f"\nQ{qid}: {question}")
        print(f"A: {answer}")

    conn.close()


if __name__ == "__main__":
    main()
