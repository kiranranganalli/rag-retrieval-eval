"""
Stage 3: for each chunk, ask an LLM to write a short note explaining
what the chunk is about and where it sits in the document. Prepend
that note to the chunk text, then re-embed.

Usage:
    python generate_context.py
"""

import os
import psycopg
from openai import OpenAI

CHAT_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
DOC_PATH = "documents/copperleaf.txt"

client = OpenAI()


def load_full_document():
    with open(DOC_PATH, "r", encoding="utf-8") as f:
        return f.read()


def generate_context(full_doc: str, chunk_text: str) -> str:
    prompt = f"""<document>
{full_doc[:12000]}
</document>

Here is a chunk from the document above:
<chunk>
{chunk_text}
</chunk>

Write a 1-2 sentence context note that situates this chunk within the
overall document, so it can be understood on its own. Answer with only
the context note, nothing else."""
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=80,
    )
    return resp.choices[0].message.content.strip()


def embed(text: str) -> list[float]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding


def main():
    full_doc = load_full_document()
    conn = psycopg.connect(os.environ["DATABASE_URL"])
    conn.execute("SET search_path TO rag_eval, public;")

    with conn.cursor() as cur:
        cur.execute("SELECT id, text FROM chunks WHERE source = 'copperleaf.txt' ORDER BY id;")
        rows = cur.fetchall()

    for chunk_id, text in rows:
        context = generate_context(full_doc, text)
        contextual_text = f"{context}\n\n{text}"
        emb = embed(contextual_text)

        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE chunks
                SET contextual_text = %s,
                    contextual_embedding = %s,
                    contextual_ts = to_tsvector('english', %s)
                WHERE id = %s
                """,
                (contextual_text, emb, contextual_text, chunk_id),
            )
        conn.commit()
        print(f"Chunk {chunk_id}: {context[:80]}...")

    conn.close()
    print(f"\nDone. Added context to {len(rows)} chunks.")


if __name__ == "__main__":
    main()
