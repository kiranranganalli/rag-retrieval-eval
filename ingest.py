"""
Stage 1: baseline ingestion.
Reads .txt files from ./documents/, chunks them, embeds with OpenAI,
and loads them into the `chunks` table.

Usage:
    export OPENAI_API_KEY=...
    export DATABASE_URL=postgresql://user:pass@localhost:5432/your_db
    python ingest.py
"""

import os
import glob
import psycopg
from openai import OpenAI

DOCS_DIR = "./documents"
CHUNK_SIZE = 800        # characters, not tokens -- good enough for stage 1
CHUNK_OVERLAP = 150
EMBED_MODEL = "text-embedding-3-small"

client = OpenAI()


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def embed(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def main():
    files = glob.glob(os.path.join(DOCS_DIR, "*.txt"))
    if not files:
        print(f"No .txt files found in {DOCS_DIR}. Add a few documents and rerun.")
        return

    conn = psycopg.connect(os.environ["DATABASE_URL"])
    conn.execute("SET search_path TO rag_eval, public;")

    total_chunks = 0
    for path in files:
        source = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        pieces = chunk_text(text)
        embeddings = embed(pieces)

        with conn.cursor() as cur:
            for idx, (piece, emb) in enumerate(zip(pieces, embeddings)):
                cur.execute(
                    """
                    INSERT INTO chunks (source, chunk_index, text, embedding)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (source, idx, piece, emb),
                )
        conn.commit()
        total_chunks += len(pieces)
        print(f"Loaded {len(pieces)} chunks from {source}")

    conn.close()
    print(f"Done. {total_chunks} chunks loaded across {len(files)} documents.")


if __name__ == "__main__":
    main()
