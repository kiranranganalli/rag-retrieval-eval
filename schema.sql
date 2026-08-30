CREATE EXTENSION IF NOT EXISTS vector;

SET search_path TO rag_eval, public;

CREATE TABLE IF NOT EXISTS chunks (
    id            SERIAL PRIMARY KEY,
    source        TEXT NOT NULL,
    section       TEXT,
    chunk_index   INT NOT NULL,
    text          TEXT NOT NULL,
    embedding     vector(1536),
    ts            tsvector,
    created_at    TIMESTAMPTZ DEFAULT now()
);

CREATE OR REPLACE FUNCTION chunks_ts_trigger() RETURNS trigger AS $$
BEGIN
  NEW.ts := to_tsvector('english', NEW.text);
  RETURN NEW;
END
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS chunks_ts_update ON chunks;
CREATE TRIGGER chunks_ts_update BEFORE INSERT OR UPDATE ON chunks
FOR EACH ROW EXECUTE FUNCTION chunks_ts_trigger();

CREATE INDEX IF NOT EXISTS idx_chunks_ts ON chunks USING GIN (ts);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS gold_questions (
    id            SERIAL PRIMARY KEY,
    question      TEXT NOT NULL,
    answer_chunk_ids INT[] NOT NULL,
    notes         TEXT,
    created_at    TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS retrieval_runs (
    id            SERIAL PRIMARY KEY,
    stage         TEXT NOT NULL,
    question_id   INT REFERENCES gold_questions(id),
    retrieved_chunk_ids INT[] NOT NULL,
    latency_ms    INT,
    run_at        TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS retrieval_metrics (
    id            SERIAL PRIMARY KEY,
    stage         TEXT NOT NULL,
    recall_at_5   NUMERIC,
    recall_at_20  NUMERIC,
    mrr           NUMERIC,
    avg_latency_ms NUMERIC,
    notes         TEXT,
    computed_at   TIMESTAMPTZ DEFAULT now()
);
