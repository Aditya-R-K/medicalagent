-- AI Doctor schema: documents and their embeddings, plus chat history.
-- Runs once, when the database volume is first created.

CREATE EXTENSION IF NOT EXISTS vector;

-- --------------------------------------------------------------------------
-- Documents
-- --------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS documents (
    id            BIGSERIAL PRIMARY KEY,
    filename      TEXT        NOT NULL UNIQUE,
    content_hash  TEXT        NOT NULL,
    pages         INTEGER     NOT NULL DEFAULT 0,
    size_bytes    BIGINT      NOT NULL DEFAULT 0,
    year          INTEGER,
    -- pending | indexing | indexed | failed | excluded
    -- 'excluded' means "do not index this", set from the interface. The file
    -- stays on disk; nothing here ever deletes a document from the folder.
    status        TEXT        NOT NULL DEFAULT 'pending',
    error         TEXT,
    chunk_count   INTEGER     NOT NULL DEFAULT 0,
    progress      REAL        NOT NULL DEFAULT 0,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    indexed_at    TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS documents_status_idx ON documents (status);

-- --------------------------------------------------------------------------
-- Passages
--
-- `text` is the small passage that gets searched; `parent_text` is the wider
-- section handed to the model, so a figure is never separated from the sentence
-- that qualifies it. Page numbers are exact because passages never span pages.
--
-- The embedding column's size is set by the backend at startup to match the
-- configured model. HNSW indexing is limited to 2000 dimensions, which is why
-- 3072-dimension models are truncated rather than stored whole.
-- --------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS chunks (
    id           BIGSERIAL PRIMARY KEY,
    document_id  BIGINT NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    text         TEXT   NOT NULL,
    parent_text  TEXT   NOT NULL,
    section      TEXT   NOT NULL DEFAULT '',
    page_start   INTEGER NOT NULL,
    page_end     INTEGER NOT NULL,
    embedding    vector(1536),
    search       tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);

CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks (document_id);
CREATE INDEX IF NOT EXISTS chunks_search_idx   ON chunks USING GIN (search);
CREATE INDEX IF NOT EXISTS chunks_vector_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- --------------------------------------------------------------------------
-- Conversations
-- --------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS chats (
    id         BIGSERIAL PRIMARY KEY,
    title      TEXT        NOT NULL DEFAULT 'New chat',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS messages (
    id         BIGSERIAL PRIMARY KEY,
    chat_id    BIGINT      NOT NULL REFERENCES chats (id) ON DELETE CASCADE,
    role       TEXT        NOT NULL CHECK (role IN ('user', 'assistant')),
    content    TEXT        NOT NULL,
    -- [{index, source, page_start, page_end, section, score}]
    citations  JSONB       NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS messages_chat_idx ON messages (chat_id, created_at);
CREATE INDEX IF NOT EXISTS chats_updated_idx ON chats (updated_at DESC);

-- --------------------------------------------------------------------------
-- Settings saved from the interface (API keys, provider, models). They take
-- precedence over .env. The backend also creates this table at startup, for
-- databases that existed before it was added.
-- --------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS app_settings (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- --------------------------------------------------------------------------
-- Token accounting, conversation memory, exact reference locations.
-- The backend also adds these at startup, for databases created earlier.
-- --------------------------------------------------------------------------

ALTER TABLE messages  ADD COLUMN IF NOT EXISTS usage JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE chats     ADD COLUMN IF NOT EXISTS summary TEXT NOT NULL DEFAULT '';
ALTER TABLE chats     ADD COLUMN IF NOT EXISTS summarized_upto BIGINT NOT NULL DEFAULT 0;
ALTER TABLE chunks    ADD COLUMN IF NOT EXISTS line_start INTEGER;
ALTER TABLE chunks    ADD COLUMN IF NOT EXISTS line_end INTEGER;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS title TEXT NOT NULL DEFAULT '';
ALTER TABLE documents ADD COLUMN IF NOT EXISTS author TEXT NOT NULL DEFAULT '';

CREATE TABLE IF NOT EXISTS usage_events (
    id                 BIGSERIAL PRIMARY KEY,
    chat_id            BIGINT REFERENCES chats (id) ON DELETE SET NULL,
    message_id         BIGINT REFERENCES messages (id) ON DELETE SET NULL,
    document_id        BIGINT REFERENCES documents (id) ON DELETE SET NULL,
    step               TEXT    NOT NULL,
    provider           TEXT    NOT NULL DEFAULT '',
    model              TEXT    NOT NULL DEFAULT '',
    input_tokens       INTEGER NOT NULL DEFAULT 0,
    output_tokens      INTEGER NOT NULL DEFAULT 0,
    cached_tokens      INTEGER NOT NULL DEFAULT 0,
    cache_write_tokens INTEGER NOT NULL DEFAULT 0,
    calls              INTEGER NOT NULL DEFAULT 1,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS usage_events_created_idx ON usage_events (created_at);
CREATE INDEX IF NOT EXISTS usage_events_chat_idx ON usage_events (chat_id);
