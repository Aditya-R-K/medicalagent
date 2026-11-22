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
