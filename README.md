# AI Doctor

A chat interface for asking questions about your own PDFs. Answers cite the
document and page they came from, and come only from the indexed material.

Three containers: a chat frontend, a FastAPI backend, and Postgres holding both
the embeddings and the conversation history.

```bash
cp .env.example .env        # add your OpenAI key
docker compose up -d
```

Open **http://localhost:8080**.

## Setup

Put your OpenAI key in `.env`:

```bash
OPENAI_API_KEY=sk-...
```

Then open the app, go to **Documents**, and press **Index new documents**.
Anything in `knowledgebase/` is picked up automatically — that folder is mounted
into the backend, so you can drop files in from your machine or upload them
through the interface, whichever you prefer.

Indexing runs in the background with per-document progress. Only new or changed
files are processed, so adding one PDF costs one PDF.

### Writing answers with Ollama instead

Ollama runs on your machine, not in a container, so it keeps your GPU. In `.env`:

```bash
LLM_PROVIDER=ollama
OLLAMA_CHAT_MODEL=medgemma:4b
OLLAMA_HOST=http://host.docker.internal:11434
```

This changes who writes the answers, not how documents are searched — that is
always `text-embedding-3-large`, so an OpenAI key is still required.

`host.docker.internal` is how a container reaches your machine — `localhost`
inside a container means the container itself.

**If you get "connection refused":** Ollama listens on `127.0.0.1` by default and
rejects traffic arriving from Docker. Set a system environment variable on
Windows, `OLLAMA_HOST=0.0.0.0:11434`, then restart Ollama.

To run Ollama as a container instead — simpler, but CPU-only unless you set up
WSL2 with the NVIDIA container toolkit:

```bash
docker compose --profile local-llm up -d
# then in .env:  OLLAMA_HOST=http://ollama:11434
```

## How it works

Every question is worked by a panel of roles, the way a senior team would:

```
triage        reads the question, classifies it (lookup / concept / management /
              comparison / vignette) and briefs 1-4 specialists
diagnostician for case vignettes only: extracts the findings, ranks a
              differential, names the leading diagnosis — which then becomes
              what the library is searched for
specialists   investigate in parallel. Each plans its own search (a hypothetical
              textbook paragraph to embed, rephrasings with abbreviations
              expanded, index-style terms), runs hybrid retrieval, and reports
              findings with numbered citations
registrar     drafts one coherent answer from the findings
consultant    reviews the draft against the cited passages themselves, and can
              send the panel back for what is missing
attending     writes the final answer in a teaching voice, quoting doses from
              the passages directly, and closes with cited Pearls
```

Retrieval is hybrid: vector search finds wording the question did not use;
full-text search finds the exact strings vector search blurs, like doses and
codes. Short index terms are searched individually — Postgres ANDs every word
it is given, so whole sentences over-constrain the match. Everything is
combined by rank (reciprocal rank fusion), which avoids reconciling different
score scales.

The answer's citation markers are checked against the sources before the reply
is stored: a marker that points at nothing is removed. After each answer, three
follow-up questions are offered as one-click chips.

Passages are built by packing paragraphs and never span a page boundary, so page
numbers in citations are exact. Each passage is stored with the wider section
around it, and that wider text is what the model reads — a figure is never
separated from the sentence that qualifies it.

## Ports

| Service | URL | Why this port |
|---|---|---|
| Frontend | http://localhost:8080 | 3000 was in use on the build machine |
| Backend API | http://localhost:9000/api | 8000 was in use |
| Postgres | localhost:5433 | 5432 is usually a local Postgres |

All three are set in `.env` — change them if these clash too. Only the frontend
port really matters; the backend and database are reachable between containers
regardless of what is published to your machine.

## Configuration

Everything lives in `.env` and is read at startup.

| Setting | Default | Notes |
|---|---|---|
| `OPENAI_API_KEY` | — | required; search always uses OpenAI |
| `LLM_PROVIDER` | `openai` | `openai` or `ollama` — affects answers only |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | |
| `EMBED_DIM` | `1536` | see below |
| `PDF_EXTRACTION` | `fast` | `fast` or `thorough` |
| `PASSAGES` | `8` | passages each specialist reads per question |
| `QUERY_VARIANTS` | `4` | rephrasings searched per specialist |
| `MAX_SPECIALISTS` | `4` | most specialists triage may assign |
| `EVIDENCE_LIMIT` | `16` | cited passages shown verbatim to reviewer and writer |
| `ENABLE_REVIEW` | `true` | consultant's review pass on multi-specialist answers |
| `ENABLE_SUGGESTIONS` | `true` | three follow-up questions after each answer |

**The embedding model is fixed in code** as `openai/text-embedding-3-large`
(`backend/app/config.py`). Search only works if every passage and every question
go through the same model, and changing it invalidates the entire index, so it
is not exposed as a setting. Setting `EMBED_PROVIDER` or `OPENAI_EMBED_MODEL` in
the environment has no effect.

One consequence: **an OpenAI key is always required**, even with
`LLM_PROVIDER=ollama`. Ollama can write the answers locally, but indexing and
search still call OpenAI.

**On `EMBED_DIM`:** the model produces 3072 dimensions, but pgvector's index
tops out at 2000. Asking for 1536 keeps the index usable, halves storage, and
still scores better than `text-embedding-3-small` at the same width. Changing it
clears the passages and marks every document for reindexing, because vectors of
different widths are not comparable.

**On `PDF_EXTRACTION`:** measured on a 1,059-page textbook, `thorough` takes
about 0.9 s per page — it keeps headings and reads scanned pages — while `fast`
is roughly 200× quicker for about 97% of the text. Use `fast` while you are
still adjusting things.

## Cost

Indexing roughly 9,400 pages (~6M tokens) with `text-embedding-3-large` costs
well under a dollar, one time. The recurring cost is answering: each question
sends about 10k tokens of passages to the chat model. Lower `PASSAGES` to spend
less per question.

## Data

Everything is in Postgres, in the `pgdata` volume.

| Table | Holds |
|---|---|
| `documents` | one row per file: page count, hash, index status, progress |
| `chunks` | passage text, page range, and its embedding |
| `chats` / `messages` | conversation history, with citations per answer |

```bash
docker compose exec db psql -U aidoctor -d aidoctor -c '\dt'
docker compose down          # stop, keep data
docker compose down -v       # stop and erase the database
```

## Layout

```
docker-compose.yml
.env                     all configuration
db/init.sql              schema, applied on first start
backend/
  app/config.py          settings from the environment
  app/db.py              connection pool, embedding-size reconciliation
  app/providers.py       OpenAI and Ollama, chat and embeddings
  app/ingest.py          PDF → passages → Postgres
  app/retrieve.py        vector + full-text search, rank fusion
  app/team.py            the panel: prompts and roles, triage → review → answer
  app/pipeline.py        runs the consultation, streams events
  app/store.py           chats and messages
  app/main.py            HTTP API
frontend/
  index.html, app.js, styles.css    chat interface, no build step
  nginx.conf             serves the page, proxies /api, streaming enabled
knowledgebase/           your PDFs, mounted into the backend
```

## Notes

- The frontend is plain HTML, CSS and JavaScript on purpose: no build step, no
  `node_modules`, and it can be edited directly. The API is ordinary REST plus
  server-sent events, so replacing it with React later changes nothing on the
  backend.
- There is no cross-encoder reranking. It was the single largest accuracy gain
  in an earlier version, but it pulls PyTorch into the image and adds gigabytes.
  Retrieval widths were increased to compensate.
- Retrieval accuracy on your documents is unmeasured until you test it.



