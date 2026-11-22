# AI Doctor

A chat interface for asking questions about your own PDFs. Answers cite the
document and page they came from, and come only from the indexed material.

Three containers: a chat frontend, a FastAPI backend, and Postgres holding both
the embeddings and the conversation history.

```bash
docker compose up -d
```

Open **http://localhost:8080**, click the **gear icon** (top right), and add an
API key. No `.env` file is needed: keys, providers and models are all set in the
interface and kept in the database.

## Setup

1. **Settings → API keys.** Add a key for any of:
   - **OpenAI** (`sk-…`)
   - **Claude** from Anthropic (`sk-ant-…`); answers only, since Anthropic has no
     embedding models
   - **NVIDIA NIM**: an `nvapi-…` key for the hosted models at build.nvidia.com,
     or the address of your own self-hosted NIM (usually no key)
   - **Ollama** on your machine (no key, just its address)

   **Test** next to each one confirms the key works and how many models it offers.
2. **Settings → Models.**
   - **Answers:** pick a provider, then a model from its live list.
   - **Search & index:** pick the embedding provider and model (OpenAI, NVIDIA
     NIM or Ollama). The index is built with this model.
3. **Documents → Index new documents.** Anything in `knowledgebase/` is picked
   up automatically. That folder is mounted into the backend, so you can drop
   files in from your machine or upload them through the interface.

Indexing runs in the background with per-document progress. Only new or changed
files are processed, so adding one PDF costs one PDF.

### Switching models

- **Answer model:** switch any time. It applies from the next question, with no
  restart and no reindexing.
- **Search (embedding) model:** switching rebuilds the index. Vectors from
  different models can't be compared, so when you save a new one the backend
  first embeds a test sentence to check that the model works and measure its
  width. It then clears the old passages and re-embeds every document with the
  new model. If the new model can't be reached, nothing changes.
- **Wide embedding models:** pgvector's index covers at most 2000 dimensions.
  Wider models are asked for 1536 when they support it (OpenAI
  `text-embedding-3-*`, NVIDIA's Matryoshka models). Otherwise search compares
  every passage exactly, which is slower on very large libraries but still
  correct.

### Ollama

`host.docker.internal` (the default address) is how a container reaches your
machine; `localhost` inside a container means the container itself.

**If you get "connection refused":** Ollama listens on `127.0.0.1` by default and
rejects traffic arriving from Docker. Set a system environment variable on
Windows, `OLLAMA_HOST=0.0.0.0:11434`, then restart Ollama.

To run Ollama as a container instead (simpler, but CPU-only unless you set up
WSL2 with the NVIDIA container toolkit):

```bash
docker compose --profile local-llm up -d
# then in Settings → API keys → Ollama:  http://ollama:11434
```

### Self-hosted NVIDIA NIM

Under **NVIDIA NIM → Endpoints**, set the answers endpoint (for example
`http://host.docker.internal:8000/v1`), and the embeddings endpoint if your
embedding NIM runs separately. A key is only required for NVIDIA's hosted API.

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

All three can be changed with environment variables (`FRONTEND_PORT`,
`BACKEND_PORT`, `POSTGRES_PORT`) if these clash too. Only the frontend port
really matters; the backend and database are reachable between containers
regardless of what is published to your machine.

## Configuration

Providers, keys and models are set in the interface (**Settings**) and stored in
the `app_settings` table. A value saved there takes precedence over the
environment.

Environment variables are optional. They act as defaults, for deployments that
prefer them (see `.env.example`):

| Setting | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai`, `anthropic`, `nvidia` or `ollama` |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `NVIDIA_API_KEY` | none | or add them in Settings |
| `EMBED_PROVIDER` / `EMBED_MODEL` | `openai` / `text-embedding-3-large` | the model the index is built with |
| `PDF_EXTRACTION` | `fast` | `fast` or `thorough` |
| `PASSAGES` | `8` | passages each specialist reads per question |
| `QUERY_VARIANTS` | `4` | rephrasings searched per specialist |
| `MAX_SPECIALISTS` | `4` | most specialists triage may assign |
| `EVIDENCE_LIMIT` | `16` | cited passages shown verbatim to reviewer and writer |
| `ENABLE_REVIEW` | `true` | consultant's review pass on multi-specialist answers |
| `ENABLE_SUGGESTIONS` | `true` | three follow-up questions after each answer |

**API keys are stored in plain text** in the local Postgres database. The API
has no login, so keep ports 8080 and 9000 off public networks.

**On `PDF_EXTRACTION`:** measured on a 1,059-page textbook, `thorough` takes
about 0.9 s per page — it keeps headings and reads scanned pages — while `fast`
is roughly 200× quicker for about 97% of the text. Use `fast` while you are
still adjusting things.

## Cost

Indexing roughly 9,400 pages (~6M tokens) with `text-embedding-3-large` costs
well under a dollar, one time. Local embeddings through Ollama cost nothing. The recurring cost is answering: each question
sends about 10k tokens of passages to the chat model. Lower `PASSAGES` to spend
less per question.

## Data

Everything is in Postgres, in the `pgdata` volume.

| Table | Holds |
|---|---|
| `documents` | one row per file: page count, hash, index status, progress |
| `chunks` | passage text, page range, and its embedding |
| `chats` / `messages` | conversation history, with citations per answer |
| `app_settings` | keys, providers and models saved from Settings, and which model built the index |

```bash
docker compose exec db psql -U aidoctor -d aidoctor -c '\dt'
docker compose down          # stop, keep data
docker compose down -v       # stop and erase the database
```

## Layout

```
docker-compose.yml
.env                     optional; Settings in the interface replaces it
db/init.sql              schema, applied on first start
backend/
  app/config.py          settings: saved values first, environment second
  app/db.py              connection pool, saved settings, index reconciliation
  app/providers.py       OpenAI, Claude, NVIDIA NIM, Ollama: chat and embeddings
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



