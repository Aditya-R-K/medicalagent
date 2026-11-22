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

Every question goes through one evidence-first loop. The model reads the
sources themselves at every step, never another step's paraphrase of them:

```
plan     one call: rewrites the question to stand alone (resolving "what about
         in pregnancy?" from the conversation), classifies it, splits it into
         sub-questions, and writes a textbook-style paragraph to search with.
         For a clinical case or an exam MCQ it also reasons out a differential
         first, since a case never names its diagnosis.
search   no model calls: every query is embedded in one request, and vector
         search plus keyword search run for each angle, fused by rank
grade    one call scores all ~30 candidates 0-3, drops index pages and tables
         of contents, and names any sub-question still uncovered. Only then
         does one targeted second search run.
pack     the strongest passages are widened with their neighbours (restoring
         the sentence that qualifies a dose), passages that touch are merged
         so overlapping text is sent once, and the total is capped at
         EVIDENCE_TOKENS
write    one call writes from the sources, citing [n] for every claim
verify   a free automatic check that every figure in a cited sentence appears
         in its source, then one call checking every claim against its source.
         A correction is written only when something is wrong.
```

Answers are shaped by question type: **quick lookup**, **concept** (a chain of
causes), **management** (first-line with exact doses, alternatives,
monitoring), **comparison** (a table), **clinical case** (findings,
differential, leading diagnosis, confirmation, management), and **exam MCQ**
(the answer letter, why it is right, why each other option is wrong, and the
concept being tested). Greetings and thanks get a short reply with no search.

Each answer shows its research trace: the sub-questions and whether your
documents covered each, an evidence-strength rating, what was searched for
again, and what the fact-check corrected.

**Tokens.** A typical question takes 4-6 model calls and about 20k input
tokens. The evidence is sent as one identical block to the writer, the checker
and the corrector, so Claude (with an explicit cache breakpoint) and OpenAI
(through automatic prefix caching) charge full price for it only once. Under
**Settings → Models → Helper model** you can choose a cheaper model for
planning, grading and checking, while the main model writes. Repeated questions
reuse their embeddings, and token use per call is written to the backend log.

Retrieval is hybrid: vector search finds wording the question did not use;
keyword search finds exact strings like doses and codes. Whole questions are
matched on any of their words, ranked by how many appear close together;
short index terms must match every word. Results are combined by reciprocal
rank fusion, so different score scales never have to be reconciled.

Passages never span a page boundary, so the page numbers in citations are
exact. A citation marker that points at nothing is removed before the answer
is stored.

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
| `SHORTLIST` | `30` | candidates retrieved for grading |
| `PASSAGES` | `12` | most passages kept after grading |
| `EVIDENCE_TOKENS` | `7000` | ceiling on the evidence sent to the writer |
| `QUERY_VARIANTS` | `4` | rephrasings searched per question |
| `ENABLE_REVIEW` | `true` | the second search for gaps, and the fact-check |
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
sends up to `EVIDENCE_TOKENS` (7k) of evidence, cached across the writer and
checker. Pick a cheaper helper model in Settings, or lower `EVIDENCE_TOKENS`,
to spend less per question.

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
  app/team.py            the tutor: prompts, plan, grade, evidence packing, fact-check
  app/pipeline.py        runs plan → search → grade → write → verify, streams events
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



