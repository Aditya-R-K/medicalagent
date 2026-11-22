"""HTTP interface."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from dataclasses import replace
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import config, db, ingest, store
from .pipeline import Pipeline
from .providers import ProviderError, available_models, list_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("aidoctor")

settings = config.load()


@asynccontextmanager
async def lifespan(_: FastAPI):
    log.info("starting with %s", settings.describe())
    for issue in settings.problems():
        log.warning("configuration: %s", issue)
    db.connect(settings)
    apply_saved_settings()
    ingest.scan(settings)
    yield
    db.close()


app = FastAPI(title="AI Doctor", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatTitle(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class Question(BaseModel):
    content: str = Field(min_length=1, max_length=8000)


class ConfigUpdate(BaseModel):
    """Omitted (null) leaves a value unchanged; an empty string clears the saved
    value, so the one from .env (or the built-in default) applies again."""

    llm_provider: str | None = None
    openai_api_key: str | None = Field(None, max_length=500)
    openai_base_url: str | None = Field(None, max_length=500)
    openai_chat_model: str | None = Field(None, max_length=200)
    nvidia_api_key: str | None = Field(None, max_length=500)
    nvidia_base_url: str | None = Field(None, max_length=500)
    nvidia_chat_model: str | None = Field(None, max_length=200)
    ollama_host: str | None = Field(None, max_length=500)
    ollama_chat_model: str | None = Field(None, max_length=200)


class ModelQuery(BaseModel):
    """Lists a provider's models, optionally with values not yet saved, so a key
    can be tried before it is kept."""

    provider: str
    api_key: str | None = Field(None, max_length=500)
    base_url: str | None = Field(None, max_length=500)
    host: str | None = Field(None, max_length=500)


def apply_saved_settings() -> None:
    """Rebuild settings from the environment plus what the panel saved.

    Every request builds its pipeline from the module-level `settings`, so
    replacing it takes effect on the next question without a restart.
    """
    global settings
    try:
        saved = db.read_app_settings()
    except Exception as exc:
        log.warning("could not read saved settings: %s", exc)
        saved = {}
    settings = config.load(saved)
    log.info("settings now %s", settings.describe())


def _mask(secret: str) -> str:
    if not secret:
        return ""
    return f"{secret[:4]}…{secret[-4:]}" if len(secret) > 12 else "…" + secret[-2:]


# --------------------------------------------------------------------------- #
# system
# --------------------------------------------------------------------------- #


@app.get("/api/health")
def health() -> dict:
    try:
        db.query_one("SELECT 1 AS ok")
    except Exception as exc:
        raise HTTPException(503, f"database unavailable: {exc}") from exc
    return {"status": "ok"}


@app.get("/api/settings")
def read_settings() -> dict:
    indexed = db.query_one(
        "SELECT count(*) AS documents, coalesce(sum(chunk_count), 0) AS passages "
        "FROM documents WHERE status = 'indexed'"
    )
    return {
        **settings.describe(),
        "documents_indexed": indexed["documents"],
        "passages_indexed": indexed["passages"],
        "indexing": ingest.is_running(),
        "problems": settings.problems(),
    }


@app.get("/api/models")
def read_models(kind: str = "chat") -> dict:
    return {"models": available_models(settings, kind)}


@app.get("/api/config")
def read_config() -> dict:
    """Provider settings for the settings panel. Keys are never sent back whole."""
    saved = db.read_app_settings()

    def source(key: str) -> str:
        if saved.get(key):
            return "saved"
        return "env" if (getattr(settings, key) or "") else "none"

    return {
        "llm_provider": settings.llm_provider,
        "providers": {
            "openai": {
                "key_set": bool(settings.openai_api_key),
                "key_hint": _mask(settings.openai_api_key),
                "key_source": source("openai_api_key"),
                "base_url": settings.openai_base_url,
                "model": settings.openai_chat_model,
            },
            "nvidia": {
                "key_set": bool(settings.nvidia_api_key),
                "key_hint": _mask(settings.nvidia_api_key),
                "key_source": source("nvidia_api_key"),
                "base_url": settings.nvidia_base_url,
                "model": settings.nvidia_chat_model,
            },
            "ollama": {
                "host": settings.ollama_host,
                "model": settings.ollama_chat_model,
            },
        },
        "embedder": f"{settings.embed_provider}:{settings.embed_model}",
        "problems": settings.problems(),
    }


@app.put("/api/config")
def update_config(body: ConfigUpdate) -> dict:
    changes = body.model_dump(exclude_none=True)
    if "llm_provider" in changes:
        changes["llm_provider"] = changes["llm_provider"].strip().lower()
        if changes["llm_provider"] not in config.LLM_PROVIDERS:
            raise HTTPException(400, f"unknown provider '{changes['llm_provider']}'")
    db.write_app_settings(
        {key: (value.strip() or None) for key, value in changes.items()}
    )
    apply_saved_settings()
    return read_config()


@app.post("/api/config/models")
def query_models(body: ModelQuery) -> dict:
    provider = body.provider.strip().lower()
    if provider not in config.LLM_PROVIDERS:
        raise HTTPException(400, f"unknown provider '{provider}'")
    trial = {}
    if body.api_key and body.api_key.strip():
        trial[f"{provider}_api_key"] = body.api_key.strip()
    if body.base_url and body.base_url.strip():
        trial[f"{provider}_base_url"] = body.base_url.strip()
    if body.host and body.host.strip():
        trial["ollama_host"] = body.host.strip()
    candidate = replace(settings, **trial) if trial else settings
    try:
        return {"provider": provider, "models": list_models(candidate, provider)}
    except ProviderError as exc:
        raise HTTPException(400, str(exc)) from exc


# --------------------------------------------------------------------------- #
# documents
# --------------------------------------------------------------------------- #


@app.get("/api/documents")
def list_documents() -> dict:
    documents = db.query("SELECT * FROM documents ORDER BY filename")
    return {"documents": documents, "indexing": ingest.is_running()}


@app.post("/api/documents")
async def upload_documents(files: list[UploadFile]) -> dict:
    root = settings.document_root
    root.mkdir(parents=True, exist_ok=True)
    saved = []

    for upload in files:
        name = Path(upload.filename or "").name
        if not name:
            continue
        if Path(name).suffix.lower() not in ingest.SUPPORTED:
            raise HTTPException(400, f"{name}: only PDF, TXT and MD files are supported")
        target = root / name
        with target.open("wb") as handle:
            shutil.copyfileobj(upload.file, handle)
        saved.append(name)

    ingest.scan(settings)
    return {"saved": saved}


@app.post("/api/documents/{document_id}/exclude")
def exclude_document(document_id: int) -> dict:
    """Drop a document from the index and stop reindexing it.

    The file itself is never touched. This application has no business deleting
    documents out of a folder on someone's machine, and a mounted folder gives
    no recycle bin to recover from.
    """
    record = db.query_one("SELECT * FROM documents WHERE id = %s", (document_id,))
    if not record:
        raise HTTPException(404, "no such document")
    # The passages are kept. Search already filters on status, so excluding is
    # a flag rather than a deletion, and including again costs nothing instead
    # of paying to embed the whole document a second time.
    db.execute("UPDATE documents SET status = 'excluded' WHERE id = %s", (document_id,))
    return {"excluded": record["filename"]}


@app.post("/api/documents/{document_id}/include")
def include_document(document_id: int) -> dict:
    record = db.query_one("SELECT * FROM documents WHERE id = %s", (document_id,))
    if not record:
        raise HTTPException(404, "no such document")
    db.execute(
        """
        UPDATE documents
        SET status = CASE WHEN chunk_count > 0 THEN 'indexed' ELSE 'pending' END
        WHERE id = %s AND status = 'excluded'
        """,
        (document_id,),
    )
    return {"included": record["filename"]}


@app.post("/api/documents/index")
def start_indexing(tasks: BackgroundTasks, rebuild: bool = False) -> dict:
    issues = settings.problems()
    if issues:
        raise HTTPException(400, issues[0])
    if ingest.is_running():
        return {"started": False, "reason": "indexing is already running"}
    tasks.add_task(ingest.index_pending, settings, rebuild)
    return {"started": True}


# --------------------------------------------------------------------------- #
# chats
# --------------------------------------------------------------------------- #


@app.get("/api/chats")
def list_chats() -> dict:
    return {"chats": store.list_chats()}


@app.post("/api/chats")
def create_chat() -> dict:
    return store.create_chat()


@app.patch("/api/chats/{chat_id}")
def rename_chat(chat_id: int, body: ChatTitle) -> dict:
    if not store.get_chat(chat_id):
        raise HTTPException(404, "no such chat")
    store.rename_chat(chat_id, body.title)
    return {"id": chat_id, "title": body.title}


@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: int) -> dict:
    if not store.get_chat(chat_id):
        raise HTTPException(404, "no such chat")
    store.delete_chat(chat_id)
    return {"deleted": chat_id}


@app.get("/api/chats/{chat_id}/messages")
def list_messages(chat_id: int) -> dict:
    if not store.get_chat(chat_id):
        raise HTTPException(404, "no such chat")
    return {"messages": store.list_messages(chat_id)}


@app.post("/api/chats/{chat_id}/messages")
async def send_message(chat_id: int, body: Question) -> StreamingResponse:
    chat = store.get_chat(chat_id)
    if not chat:
        raise HTTPException(404, "no such chat")

    issues = settings.problems()
    if issues:
        raise HTTPException(400, issues[0])

    question = body.content.strip()
    history = store.history_for_model(chat_id)
    store.add_message(chat_id, "user", question)
    if chat["title"] == "New chat":
        store.rename_chat(chat_id, store.derive_title(question))

    async def events():
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def produce() -> None:
            # The pipeline is synchronous; run it off the event loop so streaming
            # to the browser is not blocked by provider calls.
            try:
                for event in Pipeline(settings).answer(question, history):
                    loop.call_soon_threadsafe(queue.put_nowait, event)
            except Exception as exc:
                log.exception("pipeline failed")
                loop.call_soon_threadsafe(
                    queue.put_nowait, {"type": "error", "text": str(exc)}
                )
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        await loop.run_in_executor(None, lambda: None)
        task = loop.run_in_executor(None, produce)
        citations: list = []

        while True:
            event = await queue.get()
            if event is None:
                break
            if event["type"] == "citations":
                citations = event["citations"]
            if event["type"] == "complete":
                result = event["result"]
                saved = store.add_message(chat_id, "assistant", result.answer, citations)
                yield _sse({"type": "done", "message_id": saved["id"]})
                continue
            yield _sse(event)

        await task

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, default=str)}\n\n"
