"""Chat and embedding providers.

Answers:    OpenAI, Anthropic (Claude), NVIDIA NIM, Ollama
Embeddings: OpenAI, NVIDIA NIM, Ollama  (Anthropic has no embedding API)

NVIDIA NIM speaks the OpenAI protocol, whether hosted (integrate.api.nvidia.com,
keys start with nvapi-) or self-hosted (your own base URL, usually no key).
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Iterator, Sequence

log = logging.getLogger(__name__)

OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
NVIDIA_DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"

CHAT_PROVIDERS = ("openai", "anthropic", "nvidia", "ollama")
EMBED_PROVIDERS = ("openai", "nvidia", "ollama")

# pgvector's HNSW index covers at most 2000 dimensions. Models that produce
# more are asked for a shorter vector when they support it.
MAX_INDEXED_DIMENSIONS = 2000
SHORTENED_DIMENSIONS = 1536

# Claude models that accept the server-side refusal fallback. On a safety
# decline the API re-runs the request on a suitable model within the same call.
CLAUDE_FALLBACK_MODELS = {
    "claude-fable-5-2", "claude-fable-5-1", "claude-opus-5-5", "claude-opus-5",
    "claude-sonnet-5-5",
}
CLAUDE_FALLBACK_BETA = "server-side-fallback-2026-07-01"

try:
    import tiktoken

    _ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover
    _ENCODING = None


class ProviderError(RuntimeError):
    pass


def count_tokens(text: str) -> int:
    if _ENCODING is not None:
        return max(1, len(_ENCODING.encode(text)))
    return max(1, int(len(text.split()) * 1.3))


def truncate(text: str, limit: int) -> str:
    if _ENCODING is None:
        words = text.split()
        budget = int(limit / 1.3)
        return text if len(words) <= budget else " ".join(words[:budget])
    tokens = _ENCODING.encode(text)
    return text if len(tokens) <= limit else _ENCODING.decode(tokens[:limit])


def parse_json(text: str) -> dict | None:
    """Pull a JSON object out of a reply that may be wrapped in prose or fences."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, flags=re.S)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _retryable(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(
        marker in text
        for marker in ("ratelimit", "rate limit", "timeout", "connection", "502", "503", "529",
                       "overloaded")
    )


def _with_retries(operation, attempts: int, label: str):
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return operation()
        except ProviderError:
            raise
        except Exception as exc:
            last = exc
            if not _retryable(exc) or attempt == attempts - 1:
                break
            time.sleep(2**attempt)
    raise ProviderError(f"{label}: {last}")


# --------------------------------------------------------------------------- #
# clients
# --------------------------------------------------------------------------- #


def nvidia_needs_key(base_url: str) -> bool:
    """NVIDIA's hosted API needs a key; a self-hosted NIM usually does not."""
    return "api.nvidia.com" in (base_url or NVIDIA_DEFAULT_BASE_URL)


def _openai_client(settings):
    from openai import OpenAI

    if not settings.openai_api_key:
        raise ProviderError("No OpenAI API key is set. Add one in Settings.")
    # Passed explicitly rather than left to the client's environment lookup,
    # which accepts an empty OPENAI_BASE_URL as a real override.
    return OpenAI(
        api_key=settings.openai_api_key,
        base_url=(settings.openai_base_url or "").strip() or OPENAI_DEFAULT_BASE_URL,
    )


def _nvidia_client(settings, base_url: str | None = None):
    from openai import OpenAI

    base = (base_url or settings.nvidia_base_url or "").strip() or NVIDIA_DEFAULT_BASE_URL
    if nvidia_needs_key(base) and not settings.nvidia_api_key:
        raise ProviderError("No NVIDIA API key is set. Add one in Settings.")
    # The client insists on some key; a self-hosted NIM ignores it.
    return OpenAI(api_key=settings.nvidia_api_key or "not-needed", base_url=base)


def _anthropic_client(settings):
    import anthropic

    if not settings.anthropic_api_key:
        raise ProviderError("No Anthropic API key is set. Add one in Settings.")
    return anthropic.Anthropic(api_key=settings.anthropic_api_key)


def _ollama_client(settings):
    import ollama

    return ollama.Client(host=settings.ollama_host)


def _chat_client(settings):
    return {
        "openai": _openai_client,
        "anthropic": _anthropic_client,
        "nvidia": _nvidia_client,
        "ollama": _ollama_client,
    }[settings.llm_provider](settings)


def _embed_client(settings):
    if settings.embed_provider == "ollama":
        return _ollama_client(settings)
    if settings.embed_provider == "nvidia":
        return _nvidia_client(settings, settings.nvidia_embed_base_url or settings.nvidia_base_url)
    return _openai_client(settings)


# --------------------------------------------------------------------------- #
# chat
# --------------------------------------------------------------------------- #


class ChatModel:
    def __init__(self, settings) -> None:
        if settings.llm_provider not in CHAT_PROVIDERS:
            raise ProviderError(f"unknown chat provider '{settings.llm_provider}'")
        self.settings = settings
        self.provider = settings.llm_provider
        self.model = settings.chat_model
        self.name = f"{self.provider}:{self.model}"
        self._client = _chat_client(settings)
        # OpenAI-protocol providers (OpenAI and NVIDIA). Some of their models
        # refuse temperature=0 (reasoning models) or JSON mode; both are
        # dropped after the first refusal instead of failing the answer.
        self._compatible = self.provider in ("openai", "nvidia")
        self._send_temperature = True
        self._send_json_mode = self.provider == "openai"

    # ---------------------------------------------------------- message shapes

    def _messages(self, system: str, user: str, history: list[dict] | None) -> list[dict]:
        messages = [{"role": "system", "content": system}]
        messages.extend(history or [])
        messages.append({"role": "user", "content": user})
        return messages

    @staticmethod
    def _claude_messages(user: str, history: list[dict] | None) -> list[dict]:
        """History starting with a user turn and alternating roles, as Claude
        expects. A failed answer can leave two user turns in a row; they merge."""
        turns: list[dict] = []
        for turn in [*(history or []), {"role": "user", "content": user}]:
            if not turns and turn["role"] != "user":
                continue
            if turns and turns[-1]["role"] == turn["role"]:
                turns[-1] = {"role": turn["role"],
                             "content": f"{turns[-1]['content']}\n\n{turn['content']}"}
            else:
                turns.append({"role": turn["role"], "content": turn["content"]})
        return turns

    def _claude_arguments(self, system: str, user: str, history, max_tokens: int) -> dict:
        arguments = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": self._claude_messages(user, history),
        }
        if self.model in CLAUDE_FALLBACK_MODELS:
            arguments["betas"] = [CLAUDE_FALLBACK_BETA]
            arguments["extra_body"] = {"fallbacks": "default"}
        return arguments

    def _claude_api(self):
        """The beta namespace when the refusal fallback is in use, else the plain one."""
        return self._client.beta.messages if self.model in CLAUDE_FALLBACK_MODELS \
            else self._client.messages

    def _claude_check(self, message) -> None:
        if getattr(message, "stop_reason", None) == "refusal":
            details = getattr(message, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise ProviderError(
                f"{self.name} declined to answer"
                + (f" ({category})" if category else "")
                + ". Try rephrasing, or choose another model in Settings."
            )

    def _openai_arguments(self, messages: list[dict], **extra) -> dict:
        arguments = {"model": self.model, "messages": messages, **extra}
        if self._send_temperature:
            arguments["temperature"] = 0
        return arguments

    def _create(self, **arguments):
        """chat.completions.create, retried once without what the model refused."""
        try:
            return self._client.chat.completions.create(**arguments)
        except Exception as exc:
            text = str(exc).lower()
            if "temperature" in text and "temperature" in arguments:
                self._send_temperature = False
                arguments.pop("temperature")
                return self._create(**arguments)
            if "response_format" in text and "response_format" in arguments:
                self._send_json_mode = False
                arguments.pop("response_format")
                return self._create(**arguments)
            raise

    def _ollama_options(self) -> dict:
        # Ollama defaults to a small context window and silently drops the
        # start of anything longer. The specialist prompts carry several
        # thousand tokens of passages, so the window is set explicitly.
        return {"temperature": 0, "num_ctx": self.settings.ollama_num_ctx}

    # ------------------------------------------------------------------ calls

    def complete(self, system: str, user: str, history: list[dict] | None = None,
                 json_only: bool = False) -> str:
        def call() -> str:
            if self.provider == "anthropic":
                message = self._claude_api().create(
                    **self._claude_arguments(system, user, history, 16000)
                )
                self._claude_check(message)
                return "".join(b.text for b in message.content if b.type == "text")

            messages = self._messages(system, user, history)
            if self._compatible:
                arguments = self._openai_arguments(messages)
                if json_only and self._send_json_mode:
                    arguments["response_format"] = {"type": "json_object"}
                response = self._create(**arguments)
                return response.choices[0].message.content or ""
            response = self._client.chat(
                model=self.model, messages=messages, options=self._ollama_options(),
                format="json" if json_only else "",
            )
            return response["message"]["content"]

        return _with_retries(call, 4, f"{self.name} request failed").strip()

    def stream(
        self, system: str, user: str, history: list[dict] | None = None
    ) -> Iterator[str]:
        """Yield the reply in pieces, so the interface can render as it arrives."""
        try:
            if self.provider == "anthropic":
                with self._claude_api().stream(
                    **self._claude_arguments(system, user, history, 32000)
                ) as stream:
                    for text in stream.text_stream:
                        yield text
                    self._claude_check(stream.get_final_message())
            elif self._compatible:
                messages = self._messages(system, user, history)
                for part in self._create(**self._openai_arguments(messages, stream=True)):
                    if part.choices and part.choices[0].delta.content:
                        yield part.choices[0].delta.content
            else:
                for part in self._client.chat(
                    model=self.model, messages=self._messages(system, user, history),
                    options=self._ollama_options(), stream=True,
                ):
                    piece = part.get("message", {}).get("content")
                    if piece:
                        yield piece
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"{self.name} stream failed: {exc}") from exc

    def structured(self, system: str, user: str, fallback: dict) -> tuple[dict, bool]:
        """Ask for JSON, falling back rather than failing when it cannot be parsed."""
        reply = self.complete(
            f"{system}\n\nReply with a single JSON object and nothing else.", user,
            json_only=True,
        )
        parsed = parse_json(reply)
        if parsed is None:
            log.info("%s returned unparseable JSON; using fallback", self.name)
            return dict(fallback), False
        return parsed, True


# --------------------------------------------------------------------------- #
# embeddings
# --------------------------------------------------------------------------- #


class EmbeddingModel:
    """Turns passages and questions into vectors with the selected model.

    Every passage and every question must pass through the same model for the
    comparison to mean anything, so the index records which model built it and
    is rebuilt when the selection changes (see db.reconcile_index).
    """

    def __init__(self, settings) -> None:
        if settings.embed_provider not in EMBED_PROVIDERS:
            raise ProviderError(f"unknown embedding provider '{settings.embed_provider}'")
        self.settings = settings
        self.provider = settings.embed_provider
        self.model = settings.embed_model
        self.dim = settings.embed_dim
        self.name = f"{self.provider}:{self.model}"
        self.batch_size = 64 if self.provider == "nvidia" else 128
        self.max_tokens = 8000
        self._client = _embed_client(settings)

    def embed(self, texts: Sequence[str], kind: str = "passage") -> list[list[float]]:
        """`kind` is "passage" for indexed text and "query" for questions; NVIDIA's
        retrieval models embed the two differently."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [truncate(t, self.max_tokens) for t in texts[start : start + self.batch_size]]
            vectors.extend(self._encode(batch, kind))
        return vectors

    def embed_one(self, text: str, kind: str = "query") -> list[float]:
        return self._encode([truncate(text, self.max_tokens)], kind)[0]

    def _encode(self, batch: list[str], kind: str) -> list[list[float]]:
        vectors = _with_retries(
            lambda: raw_embed(self.settings, self._client, batch, kind,
                              self.dim if self.settings.embed_send_dimensions else None),
            4, f"{self.name} embedding failed",
        )
        if vectors and self.dim and len(vectors[0]) != self.dim:
            raise ProviderError(
                f"{self.name} returned {len(vectors[0])} dimensions but the index "
                f"holds {self.dim}. Re-select the embedding model in Settings."
            )
        return vectors


def raw_embed(settings, client, batch: list[str], kind: str,
              dimensions: int | None) -> list[list[float]]:
    provider = settings.embed_provider
    if provider == "ollama":
        response = client.embed(model=settings.embed_model, input=batch)
        return [list(v) for v in response["embeddings"]]
    arguments = {"model": settings.embed_model, "input": batch, "encoding_format": "float"}
    if dimensions:
        arguments["dimensions"] = dimensions
    if provider == "nvidia":
        # Retrieval models embed questions and passages differently, and
        # reject input past their window unless told to cut it.
        arguments["extra_body"] = {"input_type": kind, "truncate": "END"}
    response = client.embeddings.create(**arguments)
    return [item.embedding for item in response.data]


def probe_embedding(settings) -> tuple[int, bool]:
    """Find the vector width of the selected embedding model.

    Returns (dimensions, send_dimensions). A model wider than the vector index
    allows is asked for a shorter vector when it supports that (OpenAI's
    text-embedding-3 models, NVIDIA's Matryoshka models); otherwise the full
    width is kept and search runs without the approximate index.
    """
    client = _embed_client(settings)
    sample = ["dimension probe"]
    try:
        native = len(raw_embed(settings, client, sample, "passage", None)[0])
    except Exception as exc:
        raise ProviderError(f"{settings.embed_provider}:{settings.embed_model} "
                            f"could not embed: {exc}") from exc
    if native <= MAX_INDEXED_DIMENSIONS or settings.embed_provider == "ollama":
        return native, False
    try:
        shorter = len(raw_embed(settings, client, sample, "passage", SHORTENED_DIMENSIONS)[0])
        if shorter == SHORTENED_DIMENSIONS:
            return SHORTENED_DIMENSIONS, True
    except Exception as exc:
        log.info("%s does not shorten vectors: %s", settings.embed_model, exc)
    return native, False


# --------------------------------------------------------------------------- #
# model listings
# --------------------------------------------------------------------------- #

# Model ids on the OpenAI-protocol providers that cannot write answers.
_NOT_CHAT = (
    "embed", "rerank", "whisper", "tts", "dall-e", "image", "moderation",
    "transcribe", "audio", "realtime", "search", "babbage", "davinci",
    "reward", "guard", "safety", "clip", "parse", "ocr", "retriever", "sora",
)
_EMBED_HINTS = ("embed", "bge", "e5-", "minilm", "gte-", "arctic", "nomic", "mxbai")


def _is_embedding(name: str) -> bool:
    lowered = name.lower()
    return any(h in lowered for h in _EMBED_HINTS) and "rerank" not in lowered


def list_models(settings, provider: str, kind: str = "chat") -> list[str]:
    """Models a provider offers for answering (`chat`) or search (`embedding`).
    Raises ProviderError when the provider cannot be reached."""
    try:
        if provider == "ollama":
            entries = _ollama_client(settings).list().get("models", [])
            names = sorted(n for n in (e.get("model") or e.get("name") for e in entries) if n)
        elif provider == "anthropic":
            if kind == "embedding":
                return []
            names = [m.id for m in _anthropic_client(settings).models.list()]
        elif provider == "nvidia":
            base = settings.nvidia_embed_base_url if kind == "embedding" else None
            names = sorted({m.id for m in _nvidia_client(settings, base).models.list().data})
        elif provider == "openai":
            names = sorted({m.id for m in _openai_client(settings).models.list().data})
        else:
            raise ProviderError(f"unknown provider '{provider}'")
    except ProviderError:
        raise
    except Exception as exc:
        raise ProviderError(f"could not list {provider} models: {exc}") from exc

    if kind == "embedding":
        embedders = [n for n in names if _is_embedding(n)]
        # Ollama names are free-form; show everything rather than nothing.
        return embedders or (names if provider == "ollama" else [])
    if provider == "ollama":
        return [n for n in names if not _is_embedding(n)]
    if provider == "anthropic":
        return names  # newest first, as the API returns them
    return [n for n in names if not any(m in n.lower() for m in _NOT_CHAT)]


def available_models(settings, kind: str) -> list[str]:
    """What the configured provider reports. Empty when it cannot be reached."""
    provider = settings.embed_provider if kind == "embedding" else settings.llm_provider
    try:
        return list_models(settings, provider, kind)
    except Exception as exc:
        log.info("could not list models: %s", exc)
    return []
