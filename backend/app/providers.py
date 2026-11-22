"""Chat and embedding providers: OpenAI, NVIDIA (OpenAI-compatible) and Ollama."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Iterator, Sequence

log = logging.getLogger(__name__)

OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
NVIDIA_DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"

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
        for marker in ("ratelimit", "rate limit", "timeout", "connection", "502", "503", "529")
    )


def _with_retries(operation, attempts: int, label: str):
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as exc:
            last = exc
            if not _retryable(exc) or attempt == attempts - 1:
                break
            time.sleep(2**attempt)
    raise ProviderError(f"{label}: {last}")


def _openai_client(settings):
    from openai import OpenAI

    if not settings.openai_api_key:
        raise ProviderError("OPENAI_API_KEY is not set")
    # Passed explicitly rather than left to the client's environment lookup,
    # which accepts an empty OPENAI_BASE_URL as a real override.
    return OpenAI(
        api_key=settings.openai_api_key,
        base_url=(settings.openai_base_url or "").strip() or OPENAI_DEFAULT_BASE_URL,
    )


def _nvidia_client(settings):
    from openai import OpenAI

    if not settings.nvidia_api_key:
        raise ProviderError("No NVIDIA API key is set")
    # NVIDIA's hosted models (build.nvidia.com) speak the OpenAI protocol.
    return OpenAI(
        api_key=settings.nvidia_api_key,
        base_url=(settings.nvidia_base_url or "").strip() or NVIDIA_DEFAULT_BASE_URL,
    )


def _ollama_client(settings):
    import ollama

    return ollama.Client(host=settings.ollama_host)


def _chat_client(settings):
    if settings.llm_provider == "ollama":
        return _ollama_client(settings)
    if settings.llm_provider == "nvidia":
        return _nvidia_client(settings)
    return _openai_client(settings)


# --------------------------------------------------------------------------- #
# chat
# --------------------------------------------------------------------------- #


class ChatModel:
    def __init__(self, settings) -> None:
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

    def _messages(self, system: str, user: str, history: list[dict] | None) -> list[dict]:
        messages = [{"role": "system", "content": system}]
        messages.extend(history or [])
        messages.append({"role": "user", "content": user})
        return messages

    def _ollama_options(self) -> dict:
        # Ollama defaults to a small context window and silently drops the
        # start of anything longer. The specialist prompts carry several
        # thousand tokens of passages, so the window is set explicitly.
        return {"temperature": 0, "num_ctx": self.settings.ollama_num_ctx}

    def complete(self, system: str, user: str, history: list[dict] | None = None,
                 json_only: bool = False) -> str:
        messages = self._messages(system, user, history)

        def call() -> str:
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
        messages = self._messages(system, user, history)
        try:
            if self._compatible:
                stream = self._create(**self._openai_arguments(messages, stream=True))
                for part in stream:
                    if part.choices and part.choices[0].delta.content:
                        yield part.choices[0].delta.content
            else:
                for part in self._client.chat(
                    model=self.model, messages=messages,
                    options=self._ollama_options(), stream=True,
                ):
                    piece = part.get("message", {}).get("content")
                    if piece:
                        yield piece
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
    """Fixed to OpenAI's text-embedding-3-large.

    Every passage and every question must pass through the same model for the
    comparison to mean anything, and changing it invalidates the whole index, so
    this is deliberately not configurable.
    """

    def __init__(self, settings) -> None:
        self.settings = settings
        self.provider = settings.embed_provider
        self.model = settings.embed_model
        self.dim = settings.embed_dim
        self.name = f"{self.provider}:{self.model}"
        self.batch_size = 128
        self.max_tokens = 8000
        self._client = _openai_client(settings)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [truncate(t, self.max_tokens) for t in texts[start : start + self.batch_size]]
            vectors.extend(self._encode(batch))
        return vectors

    def embed_one(self, text: str) -> list[float]:
        return self._encode([truncate(text, self.max_tokens)])[0]

    def _encode(self, batch: list[str]) -> list[list[float]]:
        def call() -> list[list[float]]:
            arguments = {"model": self.model, "input": batch}
            if self.settings.truncating:
                # Shortening the vector keeps it inside pgvector's index limit
                # and still beats a smaller model at the same width.
                arguments["dimensions"] = self.dim
            response = self._client.embeddings.create(**arguments)
            return [item.embedding for item in response.data]

        vectors = _with_retries(call, 4, f"{self.name} embedding failed")
        if vectors and len(vectors[0]) != self.dim:
            raise ProviderError(
                f"{self.name} returned {len(vectors[0])} dimensions but EMBED_DIM "
                f"is {self.dim}."
            )
        return vectors


# Model ids on the OpenAI-protocol providers that cannot write answers.
_NOT_CHAT = (
    "embed", "rerank", "whisper", "tts", "dall-e", "image", "moderation",
    "transcribe", "audio", "realtime", "search", "babbage", "davinci",
    "reward", "guard", "safety", "clip", "parse", "ocr", "retriever",
)


def list_models(settings, provider: str | None = None) -> list[str]:
    """Chat models the given provider offers. Raises ProviderError on failure."""
    provider = provider or settings.llm_provider
    try:
        if provider == "ollama":
            entries = _ollama_client(settings).list().get("models", [])
            return sorted(
                n for n in (e.get("model") or e.get("name") for e in entries) if n
            )
        client = _nvidia_client(settings) if provider == "nvidia" else _openai_client(settings)
        ids = sorted({model.id for model in client.models.list().data})
    except ProviderError:
        raise
    except Exception as exc:
        raise ProviderError(f"could not list {provider} models: {exc}") from exc
    return [i for i in ids if not any(marker in i.lower() for marker in _NOT_CHAT)]


def available_models(settings, kind: str) -> list[str]:
    """What the configured provider reports. Empty when it cannot be reached."""
    try:
        if kind == "embedding":
            ids = sorted(m.id for m in _openai_client(settings).models.list().data)
            return [i for i in ids if "embedding" in i]
        return list_models(settings)
    except Exception as exc:
        log.info("could not list models: %s", exc)
    return []
