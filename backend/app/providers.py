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
from collections import OrderedDict
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
# Claude models that take output_config.effort. Helper calls ask for low effort.
CLAUDE_EFFORT_MODELS = re.compile(
    r"claude-(fable|mythos|opus-5|sonnet-5|haiku-5|opus-4-[5-9]|sonnet-4-6)")

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


class SafetyRejection(ProviderError):
    """The provider's content screening refused the request.

    NVIDIA's hosted API screens inputs and answers 422 "input rejected by
    content safety screening"; medical text (doses, overdose, self-harm,
    sexual health) can trip it. Retrying the same text never helps, so this is
    told apart from ordinary failures and handled where it happens.
    """


class ModelUnavailable(ProviderError):
    """The provider has no such model: retired (410 Gone) or never existed (404).

    NVIDIA retires hosted models and answers 410 for them; its model list can
    still name them for a while. Retrying never helps.
    """


def is_model_unavailable(exc: Exception) -> bool:
    if isinstance(exc, ModelUnavailable):
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    text = str(exc).lower()
    if status == 410 or ("410" in text and ("gone" in text or "error code" in text)):
        return True
    return (status == 404 or "404" in text) and any(
        m in text for m in ("model", "not found", "does not exist", "no such"))


def _unavailable_message(name: str) -> str:
    return (
        f"{name} is not available: the provider says this model has been retired "
        "or does not exist. Open Settings → Models, press Refresh, and pick a "
        "current model."
    )


def is_safety_rejection(exc: Exception) -> bool:
    text = str(exc).lower()
    return isinstance(exc, SafetyRejection) or (
        "safety" in text and any(m in text for m in ("422", "reject", "screen", "blocked"))
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
            if is_safety_rejection(exc):
                raise SafetyRejection(f"{label}: {exc}") from exc
            if is_model_unavailable(exc):
                raise ModelUnavailable(f"{label}: {exc}") from exc
            if not _retryable(exc) or attempt == attempts - 1:
                break
            time.sleep(2**attempt)
    raise ProviderError(f"{label}: {last}")


def _safety_message(name: str) -> str:
    return (
        f"{name} refused this request: the provider's content safety screening "
        "rejected it. Medical material often trips NVIDIA's screening. Choose a "
        "different answer model in Settings (another NVIDIA model, or another "
        "provider) and ask again."
    )


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
    """One chat model behind one interface, whatever the provider.

    `fast=True` selects the helper model (planning, ranking, checking) when one
    is configured, and the answer model otherwise.

    `context` is a large block shared by several calls in one question - the
    evidence. It is sent first, ahead of the role's instructions, so the calls
    share an identical prefix: Claude caches it explicitly (cache_control) and
    OpenAI-protocol providers cache matching prefixes automatically. The writer,
    the checker and the corrector then pay full price for the evidence once.
    """

    def __init__(self, settings, fast: bool = False) -> None:
        if settings.llm_provider not in CHAT_PROVIDERS:
            raise ProviderError(f"unknown chat provider '{settings.llm_provider}'")
        self.settings = settings
        self.provider = settings.llm_provider
        self.model = (settings.fast_model if fast and settings.fast_model
                      else settings.chat_model)
        self.name = f"{self.provider}:{self.model}"
        self._client = _chat_client(settings)
        # OpenAI-protocol providers (OpenAI and NVIDIA). Some of their models
        # refuse temperature=0 (reasoning models) or JSON mode; both are
        # dropped after the first refusal instead of failing the answer.
        self._compatible = self.provider in ("openai", "nvidia")
        self._send_temperature = True
        self._send_json_mode = self.provider == "openai"
        self._send_effort = self.provider == "anthropic" and bool(
            CLAUDE_EFFORT_MODELS.match(self.model))

    # ---------------------------------------------------------- message shapes

    @staticmethod
    def _system_text(system: str, context: str | None) -> str:
        return f"{context}\n\n{system}" if context else system

    def _messages(self, system: str, user: str, history: list[dict] | None,
                  context: str | None = None) -> list[dict]:
        messages = [{"role": "system", "content": self._system_text(system, context)}]
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

    def _claude_arguments(self, system: str, user: str, history, max_tokens: int,
                          context: str | None = None, effort: str | None = None) -> dict:
        if context:
            # Evidence first, with a cache breakpoint after it: every call in
            # this question that sends the same evidence reads it from cache.
            system_blocks = [
                {"type": "text", "text": context, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": system},
            ]
        else:
            system_blocks = system
        arguments = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system_blocks,
            "messages": self._claude_messages(user, history),
        }
        if effort and self._send_effort:
            arguments["output_config"] = {"effort": effort}
        if self.model in CLAUDE_FALLBACK_MODELS:
            arguments["betas"] = [CLAUDE_FALLBACK_BETA]
            arguments["extra_body"] = {"fallbacks": "default"}
        return arguments

    def _claude_api(self):
        """The beta namespace when the refusal fallback is in use, else the plain one."""
        return self._client.beta.messages if self.model in CLAUDE_FALLBACK_MODELS \
            else self._client.messages

    def _claude_call(self, method: str, arguments: dict):
        """messages.create / .stream, retried once without effort if refused."""
        try:
            return getattr(self._claude_api(), method)(**arguments)
        except Exception as exc:
            if "output_config" in arguments and ("effort" in str(exc).lower()
                                                 or "output_config" in str(exc).lower()):
                self._send_effort = False
                arguments.pop("output_config")
                return getattr(self._claude_api(), method)(**arguments)
            raise

    def _claude_check(self, message) -> None:
        if getattr(message, "stop_reason", None) == "refusal":
            details = getattr(message, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise ProviderError(
                f"{self.name} declined to answer"
                + (f" ({category})" if category else "")
                + ". Try rephrasing, or choose another model in Settings."
            )
        _log_usage(self.name, getattr(message, "usage", None))

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
        # start of anything longer. The evidence alone runs to several
        # thousand tokens, so the window is set explicitly.
        return {"temperature": 0, "num_ctx": self.settings.ollama_num_ctx}

    # ------------------------------------------------------------------ calls

    def complete(self, system: str, user: str, history: list[dict] | None = None,
                 json_only: bool = False, context: str | None = None,
                 effort: str | None = None) -> str:
        """`effort` ("low" | "medium" | "high") applies to Claude models that
        support it; helper calls pass "low" to keep thinking short."""
        def call() -> str:
            if self.provider == "anthropic":
                message = self._claude_call("create", self._claude_arguments(
                    system, user, history, 16000, context, effort))
                self._claude_check(message)
                return "".join(b.text for b in message.content if b.type == "text")

            messages = self._messages(system, user, history, context)
            if self._compatible:
                arguments = self._openai_arguments(messages)
                if json_only and self._send_json_mode:
                    arguments["response_format"] = {"type": "json_object"}
                response = self._create(**arguments)
                _log_usage(self.name, getattr(response, "usage", None))
                return response.choices[0].message.content or ""
            response = self._client.chat(
                model=self.model, messages=messages, options=self._ollama_options(),
                format="json" if json_only else "",
            )
            return response["message"]["content"]

        try:
            return _with_retries(call, 4, f"{self.name} request failed").strip()
        except SafetyRejection as exc:
            raise SafetyRejection(_safety_message(self.name)) from exc
        except ModelUnavailable as exc:
            raise ModelUnavailable(_unavailable_message(self.name)) from exc

    def stream(self, system: str, user: str, history: list[dict] | None = None,
               context: str | None = None, effort: str | None = None) -> Iterator[str]:
        """Yield the reply in pieces, so the interface can render as it arrives."""
        try:
            if self.provider == "anthropic":
                with self._claude_call("stream", self._claude_arguments(
                        system, user, history, 32000, context, effort)) as stream:
                    for text in stream.text_stream:
                        yield text
                    self._claude_check(stream.get_final_message())
            elif self._compatible:
                messages = self._messages(system, user, history, context)
                arguments = self._openai_arguments(messages, stream=True)
                if self.provider == "openai":
                    arguments["stream_options"] = {"include_usage": True}
                for part in self._create(**arguments):
                    if getattr(part, "usage", None):
                        _log_usage(self.name, part.usage)
                    if part.choices and part.choices[0].delta.content:
                        yield part.choices[0].delta.content
            else:
                for part in self._client.chat(
                    model=self.model,
                    messages=self._messages(system, user, history, context),
                    options=self._ollama_options(), stream=True,
                ):
                    piece = part.get("message", {}).get("content")
                    if piece:
                        yield piece
        except ProviderError:
            raise
        except Exception as exc:
            if is_safety_rejection(exc):
                raise SafetyRejection(_safety_message(self.name)) from exc
            if is_model_unavailable(exc):
                raise ModelUnavailable(_unavailable_message(self.name)) from exc
            raise ProviderError(f"{self.name} stream failed: {exc}") from exc

    def structured(self, system: str, user: str, fallback: dict,
                   context: str | None = None) -> tuple[dict, bool]:
        """Ask for JSON, falling back rather than failing when it cannot be parsed.

        A safety refusal also falls back: planning, grading and checking all
        have a sensible default, and one refused helper call must not cost the
        student the whole answer.
        """
        try:
            reply = self.complete(
                f"{system}\n\nReply with a single JSON object and nothing else.", user,
                json_only=True, context=context, effort="low",
            )
        except SafetyRejection:
            log.warning("%s: helper call refused by content screening; using fallback",
                        self.name)
            return dict(fallback), False
        parsed = parse_json(reply)
        if parsed is None:
            log.info("%s returned unparseable JSON; using fallback", self.name)
            return dict(fallback), False
        return parsed, True


def probe_chat(settings) -> None:
    """Send the answer model (and helper, if set) one tiny message.

    A model can be listed by its provider and still be retired, or be a typed
    name with a typo; finding out on Save is far better than on the student's
    first question. A content-screening refusal of the word "ping" would be
    odd, but it would not mean the model is unusable, so it passes.
    """
    for fast in ((False, True) if settings.fast_model else (False,)):
        model = ChatModel(settings, fast=fast)
        try:
            model.complete("Reply with the single word OK.", "ping", effort="low")
        except SafetyRejection:
            continue


def _log_usage(name: str, usage) -> None:
    """Token use per call, so the cost of a question is visible in the log."""
    if usage is None:
        return
    fields = {
        "in": getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None),
        "out": getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", None),
        "cache_read": getattr(usage, "cache_read_input_tokens", None)
        or getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", None),
        "cache_write": getattr(usage, "cache_creation_input_tokens", None),
    }
    log.info("tokens %s %s", name, " ".join(f"{k}={v}" for k, v in fields.items() if v))


# --------------------------------------------------------------------------- #
# embeddings
# --------------------------------------------------------------------------- #


_QUERY_CACHE: OrderedDict = OrderedDict()


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
        # Passages the provider's content screening refused while indexing.
        self.rejected = 0
        self._client = _embed_client(settings)

    def embed(self, texts: Sequence[str], kind: str = "passage") -> list[list[float] | None]:
        """`kind` is "passage" for indexed text and "query" for questions; NVIDIA's
        retrieval models embed the two differently.

        For passages, a vector is None where the provider's content screening
        refused that passage: the rest of the document still gets indexed.
        """
        vectors: list[list[float] | None] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [truncate(t, self.max_tokens) for t in texts[start : start + self.batch_size]]
            if kind == "passage":
                vectors.extend(self._encode_screened(batch, kind))
            else:
                vectors.extend(self._encode(batch, kind))
        return vectors

    def _encode_screened(self, batch: list[str], kind: str) -> list[list[float] | None]:
        """Encode, isolating any passage the content screening refuses.

        One refused passage fails its whole batch, so the batch is halved until
        the refused passage is alone - a few extra requests, instead of losing
        a document over one paragraph about overdose management.
        """
        try:
            return self._encode(batch, kind)
        except SafetyRejection:
            if len(batch) == 1:
                self.rejected += 1
                log.warning("%s refused a passage (content screening): %.80s...",
                            self.name, batch[0].replace("\n", " "))
                return [None]
            middle = len(batch) // 2
            return (self._encode_screened(batch[:middle], kind)
                    + self._encode_screened(batch[middle:], kind))

    def embed_one(self, text: str, kind: str = "query") -> list[float]:
        return self._encode([truncate(text, self.max_tokens)], kind)[0]

    def _encode(self, batch: list[str], kind: str) -> list[list[float]]:
        # Questions repeat (rephrasings, follow-ups, the same student asking
        # again); their vectors are kept rather than paid for twice.
        if kind == "query":
            keys = [(self.name, self.dim, text) for text in batch]
            missing = [t for k, t in zip(keys, batch) if k not in _QUERY_CACHE]
            if missing:
                for text, vector in zip(missing, self._encode_uncached(missing, kind)):
                    _QUERY_CACHE[(self.name, self.dim, text)] = vector
                    if len(_QUERY_CACHE) > 4096:
                        _QUERY_CACHE.popitem(last=False)
            return [_QUERY_CACHE[k] for k in keys]
        return self._encode_uncached(batch, kind)

    def _encode_uncached(self, batch: list[str], kind: str) -> list[list[float]]:
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
