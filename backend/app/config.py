"""Settings, read from the environment and from the settings panel.

Containers configure through environment variables (docker-compose.yml passes
them in, sourced from .env). Anything saved from the interface's settings panel
is stored in the database and takes precedence over the environment, so keys and
models can be changed without editing .env or restarting.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# The embedding model is fixed. Search quality depends on every passage and
# every question passing through the same model, and changing it invalidates the
# whole index, so it is not left to configuration.
EMBEDDING_PROVIDER = "openai"
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_NATIVE_DIMENSIONS = 3072

# pgvector's HNSW index cannot cover more than 2000 dimensions, so the model is
# asked for a shorter vector. At 1536 it still scores better than
# text-embedding-3-small at the same width.
MAX_INDEXED_DIMENSIONS = 2000
DEFAULT_DIMENSIONS = 1536

LLM_PROVIDERS = ("openai", "nvidia", "ollama")
NVIDIA_DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "nvidia": "meta/llama-3.3-70b-instruct",
    "ollama": "medgemma:4b",
}

# Values the settings panel may override. Each maps to the environment variable
# it replaces, so the panel and .env describe the same thing.
EDITABLE = {
    "llm_provider": "LLM_PROVIDER",
    "openai_api_key": "OPENAI_API_KEY",
    "openai_base_url": "OPENAI_BASE_URL",
    "openai_chat_model": "OPENAI_CHAT_MODEL",
    "nvidia_api_key": "NVIDIA_API_KEY",
    "nvidia_base_url": "NVIDIA_BASE_URL",
    "nvidia_chat_model": "NVIDIA_CHAT_MODEL",
    "ollama_host": "OLLAMA_HOST",
    "ollama_chat_model": "OLLAMA_CHAT_MODEL",
}
SECRETS = ("openai_api_key", "nvidia_api_key")


def _text(name: str, default: str = "") -> str:
    return (os.environ.get(name) or "").strip() or default


def _number(name: str, default: int) -> int:
    try:
        return int(_text(name) or default)
    except ValueError:
        return default


def _clear_blank_base_url() -> None:
    """Remove OPENAI_BASE_URL when it is defined but empty.

    compose always defines the variable, and `${OPENAI_BASE_URL:-}` makes it an
    empty string when unset. The OpenAI client treats a present-but-empty value
    as a deliberate override rather than falling back to its own default, then
    builds a URL with no scheme and reports the result as "Connection error" -
    which looks like a network fault and is not one. Unset beats empty.
    """
    if not (os.environ.get("OPENAI_BASE_URL") or "").strip():
        os.environ.pop("OPENAI_BASE_URL", None)


@dataclass(frozen=True)
class Settings:
    database_url: str
    document_root: Path

    llm_provider: str
    embed_provider: str
    chat_model: str
    embed_model: str
    embed_dim: int

    openai_api_key: str
    openai_base_url: str
    openai_chat_model: str
    nvidia_api_key: str
    nvidia_base_url: str
    nvidia_chat_model: str
    ollama_host: str
    ollama_chat_model: str
    ollama_num_ctx: int

    pdf_extraction: str
    child_tokens: int
    parent_tokens: int
    child_overlap: int

    query_variants: int
    candidates: int
    passages: int

    max_specialists: int
    shortlist: int
    evidence_limit: int
    use_hypothesis: bool
    enable_review: bool
    enable_suggestions: bool

    @property
    def native_dim(self) -> int:
        return EMBEDDING_NATIVE_DIMENSIONS

    @property
    def truncating(self) -> bool:
        """True when the model is asked for fewer dimensions than it produces."""
        return self.embed_dim != EMBEDDING_NATIVE_DIMENSIONS

    def describe(self) -> dict:
        return {
            "llm": f"{self.llm_provider}:{self.chat_model}",
            "embedder": f"{self.embed_provider}:{self.embed_model}",
            "dimensions": self.embed_dim,
            "truncated_from": self.native_dim if self.truncating else None,
            "pdf_extraction": self.pdf_extraction,
            "passages": self.passages,
            "max_specialists": self.max_specialists,
            "review": self.enable_review,
        }

    def problems(self) -> list[str]:
        """Anything that prevents answering, phrased for the person reading it.

        Reported rather than raised: a container that exits on a missing API key
        restarts forever and shows nothing useful. Better to come up, serve the
        interface, and say what to fix.
        """
        issues: list[str] = []
        # Search always uses OpenAI, so a key is required even when answers are
        # generated locally.
        if not self.openai_api_key:
            issues.append(
                "No OpenAI API key is set. It is needed for search regardless of "
                "which model writes the answers. Add it in Settings (the gear "
                "icon, top right) or as OPENAI_API_KEY in .env."
            )
        if not 1 <= self.embed_dim <= MAX_INDEXED_DIMENSIONS:
            issues.append(
                f"EMBED_DIM={self.embed_dim} must be between 1 and "
                f"{MAX_INDEXED_DIMENSIONS}, the limit for the vector index. "
                f"Set EMBED_DIM={DEFAULT_DIMENSIONS}."
            )
        if self.llm_provider not in LLM_PROVIDERS:
            issues.append(
                f"Unknown LLM_PROVIDER '{self.llm_provider}'. Use openai, nvidia "
                "or ollama."
            )
        if self.llm_provider == "nvidia" and not self.nvidia_api_key:
            issues.append(
                "NVIDIA is selected to write answers but no NVIDIA API key is set. "
                "Add one in Settings (the gear icon, top right)."
            )
        if not self.chat_model:
            issues.append("No chat model is selected. Choose one in Settings.")
        return issues


def load(overrides: dict | None = None) -> Settings:
    """Build settings from the environment, with saved panel values on top."""
    _clear_blank_base_url()
    saved = {k: (v or "").strip() for k, v in (overrides or {}).items()}

    def pick(key: str, default: str = "") -> str:
        return saved.get(key) or _text(EDITABLE[key], default)

    llm_provider = pick("llm_provider", "openai").lower()
    models = {
        "openai": pick("openai_chat_model", DEFAULT_MODELS["openai"]),
        "nvidia": pick("nvidia_chat_model", DEFAULT_MODELS["nvidia"]),
        "ollama": pick("ollama_chat_model", DEFAULT_MODELS["ollama"]),
    }
    chat_model = models.get(llm_provider, "")
    # Only the vector width is configurable; the model itself is not.
    embed_dim = _number("EMBED_DIM", DEFAULT_DIMENSIONS)

    return Settings(
        database_url=_text(
            "DATABASE_URL", "postgresql://aidoctor:aidoctor@db:5432/aidoctor"
        ),
        document_root=Path(_text("DOCUMENT_ROOT", "/data/documents")),
        llm_provider=llm_provider,
        embed_provider=EMBEDDING_PROVIDER,
        chat_model=chat_model,
        embed_model=EMBEDDING_MODEL,
        embed_dim=embed_dim,
        openai_api_key=pick("openai_api_key"),
        openai_base_url=pick("openai_base_url"),
        openai_chat_model=models["openai"],
        nvidia_api_key=pick("nvidia_api_key"),
        nvidia_base_url=pick("nvidia_base_url", NVIDIA_DEFAULT_BASE_URL),
        nvidia_chat_model=models["nvidia"],
        ollama_host=pick("ollama_host", "http://host.docker.internal:11434"),
        ollama_chat_model=models["ollama"],
        # Ollama's default window (a few thousand tokens) is smaller than one
        # specialist prompt; anything past it is dropped without a word.
        ollama_num_ctx=_number("OLLAMA_NUM_CTX", 16384),
        pdf_extraction=_text("PDF_EXTRACTION", "fast"),
        child_tokens=_number("CHILD_TOKENS", 400),
        parent_tokens=_number("PARENT_TOKENS", 1600),
        child_overlap=_number("CHILD_OVERLAP", 60),
        query_variants=_number("QUERY_VARIANTS", 4),
        candidates=_number("CANDIDATES", 30),
        passages=_number("PASSAGES", 8),
        max_specialists=_number("MAX_SPECIALISTS", 4),
        # Retrieved wide, then narrowed by the model: recall first, precision second.
        shortlist=_number("SHORTLIST", 24),
        # How many cited passages the reviewer and final writer see verbatim.
        evidence_limit=_number("EVIDENCE_LIMIT", 16),
        use_hypothesis=_text("USE_HYPOTHESIS", "true").lower() in ("1", "true", "yes"),
        enable_review=_text("ENABLE_REVIEW", "true").lower() in ("1", "true", "yes"),
        enable_suggestions=_text("ENABLE_SUGGESTIONS", "true").lower()
        in ("1", "true", "yes"),
    )
