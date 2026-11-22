"""Settings, from the settings panel first and the environment second.

API keys, providers and models are entered in the interface (the gear icon) and
stored in the database; no .env file is needed. Environment variables still
work as defaults, for deployments that prefer them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# pgvector's HNSW index cannot cover more than 2000 dimensions. Wider models are
# asked for a shorter vector where they support it (see providers.probe_embedding).
MAX_INDEXED_DIMENSIONS = 2000
DEFAULT_DIMENSIONS = 1536

LLM_PROVIDERS = ("openai", "anthropic", "nvidia", "ollama")
EMBED_PROVIDERS = ("openai", "nvidia", "ollama")
NVIDIA_DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
PROVIDER_LABELS = {
    "openai": "OpenAI", "anthropic": "Claude", "nvidia": "NVIDIA NIM", "ollama": "Ollama",
}
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-opus-5-5",
    "nvidia": "meta/llama-3.3-70b-instruct",
    "ollama": "medgemma:4b",
}
DEFAULT_EMBED_MODELS = {
    "openai": "text-embedding-3-large",
    "nvidia": "nvidia/llama-3.2-nv-embedqa-1b-v2",
    "ollama": "nomic-embed-text",
}

# Values the settings panel may override, each mapped to the environment variable
# it replaces. Neither is required: with no .env at all, everything is entered
# in the panel and kept in the database.
EDITABLE = {
    "llm_provider": "LLM_PROVIDER",
    "openai_api_key": "OPENAI_API_KEY",
    "openai_base_url": "OPENAI_BASE_URL",
    "openai_chat_model": "OPENAI_CHAT_MODEL",
    "anthropic_api_key": "ANTHROPIC_API_KEY",
    "anthropic_chat_model": "ANTHROPIC_CHAT_MODEL",
    "nvidia_api_key": "NVIDIA_API_KEY",
    "nvidia_base_url": "NVIDIA_BASE_URL",
    "nvidia_embed_base_url": "NVIDIA_EMBED_BASE_URL",
    "nvidia_chat_model": "NVIDIA_CHAT_MODEL",
    "ollama_host": "OLLAMA_HOST",
    "ollama_chat_model": "OLLAMA_CHAT_MODEL",
    "embed_provider": "EMBED_PROVIDER",
    "embed_model": "EMBED_MODEL",
    # Optional cheaper model per provider for planning, grading and checking.
    "openai_fast_model": "OPENAI_FAST_MODEL",
    "anthropic_fast_model": "ANTHROPIC_FAST_MODEL",
    "nvidia_fast_model": "NVIDIA_FAST_MODEL",
    "ollama_fast_model": "OLLAMA_FAST_MODEL",
}
SECRETS = ("openai_api_key", "anthropic_api_key", "nvidia_api_key")
# Written by the backend after it measures the embedding model, not by people.
MEASURED = ("embed_dim", "embed_send_dimensions")


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
    chat_model: str
    # Helper model for the active provider; empty means "use chat_model".
    fast_model: str
    embed_provider: str
    embed_model: str
    embed_dim: int
    # True when the model is asked for fewer dimensions than it natively makes.
    embed_send_dimensions: bool

    openai_api_key: str
    openai_base_url: str
    openai_chat_model: str
    anthropic_api_key: str
    anthropic_chat_model: str
    nvidia_api_key: str
    nvidia_base_url: str
    nvidia_embed_base_url: str
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
    shortlist: int
    evidence_tokens: int
    use_hypothesis: bool
    enable_review: bool
    enable_suggestions: bool
    fast_models: dict = field(default_factory=dict)

    @property
    def truncating(self) -> bool:
        return self.embed_send_dimensions

    @property
    def index_signature(self) -> str:
        """Identifies the vectors in the index: built by this model, this wide."""
        return f"{self.embed_provider}:{self.embed_model}:{self.embed_dim}"

    @property
    def vector_indexed(self) -> bool:
        return 1 <= self.embed_dim <= MAX_INDEXED_DIMENSIONS

    def describe(self) -> dict:
        return {
            "llm": f"{self.llm_provider}:{self.chat_model}",
            "embedder": f"{self.embed_provider}:{self.embed_model}",
            "dimensions": self.embed_dim,
            "pdf_extraction": self.pdf_extraction,
            "helper": f"{self.llm_provider}:{self.fast_model}" if self.fast_model else None,
            "passages": self.passages,
            "evidence_tokens": self.evidence_tokens,
            "review": self.enable_review,
        }

    def key_problem(self, provider: str, purpose: str) -> str | None:
        label = PROVIDER_LABELS.get(provider, provider)
        nvidia_url = (self.nvidia_base_url if purpose == "answers"
                      else self.nvidia_embed_base_url or self.nvidia_base_url)
        missing = (
            (provider == "openai" and not self.openai_api_key)
            or (provider == "anthropic" and not self.anthropic_api_key)
            or (provider == "nvidia" and not self.nvidia_api_key
                and "api.nvidia.com" in nvidia_url)
        )
        if missing:
            return (f"{label} is selected for {purpose} but no {label} API key is set. "
                    "Open Settings (the gear icon, top right) to add one.")
        return None

    def problems(self) -> list[str]:
        """Anything that prevents answering, phrased for the person reading it.

        Reported rather than raised: a container that exits on a missing API key
        restarts forever and shows nothing useful. Better to come up, serve the
        interface, and say what to fix.
        """
        issues: list[str] = []
        if self.embed_provider not in EMBED_PROVIDERS:
            issues.append(f"Unknown embedding provider '{self.embed_provider}'. "
                          "Choose one in Settings.")
        else:
            problem = self.key_problem(self.embed_provider, "search")
            if problem:
                issues.append(problem)
        if self.llm_provider not in LLM_PROVIDERS:
            issues.append(f"Unknown answer provider '{self.llm_provider}'. "
                          "Choose one in Settings.")
        else:
            problem = self.key_problem(self.llm_provider, "answers")
            if problem and problem.split(" is selected")[0] not in " ".join(issues):
                issues.append(problem)
        if not self.chat_model:
            issues.append("No answer model is selected. Choose one in Settings.")
        if not self.embed_model:
            issues.append("No search (embedding) model is selected. Choose one in Settings.")
        return issues


def load(overrides: dict | None = None) -> Settings:
    """Build settings from the environment, with saved panel values on top."""
    _clear_blank_base_url()
    saved = {k: (v or "").strip() for k, v in (overrides or {}).items()}

    def pick(key: str, default: str = "") -> str:
        return saved.get(key) or _text(EDITABLE[key], default)

    llm_provider = pick("llm_provider", "openai").lower()
    models = {p: pick(f"{p}_chat_model", DEFAULT_MODELS[p]) for p in LLM_PROVIDERS}
    fast_models = {p: pick(f"{p}_fast_model") for p in LLM_PROVIDERS}
    embed_provider = pick("embed_provider", "openai").lower()
    embed_model = pick("embed_model", DEFAULT_EMBED_MODELS.get(embed_provider, ""))

    # Measured widths are saved alongside the model they were measured for.
    # Before any measurement, the original default (text-embedding-3-large
    # shortened to 1536) still describes an existing index exactly.
    if saved.get("embed_dim"):
        embed_dim = int(saved["embed_dim"])
        send_dimensions = saved.get("embed_send_dimensions") == "true"
    else:
        embed_dim = _number("EMBED_DIM", DEFAULT_DIMENSIONS)
        send_dimensions = embed_model.startswith("text-embedding-3")
    nvidia_base = pick("nvidia_base_url", NVIDIA_DEFAULT_BASE_URL)

    return Settings(
        database_url=_text(
            "DATABASE_URL", "postgresql://aidoctor:aidoctor@db:5432/aidoctor"
        ),
        document_root=Path(_text("DOCUMENT_ROOT", "/data/documents")),
        llm_provider=llm_provider,
        chat_model=models.get(llm_provider, ""),
        fast_model=fast_models.get(llm_provider, ""),
        fast_models=fast_models,
        embed_provider=embed_provider,
        embed_model=embed_model,
        embed_dim=embed_dim,
        embed_send_dimensions=send_dimensions,
        openai_api_key=pick("openai_api_key"),
        openai_base_url=pick("openai_base_url"),
        openai_chat_model=models["openai"],
        anthropic_api_key=pick("anthropic_api_key"),
        anthropic_chat_model=models["anthropic"],
        nvidia_api_key=pick("nvidia_api_key"),
        nvidia_base_url=nvidia_base,
        nvidia_embed_base_url=pick("nvidia_embed_base_url", nvidia_base),
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
        # Most passages kept after grading, before widening and merging.
        passages=_number("PASSAGES", 12),
        # Retrieved wide, then graded by the model: recall first, precision second.
        shortlist=_number("SHORTLIST", 30),
        # Ceiling on the evidence sent to the writer and checker, in tokens.
        evidence_tokens=_number("EVIDENCE_TOKENS", 7000),
        use_hypothesis=_text("USE_HYPOTHESIS", "true").lower() in ("1", "true", "yes"),
        enable_review=_text("ENABLE_REVIEW", "true").lower() in ("1", "true", "yes"),
        enable_suggestions=_text("ENABLE_SUGGESTIONS", "true").lower()
        in ("1", "true", "yes"),
    )
