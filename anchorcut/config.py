"""Settings from environment variables (.env locally, Streamlit secrets when deployed).

Swapping the model never needs a code change. Three ways, highest priority first:
  1. the "Model & API key" panel in the app (this browser session only, never saved or shown again)
  2. real environment variables / Streamlit Cloud secrets
  3. the project's .env file (re-read on every page load, so edits apply without a restart)

Secrets are only ever read here and never written to logs, raw outputs or the UI.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# provider -> (default model, default base URL, env vars that may hold its key, in order)
PROVIDERS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "gemini": ("gemini-3.5-flash", "", ("GEMINI_API_KEY", "GOOGLE_API_KEY")),
    "openrouter": ("google/gemini-2.5-flash", "https://openrouter.ai/api/v1", ("OPENROUTER_API_KEY",)),
    "groq": ("openai/gpt-oss-120b", "https://api.groq.com/openai/v1", ("GROQ_API_KEY",)),
    "openai": ("", "https://api.openai.com/v1", ("OPENAI_API_KEY",)),   # any OpenAI-compatible endpoint
}

_from_dotenv: set[str] = set()   # variables that came from .env (safe to refresh when the file changes)


def _load_env_file() -> None:
    """Read <project>/.env. Variables that came from .env are refreshed on every call, so changing a key or
    model in .env takes effect on the next page load. Variables set outside .env (shell, Streamlit secrets)
    are never overridden."""
    try:
        from dotenv import dotenv_values
    except ImportError:  # optional dependency
        return
    values = dotenv_values(ROOT / ".env") if (ROOT / ".env").exists() else {}
    for k in list(_from_dotenv):
        if k not in values or values[k] is None:      # removed from .env -> forget it
            os.environ.pop(k, None)
            _from_dotenv.discard(k)
    for k, v in values.items():
        if v is None:
            continue
        if k in _from_dotenv or not os.getenv(k):
            os.environ[k] = v
            _from_dotenv.add(k)


_load_env_file()


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _provider() -> str:
    p = _env("LLM_PROVIDER", "gemini").lower()
    return p if p in PROVIDERS else p   # unknown names are reported by the client, not silently replaced


def _key_for(provider: str) -> str:
    """LLM_API_KEY wins; otherwise the provider's own key variable. Never another provider's key."""
    if _env("LLM_API_KEY"):
        return _env("LLM_API_KEY")
    for name in PROVIDERS.get(provider, ("", "", ()))[2]:
        if _env(name):
            return _env(name)
    return ""


@dataclass
class Settings:
    # ---- model (swap without touching code)
    provider: str = field(default_factory=_provider)                     # gemini | openrouter | groq | openai
    model: str = field(default_factory=lambda: _env("LLM_MODEL") or PROVIDERS.get(_provider(), ("",))[0])
    fallback_models: str = field(default_factory=lambda: _env("LLM_FALLBACK_MODELS", ""))  # comma list, same provider
    api_key: str = field(default_factory=lambda: _key_for(_provider()), repr=False)
    base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL"))  # blank = provider default
    temperature: float = field(default_factory=lambda: float(_env("LLM_TEMPERATURE", "0.2")))
    requests_per_minute: float = field(default_factory=lambda: float(_env("LLM_RPM", "8")))
    max_retries: int = field(default_factory=lambda: int(_env("LLM_MAX_RETRIES", "4")))
    timeout_s: float = field(default_factory=lambda: float(_env("LLM_TIMEOUT_S", "180")))
    concurrency: int = field(default_factory=lambda: max(1, int(_env("LLM_CONCURRENCY", "4"))))  # parallel calls
    # ---- pipeline knobs
    max_hooks_per_episode: int = field(default_factory=lambda: int(_env("MAX_HOOKS_PER_EPISODE", "6")))
    max_cliffs_per_episode: int = field(default_factory=lambda: int(_env("MAX_CLIFFS_PER_EPISODE", "6")))
    candidates_per_episode: int = field(default_factory=lambda: int(_env("CANDIDATES_PER_EPISODE", "3")))
    critic_batch_size: int = field(default_factory=lambda: int(_env("CRITIC_BATCH_SIZE", "5")))
    workspace: Path = field(default_factory=lambda: Path(_env("ANCHORCUT_WORKSPACE", str(ROOT / "workspace"))))
    key_source: str = field(default="environment / .env", repr=False)

    @property
    def model_chain(self) -> list[str]:
        extra = [m.strip() for m in self.fallback_models.split(",") if m.strip()]
        return [m for m in [self.model] + [x for x in extra if x != self.model] if m]

    @property
    def endpoint(self) -> str:
        return (self.base_url or PROVIDERS.get(self.provider, ("", "", ()))[1]).rstrip("/")

    def with_overrides(self, provider: str = "", model: str = "", fallback_models: str | None = None,
                       base_url: str | None = None, api_key: str = "") -> "Settings":
        """A copy with session overrides from the UI. Blank fields keep the configured value; switching
        provider without a key looks up that provider's own key variable."""
        s = replace(self)
        if provider and provider != s.provider:
            s.provider = provider
            s.api_key = _key_for(provider)
            s.model = PROVIDERS.get(provider, ("",))[0]
            s.fallback_models = ""
            s.base_url = ""
        if model:
            s.model = model.strip()
        if fallback_models is not None:
            s.fallback_models = fallback_models
        if base_url is not None and base_url.strip():
            s.base_url = base_url.strip()
        if api_key:
            s.api_key = api_key.strip()
            s.key_source = "entered in the app (this session only)"
        return s

    def problems(self) -> list[str]:
        """Human-readable reasons this configuration cannot run (empty list = ready)."""
        out = []
        if self.provider not in PROVIDERS:
            out.append(f"unknown provider '{self.provider}' (use one of: {', '.join(PROVIDERS)})")
        if not self.model_chain:
            out.append("no model name set (LLM_MODEL)")
        if not self.api_key:
            names = " or ".join(("LLM_API_KEY",) + PROVIDERS.get(self.provider, ("", "", ()))[2])
            out.append(f"no API key for {self.provider} (set {names})")
        return out

    def public(self) -> dict:
        """Safe-to-display view (never the key itself)."""
        return {"provider": self.provider, "model_chain": self.model_chain, "endpoint": self.endpoint or "(default)",
                "api_key_set": bool(self.api_key), "key_source": self.key_source if self.api_key else "-",
                "temperature": self.temperature, "rpm": self.requests_per_minute, "concurrency": self.concurrency,
                "candidates_per_episode": self.candidates_per_episode, "critic_batch_size": self.critic_batch_size}


def get_settings() -> Settings:
    _load_env_file()
    return Settings()
