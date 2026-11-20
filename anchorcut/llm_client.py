"""Swappable model wrapper.

The pipeline only calls `LLMClient.ask_json(prompt, Schema, tag)`. Which model answers is decided by
.env, so a new model or a different gateway can be plugged in during an interview without code changes:

    LLM_PROVIDER=gemini   LLM_MODEL=gemini-3.5-flash         LLM_API_KEY=...   (Google AI Studio)
    LLM_PROVIDER=openai   LLM_MODEL=anthropic/claude-x       LLM_API_KEY=...   LLM_BASE_URL=https://openrouter.ai/api/v1
    LLM_PROVIDER=openai   LLM_MODEL=<any>                    LLM_BASE_URL=<any OpenAI-compatible endpoint>

Handled here once, for every provider:
  * a simple requests-per-minute throttle
  * retries with backoff on 429 / 5xx / network errors, honouring the server's retry delay
  * a long 429 (daily quota) moves on to the next model in LLM_FALLBACK_MODELS, or pauses the run
  * JSON extraction + Pydantic validation with one corrective re-ask
  * every attempt saved UNEDITED to raw_outputs/ (prompt, raw response text, model, time, error)
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Type, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from .config import Settings

log = logging.getLogger("anchorcut.llm")
T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """A call failed in a way that only affects this unit of work."""


class QuotaPause(LLMError):
    """Every configured model is out of quota / unreachable. The run pauses; `resume` continues later."""


class _RetryableQuota(Exception):
    """This model cannot serve us for the rest of the run (daily quota gone, model not available)."""

    def __init__(self, wait_s: float, msg: str):
        super().__init__(msg)
        self.wait_s = wait_s


class _Overloaded(Exception):
    """5xx / timeouts kept happening for THIS call. Temporary: try the next model for this call only,
    and if none can serve it, fail just this unit (resume retries it) instead of pausing the run."""


class LLMClient:
    def __init__(self, settings: Settings, raw_dir: Path | None = None):
        self.s = settings
        self.raw_dir = raw_dir
        self._lock = threading.Lock()
        self._last = 0.0
        self._dead: set[str] = set()      # models that ran out of quota during this run
        self._count_lock = threading.Lock()
        self._local = threading.local()   # per-thread: which model answered this thread's last call
        self.calls = 0

    @property
    def last_model(self) -> str:
        """Model that served the calling thread's most recent request (safe with parallel workers)."""
        return getattr(self._local, "model", "")

    # ------------------------------------------------------------------ transport
    def _post(self, model: str, prompt: str) -> str:
        if not self.s.api_key:
            raise LLMError("No API key configured: " + "; ".join(self.s.problems()))
        if self.s.provider == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            r = httpx.post(url, headers={"x-goog-api-key": self.s.api_key},
                           json={"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                                 "generationConfig": {"temperature": self.s.temperature,
                                                      "responseMimeType": "application/json"}},
                           timeout=self.s.timeout_s)
            r.raise_for_status()
            data = r.json()
            try:
                return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
            except (KeyError, IndexError, TypeError) as e:
                raise LLMError(f"empty or blocked Gemini response: {str(data)[:300]}") from e
        if self.s.provider in ("openai", "openrouter", "groq"):
            r = httpx.post(f"{self.s.endpoint}/chat/completions", headers={"Authorization": f"Bearer {self.s.api_key}"},
                           json={"model": model, "temperature": self.s.temperature,
                                 "response_format": {"type": "json_object"},
                                 "messages": [{"role": "user", "content": prompt}]},
                           timeout=self.s.timeout_s)
            r.raise_for_status()
            try:
                return r.json()["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError, TypeError) as e:
                raise LLMError(f"unexpected response shape: {r.text[:300]}") from e
        raise LLMError(f"unknown LLM_PROVIDER '{self.s.provider}' (use gemini, openrouter, groq or openai)")

    def _throttle(self) -> None:
        gap = 60.0 / max(self.s.requests_per_minute, 0.1)
        with self._lock:
            wait = self._last + gap - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()

    def _call(self, model: str, prompt: str) -> str:
        delay = 4.0
        for attempt in range(1, self.s.max_retries + 1):
            self._throttle()
            with self._count_lock:
                self.calls += 1
            try:
                return self._post(model, prompt)
            except httpx.HTTPStatusError as e:
                code, body = e.response.status_code, e.response.text
                if code == 429:
                    m = re.search(r'"retryDelay":\s*"(\d+)', body)
                    ra = e.response.headers.get("retry-after")
                    wait = float(m.group(1)) if m else float(ra) if ra and ra.replace(".", "").isdigit() else delay
                    if wait > 120:
                        raise _RetryableQuota(wait, f"{model}: quota exhausted (retry in {wait / 3600:.1f} h)")
                elif code >= 500:
                    wait = delay
                elif code in (401, 403):
                    raise LLMError(f"{model}: HTTP {code} - the API key was rejected") from e
                elif code == 404:
                    raise _RetryableQuota(0, f"{model}: HTTP 404 - model not available for this key")
                else:
                    raise LLMError(f"{model}: HTTP {code}: {body[:300]}") from e
                log.warning("%s HTTP %s (attempt %d/%d) - waiting %.0fs", model, code, attempt, self.s.max_retries, wait)
                time.sleep(min(wait, 90))
                delay *= 2
            except (httpx.TimeoutException, httpx.TransportError) as e:
                log.warning("%s network error (attempt %d): %s", model, attempt, e)
                time.sleep(delay)
                delay *= 2
        raise _Overloaded(f"{model}: overloaded / unreachable after {self.s.max_retries} attempts")

    # ------------------------------------------------------------------ raw log
    def _save_raw(self, tag: str, model: str, attempt: int, prompt: str, text: str, error: str | None) -> None:
        if not self.raw_dir:
            return
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        rec = {"call": tag, "attempt": attempt, "provider": self.s.provider, "model": model,
               "temperature": self.s.temperature, "time_utc": datetime.now(timezone.utc).isoformat(),
               "prompt_sha1": hashlib.sha1(prompt.encode("utf-8")).hexdigest(),
               "parse_error": error, "prompt": prompt, "raw_response": text}
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag)
        (self.raw_dir / f"{safe}__{attempt}.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _json_text(text: str) -> str:
        t = text.strip()
        fence = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
        if fence:
            t = fence.group(1).strip()
        if not t.startswith("{"):
            i = t.find("{")
            if i >= 0:
                t = t[i:]
        return t

    # ------------------------------------------------------------------ public
    def ask_json(self, prompt: str, schema: Type[T], tag: str) -> T:
        notes: list[str] = []
        for model in self.s.model_chain:
            if model in self._dead:
                continue
            ask = prompt
            try:
                for attempt in (1, 2):
                    text = self._call(model, ask)
                    try:
                        obj = schema.model_validate_json(self._json_text(text))
                    except (ValidationError, ValueError) as e:
                        err = str(e)[:1200]
                        self._save_raw(tag, model, attempt, ask, text, err)
                        ask = (prompt + "\n\nYour previous reply could not be parsed:\n" + err +
                               "\nReply again with ONE JSON object matching the requested shape. No prose.")
                        continue
                    self._save_raw(tag, model, attempt, ask, text, None)
                    self._local.model = f"{self.s.provider}:{model}"
                    return obj
                # the model answered but the answer is unusable: try the next model for this call only
                notes.append(f"{model}: invalid JSON twice")
            except _RetryableQuota as e:
                log.warning("[%s] %s -> trying next model", tag, e)
                self._dead.add(model)
                notes.append(str(e))
            except _Overloaded as e:
                log.warning("[%s] %s -> trying next model for this call", tag, e)
                notes.append(f"temporary: {e}")
        if any("invalid JSON" in n or n.startswith("temporary") for n in notes):
            # at least one model is still alive: fail only this unit; the run goes on and resume retries it
            raise LLMError(f"[{tag}] " + "; ".join(notes))
        raise QuotaPause("All configured models are out of quota or unavailable: " + "; ".join(notes or ["none left"]))
