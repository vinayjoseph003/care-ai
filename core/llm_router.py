# ================================================================
#  core/llm_router.py — Multi-Provider LLM Router (Phase 2)
#
#  Fallback chain:
#    1. OpenRouter (primary — verified meta-llama/llama-3.3-70b-instruct)
#    2. Gemini     (fallback — verified gemini-3.5-flash-lite)
#    3. Groq       (tertiary — llama-3.3-70b-versatile, if key valid)
#    4. Ollama     (local fallback — only if actively running and reachable)
#
#  Key Features:
#    - Instant fallback: switches immediately to next provider when
#      a provider hits rate limits or errors, instead of long blocking sleeps.
#    - Secure auth: passes Gemini key via x-goog-api-key header (never in URLs/logs).
#    - Controlled Gemini parsing: safely handles missing candidates, empty parts,
#      and blocked safety filters without unhandled index/key exceptions.
#    - Controlled OpenRouter parsing: verifies choices and non-empty content.
#    - Dynamic Ollama probe: fast bounded check; never hangs in hosted setups.
#    - Full streaming: native SSE chunk streaming for both OpenRouter and Gemini.
# ================================================================

import os
import json
import re
import sys
import time
import math
from typing import Generator


# ── Logging — always stderr, never stdout ────────────────────────
def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)


# ── Retry & Timeout Configuration ────────────────────────────────
MAX_RETRIES_PER_CALL = 2      # Retries only if no healthy fallback provider is available
DEFAULT_RETRY_WAIT   = 3      # Shorter wait before retrying last provider
MAX_RETRY_WAIT       = 10     # Cap on any single wait
DEFAULT_TIMEOUT      = 30     # Bounded HTTP timeout in seconds


# ── Known Model Migrations (auto-map deprecated/retired models) ───
_MODEL_MIGRATIONS = {
    "meta-llama/llama-3.3-70b-instruct:free": "meta-llama/llama-3.3-70b-instruct",
    "gemini-1.5-flash":             "gemini-3.5-flash-lite",
    "gemini-1.5-flash-latest":      "gemini-3.5-flash-lite",
    "gemini-2.0-flash":             "gemini-3.5-flash-lite",
    "gemini-2.0-flash-lite":        "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite-preview": "gemini-3.5-flash-lite",
}


def _is_rate_limit(err: str) -> bool:
    s = err.lower()
    return any(k in s for k in (
        "429", "rate_limit", "rate limit", "quota", "too many requests",
        "resource_exhausted", "daily limit", "requests per minute",
        "tokens per", "per day", "503", "high demand", "temporarily unavailable",
    ))


def _is_auth_error(err: str) -> bool:
    s = err.lower()
    return any(k in s for k in (
        "401", "invalid_api_key", "invalid api key", "unauthorized",
        "api key not valid", "forbidden", "403", "permission_denied",
    ))


def _is_daily_quota(err: str) -> bool:
    s = err.lower()
    return any(k in s for k in (
        "daily limit", "daily quota", "per day", "requests per day",
        "rate limit reached for model",
    ))


def _parse_wait(err: str) -> int:
    s = err.lower()
    m = re.search(r'in\s+([\d.]+)\s*s', s)
    if m:
        return min(int(math.ceil(float(m.group(1)))) + 1, MAX_RETRY_WAIT)
    m = re.search(r'in\s+(\d+)m(?:(\d+)s)?', s)
    if m:
        return min(int(m.group(1)) * 60 + int(m.group(2) or 0) + 1, MAX_RETRY_WAIT)
    m = re.search(r'(?:retry|after)\s+(\d+)', s)
    if m:
        return min(int(m.group(1)) + 1, MAX_RETRY_WAIT)
    return DEFAULT_RETRY_WAIT


def _is_ollama_online(host: str, timeout: float = 0.8) -> bool:
    """Fast check whether local Ollama server is running and responsive."""
    import urllib.request
    try:
        url = f"{host.rstrip('/')}/api/tags"
        req = urllib.request.Request(url, headers={"User-Agent": "Care-AI/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


# ── Provider call functions ──────────────────────────────────────

def _call_openrouter(
    messages: list,
    api_key: str,
    model: str,
    max_tokens: int,
    temperature: float,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    import urllib.request
    import urllib.error
    payload = json.dumps({
        "model":       model,
        "messages":    messages,
        "max_tokens":  max_tokens,
        "temperature": temperature,
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  "https://care-ai.local",
            "X-Title":       "Care-AI",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(f"OpenRouter HTTP {e.code}: {body[:500]}") from e
    except Exception as e:
        raise RuntimeError(f"OpenRouter request error: {type(e).__name__}: {str(e)[:200]}") from e

    if "error" in data:
        err_info = data["error"]
        msg = err_info.get("message", str(err_info)) if isinstance(err_info, dict) else str(err_info)
        raise RuntimeError(f"OpenRouter API error: {msg}")

    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError("OpenRouter returned no choices in response")

    msg = choices[0].get("message", {})
    content = msg.get("content", "")
    if content is None:
        content = ""
    content = content.strip()
    if not content:
        raise RuntimeError("OpenRouter returned empty message content")

    return content


def _call_openrouter_stream(
    messages: list,
    api_key: str,
    model: str,
    max_tokens: int,
    temperature: float,
    timeout: int = DEFAULT_TIMEOUT,
) -> Generator[str, None, None]:
    import urllib.request
    import urllib.error
    payload = json.dumps({
        "model":       model,
        "messages":    messages,
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "stream":      True,
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  "https://care-ai.local",
            "X-Title":       "Care-AI",
        },
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(f"OpenRouter stream HTTP {e.code}: {body[:500]}") from e
    except Exception as e:
        raise RuntimeError(f"OpenRouter stream connection failed: {e}") from e

    with resp:
        for line in resp:
            line_str = line.decode("utf-8", errors="replace").strip()
            if not line_str or not line_str.startswith("data: "):
                continue
            if line_str == "data: [DONE]":
                break
            try:
                data = json.loads(line_str[6:])
                choices = data.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        yield token
            except Exception:
                continue


def _call_gemini(
    messages: list,
    api_key: str,
    model: str,
    max_tokens: int,
    temperature: float,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    import urllib.request
    import urllib.error

    contents = []
    system_parts = []
    for m in messages:
        role = m["role"]
        text = m.get("content", "")
        if role == "system":
            system_parts.append({"text": text})
        elif role == "user":
            contents.append({"role": "user",  "parts": [{"text": text}]})
        elif role == "assistant":
            contents.append({"role": "model", "parts": [{"text": text}]})

    body: dict = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature":     temperature,
        },
    }
    if system_parts:
        body["systemInstruction"] = {"parts": system_parts}

    # Pass API key via header to ensure credentials never appear in URLs, error messages, or logs
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_text = ""
        try:
            body_text = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(f"Gemini HTTP {e.code}: {body_text[:500]}") from e
    except Exception as e:
        raise RuntimeError(f"Gemini request failed: {type(e).__name__}: {str(e)[:200]}") from e

    if "error" in data:
        err_msg = data["error"].get("message", str(data["error"]))
        raise RuntimeError(f"Gemini API error: {err_msg}")

    candidates = data.get("candidates")
    if not candidates:
        feedback = data.get("promptFeedback", {})
        block_reason = feedback.get("blockReason", "no candidates returned")
        raise RuntimeError(f"Gemini returned no candidates (blockReason: {block_reason})")

    first_cand = candidates[0]
    content = first_cand.get("content", {})
    parts = content.get("parts", [])

    text_parts = [
        p["text"] for p in parts
        if isinstance(p, dict) and "text" in p and p["text"]
    ]
    combined_text = "".join(text_parts).strip()

    if not combined_text:
        finish_reason = first_cand.get("finishReason", "UNKNOWN")
        raise RuntimeError(f"Gemini response candidate contained no text content (finishReason: {finish_reason})")

    return combined_text


def _call_gemini_stream(
    messages: list,
    api_key: str,
    model: str,
    max_tokens: int,
    temperature: float,
    timeout: int = DEFAULT_TIMEOUT,
) -> Generator[str, None, None]:
    import urllib.request
    import urllib.error

    contents = []
    system_parts = []
    for m in messages:
        role = m["role"]
        text = m.get("content", "")
        if role == "system":
            system_parts.append({"text": text})
        elif role == "user":
            contents.append({"role": "user",  "parts": [{"text": text}]})
        elif role == "assistant":
            contents.append({"role": "model", "parts": [{"text": text}]})

    body: dict = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature":     temperature,
        },
    }
    if system_parts:
        body["systemInstruction"] = {"parts": system_parts}

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse"
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(f"Gemini stream HTTP {e.code}: {body[:500]}") from e
    except Exception as e:
        raise RuntimeError(f"Gemini stream connection failed: {e}") from e

    with resp:
        for line in resp:
            line_str = line.decode("utf-8", errors="replace").strip()
            if not line_str or not line_str.startswith("data: "):
                continue
            try:
                data = json.loads(line_str[6:])
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    chunk_text = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text"))
                    if chunk_text:
                        yield chunk_text
            except Exception:
                continue


def _call_groq(
    messages: list,
    api_key: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    from groq import Groq
    client = Groq(api_key=api_key, timeout=DEFAULT_TIMEOUT)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    choices = resp.choices
    if not choices or not choices[0].message.content:
        raise RuntimeError("Groq returned empty response")
    return choices[0].message.content.strip()


def _call_groq_stream(
    messages: list,
    api_key: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> Generator[str, None, None]:
    from groq import Groq
    client = Groq(api_key=api_key, timeout=DEFAULT_TIMEOUT)
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content


def _call_ollama(
    messages: list,
    model: str,
    max_tokens: int,
    temperature: float,
    host: str = "http://localhost:11434",
    timeout: int = 25,
) -> str:
    import urllib.request
    payload = json.dumps({
        "model":    model,
        "messages": messages,
        "stream":   False,
        "options":  {"num_predict": max_tokens, "temperature": temperature},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    msg = data.get("message", {}).get("content", "")
    if not msg:
        raise RuntimeError("Ollama returned empty response")
    return msg.strip()


def _call_ollama_stream(
    messages: list,
    model: str,
    max_tokens: int,
    temperature: float,
    host: str = "http://localhost:11434",
    timeout: int = 25,
) -> Generator[str, None, None]:
    import urllib.request
    payload = json.dumps({
        "model":    model,
        "messages": messages,
        "stream":   True,
        "options":  {"num_predict": max_tokens, "temperature": temperature},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for line in resp:
            if not line.strip():
                continue
            try:
                chunk = json.loads(line.decode("utf-8"))
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token
                if chunk.get("done"):
                    break
            except json.JSONDecodeError:
                continue


# ── LLMRouter class ──────────────────────────────────────────────

class LLMRouter:
    """
    Multi-Provider LLM Router with verified priority chain and immediate fallback.

    Priority order:
      1. OpenRouter (primary)
      2. Gemini     (fallback)
      3. Groq       (tertiary)
      4. Ollama     (local fallback, probed dynamically)

    Sticky-provider behaviour:
      - Tracks which provider last succeeded (_sticky).
      - On failure or rate-limit: switches immediately to fallback without long blocking sleeps.
      - All debug/warn messages go to stderr — never stdout.
    """

    def __init__(self, config: dict):
        self.cfg = dict(config or {})

        # Model configuration with automatic deprecated model migrations
        raw_or = self.cfg.get("openrouter_model", "meta-llama/llama-3.3-70b-instruct")
        self._or_model = _MODEL_MIGRATIONS.get(raw_or, raw_or)

        raw_gemini = self.cfg.get("gemini_model", "gemini-3.5-flash-lite")
        self._gemini_model = _MODEL_MIGRATIONS.get(raw_gemini, raw_gemini)

        self._groq_model   = self.cfg.get("groq_model", "llama-3.3-70b-versatile")
        self._ollama_model = self.cfg.get("ollama_model", "qwen2.5:7b")
        self._ollama_host  = self.cfg.get("ollama_host", "http://localhost:11434")

        # Pre-validate Groq library
        try:
            from groq import Groq as _GroqCheck  # noqa: F401
            self._groq_available = True
        except ImportError:
            self._groq_available = False

        # Fast dynamic probe for Ollama (timeout: 0.8s)
        self._ollama_online = _is_ollama_online(self._ollama_host, timeout=0.8)

        # Provider availability status reflects actual runtime state
        self._status = {
            "openrouter": bool(self.cfg.get("openrouter_api_key")),
            "gemini":     bool(self.cfg.get("gemini_api_key")),
            "groq":       bool(self.cfg.get("groq_api_key")) and self._groq_available,
            "ollama":     self._ollama_online,
        }

        # Track verified working status
        self._verified_status = dict(self._status)

        # Configurable primary provider (e.g. LLM_PRIMARY_PROVIDER=gemini)
        raw_primary = (
            self.cfg.get("primary_provider")
            or self.cfg.get("provider")
            or os.environ.get("LLM_PRIMARY_PROVIDER", "")
        ).strip().lower()
        if raw_primary in ("gemini", "google"):
            self._primary_provider = "gemini"
        elif raw_primary in ("openrouter", "llama"):
            self._primary_provider = "openrouter"
        elif raw_primary in ("groq",):
            self._primary_provider = "groq"
        elif raw_primary in ("ollama", "local"):
            self._primary_provider = "ollama"
        else:
            self._primary_provider = "openrouter"

        # Sticky provider — preferred if provided and available, else auto
        preferred = self.cfg.get("provider")
        self._sticky: str | None = preferred if (preferred and self._status.get(preferred)) else None

        active = [p for p, v in self._status.items() if v]
        _log(f"[LLMRouter] initialized: primary={self._primary_provider} active={active} sticky={self._sticky or 'auto'}")

    def _disable(self, provider: str, reason: str):
        """Disable provider for this session (auth failure or daily quota)."""
        self._status[provider] = False
        if self._sticky == provider:
            self._sticky = None
        _log(f"[LLMRouter] {provider} disabled: {reason[:120]}")

    def _has_healthy_fallback(self, current_provider: str) -> bool:
        """Return True if any other provider is currently available."""
        return any(
            self._status.get(p)
            for p in ["openrouter", "gemini", "groq", "ollama"]
            if p != current_provider
        )

    def _call_with_retry(self, provider: str, call_fn, *args, **kwargs) -> str:
        """
        Call a provider with adaptive retry.
        If another healthy fallback is available, switches immediately rather than sleeping.
        """
        last_err = None
        for attempt in range(MAX_RETRIES_PER_CALL):
            try:
                return call_fn(*args, **kwargs)
            except Exception as e:
                last_err = e
                err = str(e)

                # Authentication error (401 / invalid key) -> permanently disable
                if _is_auth_error(err):
                    self._disable(provider, f"Auth failure: {err}")
                    raise

                # Daily quota exhausted -> permanently disable
                if _is_daily_quota(err):
                    self._disable(provider, f"Daily quota: {err}")
                    raise

                # Rate limit or temporary outage (429 / 503)
                if _is_rate_limit(err):
                    # If we have a healthy fallback provider ready, switch immediately!
                    if self._has_healthy_fallback(provider):
                        _log(f"[LLMRouter] {provider} rate limited ({err[:60]}), switching to fallback immediately.")
                        raise
                    elif attempt < MAX_RETRIES_PER_CALL - 1:
                        wait = min(_parse_wait(err), MAX_RETRY_WAIT)
                        _log(f"[LLMRouter] {provider} rate limit (last provider), waiting {wait}s...")
                        time.sleep(wait)
                        continue
                    else:
                        raise
                else:
                    # Generic error -> raise immediately so router tries next provider
                    raise
        raise last_err

    def _provider_chain(self) -> list[tuple[str, callable, tuple]]:
        """
        Return ordered list of (provider_name, call_fn, args).
        Priority: OpenRouter -> Gemini -> Groq -> Ollama.
        Sticky provider goes first if active.
        """
        def _entry(name):
            if name == "openrouter" and self._status["openrouter"] and self.cfg.get("openrouter_api_key"):
                return (name, _call_openrouter, (
                    None,
                    self.cfg["openrouter_api_key"],
                    self._or_model,
                ))
            if name == "gemini" and self._status["gemini"] and self.cfg.get("gemini_api_key"):
                return (name, _call_gemini, (
                    None,
                    self.cfg["gemini_api_key"],
                    self._gemini_model,
                ))
            if name == "groq" and self._status["groq"] and self.cfg.get("groq_api_key") and self._groq_available:
                return (name, _call_groq, (
                    None,
                    self.cfg["groq_api_key"],
                    self._groq_model,
                ))
            if name == "ollama" and self._status["ollama"]:
                return (name, _call_ollama, (
                    None,
                    self._ollama_model,
                ))
            return None

        # Priority chain driven by configured primary provider
        order = list(self.priority)

        # Sticky provider prioritized if set and available
        if self._sticky and self._sticky in order and self._status.get(self._sticky):
            order = [self._sticky] + [p for p in order if p != self._sticky]

        return [e for name in order for e in [_entry(name)] if e is not None]

    # ── Public API ───────────────────────────────────────────────

    def complete(
        self,
        messages:   list,
        temp:       float = 0.2,
        max_tokens: int   = 900,
    ) -> str:
        """
        Try providers in order (sticky first). Returns first success.
        Raises RuntimeError only if ALL providers fail.
        """
        errors = []

        for provider, call_fn, base_args in self._provider_chain():
            try:
                args = (messages,) + base_args[1:] + (max_tokens, temp)
                result = self._call_with_retry(provider, call_fn, *args)
                if self._sticky != provider:
                    _log(f"[LLMRouter] switched to {provider} (now sticky)")
                    self._sticky = provider
                return result
            except Exception as e:
                err = str(e)
                errors.append(f"{provider}: {err[:150]}")
                if self._sticky == provider:
                    _log(f"[LLMRouter] {provider} failed, clearing sticky")
                    self._sticky = None
                continue

        _log(f"[LLMRouter] ALL providers failed: {' | '.join(errors)}")
        raise RuntimeError(
            f"All LLM providers failed.\n{' | '.join(errors)}\n\n"
            "Check your API keys in .env and ensure at least one provider is available."
        )

    def stream(
        self,
        messages:   list,
        max_tokens: int   = 600,
        temp:       float = 0.3,
    ) -> Generator[str, None, None]:
        """
        Streaming variant with native SSE chunk delivery and immediate fallback.
        """
        errors = []
        order = list(self.priority)
        if self._sticky and self._sticky in order and self._status.get(self._sticky):
            order = [self._sticky] + [p for p in order if p != self._sticky]

        for provider in order:
            if not self._status.get(provider):
                continue

            try:
                if provider == "openrouter" and self.cfg.get("openrouter_api_key"):
                    gen = _call_openrouter_stream(
                        messages, self.cfg["openrouter_api_key"],
                        self._or_model, max_tokens, temp,
                    )
                    # Pull first token to verify connection before committing
                    first_token = next(gen)
                    if self._sticky != provider:
                        _log(f"[LLMRouter] stream: switched to {provider}")
                        self._sticky = provider
                    yield first_token
                    yield from gen
                    return

                elif provider == "gemini" and self.cfg.get("gemini_api_key"):
                    gen = _call_gemini_stream(
                        messages, self.cfg["gemini_api_key"],
                        self._gemini_model, max_tokens, temp,
                    )
                    first_token = next(gen)
                    if self._sticky != provider:
                        _log(f"[LLMRouter] stream: switched to {provider}")
                        self._sticky = provider
                    yield first_token
                    yield from gen
                    return

                elif provider == "groq" and self.cfg.get("groq_api_key") and self._groq_available:
                    gen = _call_groq_stream(
                        messages, self.cfg["groq_api_key"],
                        self._groq_model, max_tokens, temp,
                    )
                    first_token = next(gen)
                    if self._sticky != provider:
                        _log(f"[LLMRouter] stream: switched to {provider}")
                        self._sticky = provider
                    yield first_token
                    yield from gen
                    return

                elif provider == "ollama" and self._status["ollama"]:
                    gen = _call_ollama_stream(
                        messages, self._ollama_model,
                        max_tokens, temp, self._ollama_host,
                    )
                    first_token = next(gen)
                    if self._sticky != provider:
                        _log(f"[LLMRouter] stream: switched to {provider}")
                        self._sticky = provider
                    yield first_token
                    yield from gen
                    return

            except Exception as e:
                err = str(e)
                errors.append(f"{provider}: {err[:150]}")
                if _is_auth_error(err):
                    self._disable(provider, f"Auth failure: {err}")
                if self._sticky == provider:
                    _log(f"[LLMRouter] stream: {provider} failed, clearing sticky")
                    self._sticky = None
                continue

        _log(f"[LLMRouter] stream: ALL providers failed: {' | '.join(errors)}")
        yield (
            "\n[ERROR] All LLM providers are currently unavailable. "
            "Please check your API keys and try again."
        )

    @property
    def priority(self) -> list[str]:
        """Return configured provider priority order."""
        if self._primary_provider == "gemini":
            return ["gemini", "openrouter", "groq", "ollama"]
        elif self._primary_provider == "groq":
            return ["groq", "gemini", "openrouter", "ollama"]
        return ["openrouter", "gemini", "groq", "ollama"]

    @property
    def active_provider(self) -> str | None:
        """Return currently active / sticky provider."""
        return self._sticky

    def status(self) -> dict:
        """Return current provider availability."""
        return {
            "openrouter": bool(self._status["openrouter"]),
            "gemini":     bool(self._status["gemini"]),
            "groq":       bool(self._status["groq"]),
            "ollama":     bool(self._status["ollama"]),
            "primary":    self._primary_provider,
            "priority":   self.priority,
            "sticky":     self._sticky,
            "models": {
                "openrouter": self._or_model,
                "gemini":     self._gemini_model,
                "groq":       self._groq_model,
                "ollama":     self._ollama_model,
            }
        }

    def reset_providers(self):
        """Re-enable all providers (e.g. new session / daily reset)."""
        self._ollama_online = _is_ollama_online(self._ollama_host, timeout=0.8)
        self._status = {
            "openrouter": bool(self.cfg.get("openrouter_api_key")),
            "gemini":     bool(self.cfg.get("gemini_api_key")),
            "groq":       bool(self.cfg.get("groq_api_key")) and self._groq_available,
            "ollama":     self._ollama_online,
        }
        self._sticky = self.cfg.get("provider") or None
        _log("[LLMRouter] all providers re-enabled.")