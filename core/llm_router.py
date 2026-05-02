# ================================================================
#  core/llm_router.py — Multi-Provider LLM Router
#
#  Fallback chain:
#    1. OpenRouter (primary — free tier, Llama-3.3-70B)
#    2. Groq       (secondary — llama-3.3-70b-versatile)
#    3. Gemini     (tertiary — gemini-2.0-flash-lite, 1500 req/day free)
#    4. Ollama     (final — local, unlimited, guaranteed)
#
#  All providers expose the same interface:
#    router.complete(messages, temp, max_tokens) → str
#    router.stream(messages, max_tokens, temp)   → generator[str]
#
#  v3 changes:
#    - STICKY PROVIDER: once a provider succeeds, all subsequent calls
#      in the same session reuse it. Only falls back when it fails
#      with a rate-limit after retries are exhausted.
#    - ALL logging goes to sys.stderr (never stdout).
#      This prevents [WAIT]/[WARN] messages leaking into the Streamlit UI
#      which captures sys.stdout for streaming tokens.
#    - MAX_RETRIES_PER_CALL reduced to 2 (was 3) to fail faster to fallback.
# ================================================================

import json
import re
import sys
import time
import math
from typing import Generator


# ── Logging — always stderr, never stdout ────────────────────────
def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)


# ── Retry configuration ──────────────────────────────────────────
MAX_RETRIES_PER_CALL = 2      # retries before giving up on provider for this call
DEFAULT_RETRY_WAIT   = 10     # seconds if wait time unparseable
MAX_RETRY_WAIT       = 60     # cap on any single wait


def _is_rate_limit(err: str) -> bool:
    s = err.lower()
    return any(k in s for k in (
        "429", "rate_limit", "rate limit", "quota", "too many requests",
        "resource_exhausted", "daily limit", "requests per minute",
        "tokens per", "per day",
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
        return min(int(math.ceil(float(m.group(1)))) + 2, MAX_RETRY_WAIT)
    m = re.search(r'in\s+(\d+)m(?:(\d+)s)?', s)
    if m:
        return min(int(m.group(1)) * 60 + int(m.group(2) or 0) + 3, MAX_RETRY_WAIT)
    m = re.search(r'(?:retry|after)\s+(\d+)', s)
    if m:
        return min(int(m.group(1)) + 3, MAX_RETRY_WAIT)
    return DEFAULT_RETRY_WAIT


# ── Provider call functions ──────────────────────────────────────

def _call_openrouter(
    messages: list,
    api_key: str,
    model: str,
    max_tokens: int,
    temperature: float,
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
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(f"OpenRouter HTTP {e.code}: {body[:500]}") from e
    if "error" in data:
        raise RuntimeError(str(data["error"]))
    return data["choices"][0]["message"]["content"].strip()


def _call_groq(
    messages: list,
    api_key: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    from groq import Groq
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


def _call_groq_stream(
    messages: list,
    api_key: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> Generator[str, None, None]:
    from groq import Groq
    client = Groq(api_key=api_key)
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


def _call_gemini(
    messages: list,
    api_key: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    import urllib.request
    import urllib.error

    contents = []
    system_parts = []
    for m in messages:
        role = m["role"]
        text = m["content"]
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

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_text = ""
        try:
            body_text = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(f"Gemini HTTP {e.code}: {body_text[:500]}") from e
    if "error" in data:
        raise RuntimeError(str(data["error"]))
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def _call_ollama(
    messages: list,
    model: str,
    max_tokens: int,
    temperature: float,
    host: str = "http://localhost:11434",
) -> str:
    import urllib.request
    payload = json.dumps({
        "model":    model,
        "messages": messages,
        "stream":   False,
        "options":  {"num_predict": max_tokens, "temperature": temperature},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["message"]["content"].strip()


def _call_ollama_stream(
    messages: list,
    model: str,
    max_tokens: int,
    temperature: float,
    host: str = "http://localhost:11434",
) -> Generator[str, None, None]:
    import urllib.request
    payload = json.dumps({
        "model":    model,
        "messages": messages,
        "stream":   True,
        "options":  {"num_predict": max_tokens, "temperature": temperature},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
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
    Drop-in replacement for direct Groq calls in CareAI.

    Sticky-provider behaviour:
      - Tracks which provider last succeeded (_sticky).
      - On each call, tries _sticky first (skipping the top of the chain).
      - If _sticky fails with rate-limit after retries, clears _sticky and
        falls through to the next available provider.
      - All debug/warn messages go to stderr — never stdout.

    Usage (in careai.py):
        self.router = LLMRouter(config)
        text = self.router.complete(messages, temp=0.2, max_tokens=900)
        for token in self.router.stream(messages, max_tokens=250):
            yield token
    """

    def __init__(self, config: dict):
        self.cfg = config

        self._or_model     = config.get("openrouter_model", "meta-llama/llama-3.3-70b-instruct:free")
        self._groq_model   = config.get("groq_model",       "llama-3.3-70b-versatile")
        self._gemini_model = config.get("gemini_model",     "gemini-2.0-flash-lite")
        self._ollama_model = config.get("ollama_model",     "qwen2.5:7b")
        self._ollama_host  = config.get("ollama_host",      "http://localhost:11434")

        # Pre-validate Groq package
        try:
            from groq import Groq as _GroqCheck  # noqa: F401
            self._groq_available = True
        except ImportError:
            self._groq_available = False
            _log("[LLMRouter] WARNING: 'groq' package not installed. Run: pip install groq")

        # Per-session availability (permanently disabled on daily quota exhaustion)
        self._status = {
            "openrouter": bool(config.get("openrouter_api_key")),
            "groq":       bool(config.get("groq_api_key")) and self._groq_available,
            "gemini":     bool(config.get("gemini_api_key")),
            "ollama":     True,
        }

        # Sticky provider — last one that succeeded
        # Seeded from config["provider"] if set, else None (auto-discover on first call)
        preferred = config.get("provider")
        self._sticky: str | None = preferred if (preferred and self._status.get(preferred)) else None

        active = [p for p, v in self._status.items() if v]
        _log(f"[LLMRouter] active={active} sticky={self._sticky or 'auto'}")

    def _disable(self, provider: str, reason: str):
        """Permanently disable provider for this session (daily quota only)."""
        self._status[provider] = False
        if self._sticky == provider:
            self._sticky = None
        _log(f"[LLMRouter] {provider} disabled (daily quota): {reason[:120]}")

    def _call_with_retry(self, provider: str, call_fn, *args, **kwargs) -> str:
        """
        Call a provider with retry on rate-limit.
        - Retries up to MAX_RETRIES_PER_CALL times.
        - On daily quota: permanently disables provider and raises.
        - On rate-limit after all retries: raises (caller tries next provider).
        - On non-rate-limit error: raises immediately.
        """
        last_err = None
        for attempt in range(MAX_RETRIES_PER_CALL):
            try:
                return call_fn(*args, **kwargs)
            except Exception as e:
                last_err = e
                err = str(e)
                if _is_daily_quota(err):
                    self._disable(provider, err)
                    raise
                if _is_rate_limit(err):
                    if attempt < MAX_RETRIES_PER_CALL - 1:
                        wait = min(_parse_wait(err) * (1 + attempt * 0.5), MAX_RETRY_WAIT)
                        _log(f"[LLMRouter] {provider} rate limit, retrying in {wait:.0f}s "
                             f"(attempt {attempt+1}/{MAX_RETRIES_PER_CALL})")
                        time.sleep(wait)
                        continue
                    else:
                        _log(f"[LLMRouter] {provider} rate limit after {MAX_RETRIES_PER_CALL} retries — skipping")
                        raise
                else:
                    raise
        raise last_err

    def _provider_chain(self) -> list[tuple[str, callable, tuple]]:
        """
        Return ordered list of (provider_name, call_fn, args) to try.
        Sticky provider goes first if set and still available.
        """
        def _entry(name):
            if name == "openrouter" and self._status["openrouter"] and self.cfg.get("openrouter_api_key"):
                return (name, _call_openrouter, (
                    None,  # messages placeholder
                    self.cfg["openrouter_api_key"],
                    self._or_model,
                ))
            if name == "groq" and self._status["groq"] and self.cfg.get("groq_api_key") and self._groq_available:
                return (name, _call_groq, (
                    None,
                    self.cfg["groq_api_key"],
                    self._groq_model,
                ))
            if name == "gemini" and self._status["gemini"] and self.cfg.get("gemini_api_key"):
                return (name, _call_gemini, (
                    None,
                    self.cfg["gemini_api_key"],
                    self._gemini_model,
                ))
            if name == "ollama":
                return (name, _call_ollama, (
                    None,
                    self._ollama_model,
                ))
            return None

        order = ["openrouter", "groq", "gemini", "ollama"]

        # Put sticky first
        if self._sticky and self._sticky in order:
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
                errors.append(f"{provider}: {err[:200]}")
                # Clear sticky if it just failed
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
        Streaming variant. Sticky provider first.
        Falls back gracefully to complete() for non-streaming providers.
        """
        errors = []

        # Build stream-capable order (same sticky logic)
        order = ["openrouter", "groq", "gemini", "ollama"]
        if self._sticky and self._sticky in order:
            order = [self._sticky] + [p for p in order if p != self._sticky]

        for provider in order:
            if not self._status.get(provider):
                continue

            try:
                if provider == "openrouter" and self.cfg.get("openrouter_api_key"):
                    result = self._call_with_retry(
                        provider, _call_openrouter,
                        messages, self.cfg["openrouter_api_key"],
                        self._or_model, max_tokens, temp,
                    )
                    if self._sticky != provider:
                        _log(f"[LLMRouter] stream: switched to {provider}")
                        self._sticky = provider
                    yield result
                    return

                elif provider == "groq" and self.cfg.get("groq_api_key") and self._groq_available:
                    # Native Groq streaming — no retry wrapper (stream can't retry mid-stream)
                    if self._sticky != provider:
                        _log(f"[LLMRouter] stream: switched to {provider}")
                        self._sticky = provider
                    yield from _call_groq_stream(
                        messages, self.cfg["groq_api_key"],
                        self._groq_model, max_tokens, temp,
                    )
                    return

                elif provider == "gemini" and self.cfg.get("gemini_api_key"):
                    result = self._call_with_retry(
                        provider, _call_gemini,
                        messages, self.cfg["gemini_api_key"],
                        self._gemini_model, max_tokens, temp,
                    )
                    if self._sticky != provider:
                        _log(f"[LLMRouter] stream: switched to {provider}")
                        self._sticky = provider
                    yield result
                    return

                elif provider == "ollama":
                    if self._sticky != provider:
                        _log(f"[LLMRouter] stream: switched to {provider}")
                        self._sticky = provider
                    yield from _call_ollama_stream(
                        messages, self._ollama_model,
                        max_tokens, temp, self._ollama_host,
                    )
                    return

            except Exception as e:
                err = str(e)
                errors.append(f"{provider}: {err[:200]}")
                if self._sticky == provider:
                    _log(f"[LLMRouter] stream: {provider} failed, clearing sticky")
                    self._sticky = None
                continue

        _log(f"[LLMRouter] stream: ALL providers failed: {' | '.join(errors)}")
        yield (
            "\n[ERROR] All LLM providers are currently unavailable. "
            "Please check your API keys and try again."
        )

    def status(self) -> dict:
        """Return current provider availability."""
        return {**self._status, "sticky": self._sticky}

    def reset_providers(self):
        """Re-enable all providers (e.g. new session / daily reset)."""
        self._status = {
            "openrouter": bool(self.cfg.get("openrouter_api_key")),
            "groq":       bool(self.cfg.get("groq_api_key")) and self._groq_available,
            "gemini":     bool(self.cfg.get("gemini_api_key")),
            "ollama":     True,
        }
        self._sticky = self.cfg.get("provider") or None
        _log("[LLMRouter] all providers re-enabled.")