from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Protocol
import requests

ROOT = Path(__file__).resolve().parents[2]


def load_env() -> None:
    """Load .env file into os.environ if not already present."""
    env_path = ROOT / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip()
                    if k not in os.environ:
                        os.environ[k] = v


load_env()


class Provider(Protocol):
    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.0, model: str | None = None) -> str:
        ...


class FakeProvider:
    """Deterministic provider for tests; never contacts a network."""

    def __init__(self, response: str = "fake response"):
        self.response = response
        self.calls = 0

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.0, model: str | None = None) -> str:
        self.calls += 1
        return self.response


class GroqProvider:
    """Provider for Groq API with multi-key rotation and rate-limit backoff."""

    def __init__(self, keys: list[str]):
        if not keys:
            raise ValueError("No Groq API keys provided")
        self.keys = keys
        self.current_key_idx = 0

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.0, model: str | None = None) -> str:
        target_model = model or os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        max_attempts = len(self.keys) * 3
        for attempt in range(max_attempts):
            api_key = self.keys[self.current_key_idx]
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": target_model,
                "messages": messages,
                "temperature": temperature,
            }
            try:
                res = requests.post(url, headers=headers, json=payload, timeout=60)
                if res.status_code == 200:
                    data = res.json()
                    return data["choices"][0]["message"]["content"]
                elif res.status_code == 429:
                    # Rate limit: advance key and pause
                    self.current_key_idx = (self.current_key_idx + 1) % len(self.keys)
                    delay = min(2.0 * (attempt + 1), 60.0)
                    time.sleep(delay)
                    continue
                else:
                    raise RuntimeError(f"Groq API error ({res.status_code}): {res.text}")
            except requests.exceptions.RequestException as e:
                self.current_key_idx = (self.current_key_idx + 1) % len(self.keys)
                if attempt == max_attempts - 1:
                    raise RuntimeError(f"Groq request failed after {max_attempts} attempts: {e}")
                time.sleep(2.0)

        raise RuntimeError(f"Groq API failed after rotating through all {len(self.keys)} keys")


class GeminiProvider:
    """Provider for Gemini API via OpenAI-compatible endpoint."""

    def __init__(self, key: str):
        if not key:
            raise ValueError("No Gemini API key provided")
        self.key = key

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.0, model: str | None = None) -> str:
        target_model = model or os.environ.get("JUDGE_MODEL", "gemini-3.5-flash")
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
        }

        for attempt in range(5):
            try:
                res = requests.post(url, headers=headers, json=payload, timeout=60)
                if res.status_code == 200:
                    data = res.json()
                    # Small pacing delay to avoid hitting RPM limit
                    time.sleep(1.0)
                    return data["choices"][0]["message"]["content"]
                elif res.status_code in (429, 503):
                    time.sleep(3.0 * (attempt + 1))
                    continue
                else:
                    raise RuntimeError(f"Gemini API error ({res.status_code}): {res.text}")
            except requests.exceptions.RequestException as e:
                if attempt == 4:
                    raise RuntimeError(f"Gemini request failed: {e}")
                time.sleep(2.0)

        raise RuntimeError("Gemini API failed after multiple retries")


class HybridGroqGeminiProvider:
    """Hybrid provider routing generator models to Groq and judge models to Gemini."""

    def __init__(self, groq_keys: list[str], gemini_key: str):
        self.groq = GroqProvider(groq_keys)
        self.gemini = GeminiProvider(gemini_key)

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.0, model: str | None = None) -> str:
        target_model = model or os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
        if "gemini" in target_model.lower():
            return self.gemini.complete(prompt, system=system, temperature=temperature, model=target_model)
        else:
            return self.groq.complete(prompt, system=system, temperature=temperature, model=target_model)


def _cache_key(model: str, system: str | None, prompt: str) -> str:
    payload = json.dumps(
        {"model": model, "system": system, "prompt": prompt},
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def check_preflight() -> None:
    """Verify required LLM environment variables."""
    load_env()
    provider = os.environ.get("LLM_PROVIDER")
    model = os.environ.get("LLM_MODEL")
    judge = os.environ.get("JUDGE_MODEL")
    max_calls = os.environ.get("LLM_MAX_CALLS")

    missing = []
    if not provider:
        missing.append("LLM_PROVIDER")
    if not model:
        missing.append("LLM_MODEL")
    if not judge:
        missing.append("JUDGE_MODEL")
    if not max_calls:
        missing.append("LLM_MAX_CALLS")

    if missing:
        raise RuntimeError(f"Preflight check failed: missing environment variable(s): {', '.join(missing)}")

    if model == judge:
        raise RuntimeError(f"Preflight check failed: JUDGE_MODEL ({judge}) must differ from LLM_MODEL ({model})")


def get_call_count(cache_dir: Path) -> int:
    path = cache_dir / "call_count.json"
    if path.exists():
        try:
            return int(json.loads(path.read_text(encoding="utf-8")).get("calls", 0))
        except Exception:
            return 0
    return 0


def increment_call_count(cache_dir: Path) -> int:
    calls = get_call_count(cache_dir) + 1
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "call_count.json"
    path.write_text(json.dumps({"calls": calls}), encoding="utf-8")
    return calls


def complete(
    prompt: str,
    system: str | None = None,
    temperature: float = 0.0,
    *,
    model: str | None = None,
    provider: Provider | None = None,
    cache_dir: Path = Path("cache/llm"),
) -> str:
    """Complete one request through the configured provider and disk cache."""
    if temperature != 0.0:
        raise ValueError("pilot LLM calls must use temperature=0.0")

    load_env()
    provider_name = os.environ.get("LLM_PROVIDER")
    effective_model = model or os.environ.get("LLM_MODEL")

    if provider is None:
        check_preflight()
        if not effective_model:
            raise RuntimeError("LLM_MODEL is required; no provider call was made")

    # Check cache first
    key = _cache_key(effective_model or "default", system, prompt)
    cache_file = cache_dir / f"{key}.json"
    if cache_file.exists():
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        return payload["response"]

    # Check call cap
    max_calls_str = os.environ.get("LLM_MAX_CALLS", "500")
    try:
        max_calls = int(max_calls_str)
    except ValueError:
        max_calls = 500

    current_calls = get_call_count(cache_dir)
    if current_calls >= max_calls and provider is None:
        raise RuntimeError(f"LLM call cap reached ({current_calls}/{max_calls}); stopping to prevent unexpected spend")

    selected = provider or _provider_from_environment(provider_name or "fake")
    response = selected.complete(prompt, system=system, temperature=temperature, model=effective_model)

    # Save to cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps({"model": effective_model, "response": response}, ensure_ascii=False),
        encoding="utf-8",
    )
    if provider is None:
        increment_call_count(cache_dir)

    return response


def _provider_from_environment(name: str) -> Provider:
    if name == "fake":
        return FakeProvider()
    elif name == "groq":
        keys_str = os.environ.get("GROQ_API_KEYS", "") or os.environ.get("GROQ_API_KEY", "")
        keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        return GroqProvider(keys)
    elif name == "gemini":
        key = os.environ.get("GEMINI_API_KEY", "").strip()
        return GeminiProvider(key)
    elif name == "hybrid_groq_gemini":
        keys_str = os.environ.get("GROQ_API_KEYS", "") or os.environ.get("GROQ_API_KEY", "")
        keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
        return HybridGroqGeminiProvider(keys, gemini_key)

    raise RuntimeError(
        f"provider {name!r} is not implemented in the offline pilot; no provider call was made"
    )
