from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Protocol


class Provider(Protocol):
    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
        ...


class FakeProvider:
    """Deterministic provider for tests; never contacts a network."""

    def __init__(self, response: str = "fake response"):
        self.response = response
        self.calls = 0

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
        self.calls += 1
        return self.response


def _cache_key(model: str, system: str | None, prompt: str) -> str:
    payload = json.dumps(
        {"model": model, "system": system, "prompt": prompt},
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def complete(
    prompt: str,
    system: str | None = None,
    temperature: float = 0.0,
    *,
    provider: Provider | None = None,
    cache_dir: Path = Path("cache/llm"),
) -> str:
    """Complete one request through the configured provider and disk cache."""
    if temperature != 0.0:
        raise ValueError("pilot LLM calls must use temperature=0.0")
    provider_name = os.environ.get("LLM_PROVIDER")
    model = os.environ.get("LLM_MODEL")
    if not provider_name:
        raise RuntimeError("LLM_PROVIDER is required; no provider call was made")
    if not model:
        raise RuntimeError("LLM_MODEL is required; no provider call was made")
    key = _cache_key(model, system, prompt)
    path = cache_dir / f"{key}.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload["response"]
    selected = provider or _provider_from_environment(provider_name)
    response = selected.complete(prompt, system=system, temperature=temperature)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"model": model, "response": response}, ensure_ascii=False),
        encoding="utf-8",
    )
    return response


def _provider_from_environment(name: str) -> Provider:
    if name == "fake":
        return FakeProvider()
    raise RuntimeError(
        f"provider {name!r} is not implemented in the offline pilot; no provider call was made"
    )
