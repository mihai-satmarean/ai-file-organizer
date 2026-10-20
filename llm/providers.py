# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright (c) 2026 George Zhang — https://github.com/TheYellowDuck

"""
Pluggable LLM providers for classification.

The user brings their own API key for whichever provider they choose. Two
provider kinds cover essentially every vendor:

  * AnthropicProvider          — the official `anthropic` SDK (Claude).
  * OpenAICompatibleProvider   — the official `openai` SDK pointed at any
                                 OpenAI-compatible endpoint. This single class
                                 handles OpenAI, Google Gemini, Groq, Mistral,
                                 DeepSeek, Together, OpenRouter, local Ollama,
                                 and any custom base URL.

Each provider exposes one method, `complete_json(...)`, which returns parsed
JSON. The classifier builds the prompts; providers only handle transport and
the per-vendor structured-output knobs.

Keys are resolved lazily through `llm.secrets` and are never logged or stored.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from llm.secrets import resolve_api_key, mask, KeyResolution


# Each provider kind needs exactly one pip package. The import name and the
# pip requirement spec differ, so keep both.
_KIND_PACKAGES: dict[str, tuple[str, str]] = {
    "anthropic": ("anthropic", "anthropic>=0.40.0"),
    "openai": ("openai", "openai>=1.40.0"),
}


def _ensure_sdk(kind: str, *, auto_install: bool = True) -> None:
    """
    Make sure the SDK for a provider kind is importable, installing it on
    demand. No-op if it's already present or unknown. If auto_install is off,
    this does nothing and the provider raises a helpful ImportError instead.
    """
    spec = _KIND_PACKAGES.get(kind)
    if spec is None:
        return
    import_name, pip_spec = spec
    if importlib.util.find_spec(import_name) is not None:
        return
    if not auto_install:
        return

    print(f"📦 Installing '{import_name}' (required for this provider)…")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", pip_spec],
            check=True,
        )
    except (subprocess.CalledProcessError, OSError) as e:
        raise ImportError(
            f"Could not auto-install '{import_name}'. Install it manually with:\n"
            f"  {sys.executable} -m pip install {import_name}"
        ) from e
    importlib.invalidate_caches()


@dataclass
class ProviderPreset:
    """Static description of a known provider."""
    key: str
    label: str
    kind: str  # "anthropic" | "openai"
    env_names: list[str]
    default_model: str
    base_url: Optional[str] = None
    keyless: bool = False  # e.g. local Ollama needs no real key
    suggested_models: list[str] = field(default_factory=list)  # for the wizard


# Default models lean cheap/fast where the vendor offers a clear small model,
# since classification is a high-volume, simple task. All are overridable with
# --model. (Anthropic intentionally defaults to the flagship per Anthropic's
# guidance; switch to claude-haiku-4-5 for the cheapest Claude classification.)
PRESETS: dict[str, ProviderPreset] = {
    "anthropic": ProviderPreset(
        key="anthropic", label="Anthropic Claude", kind="anthropic",
        env_names=["ANTHROPIC_API_KEY"], default_model="claude-opus-4-8",
        suggested_models=["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"],
    ),
    "openai": ProviderPreset(
        key="openai", label="OpenAI", kind="openai",
        env_names=["OPENAI_API_KEY"], default_model="gpt-4o-mini",
        suggested_models=["gpt-4o-mini", "gpt-4o"],
    ),
    "gemini": ProviderPreset(
        key="gemini", label="Google Gemini", kind="openai",
        env_names=["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        default_model="gemini-2.0-flash",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        suggested_models=["gemini-2.0-flash", "gemini-1.5-pro"],
    ),
    "groq": ProviderPreset(
        key="groq", label="Groq", kind="openai",
        env_names=["GROQ_API_KEY"], default_model="llama-3.3-70b-versatile",
        base_url="https://api.groq.com/openai/v1",
    ),
    "mistral": ProviderPreset(
        key="mistral", label="Mistral", kind="openai",
        env_names=["MISTRAL_API_KEY"], default_model="mistral-small-latest",
        base_url="https://api.mistral.ai/v1",
    ),
    "deepseek": ProviderPreset(
        key="deepseek", label="DeepSeek", kind="openai",
        env_names=["DEEPSEEK_API_KEY"], default_model="deepseek-chat",
        base_url="https://api.deepseek.com",
    ),
    "together": ProviderPreset(
        key="together", label="Together AI", kind="openai",
        env_names=["TOGETHER_API_KEY"],
        default_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        base_url="https://api.together.xyz/v1",
    ),
    "openrouter": ProviderPreset(
        key="openrouter", label="OpenRouter", kind="openai",
        env_names=["OPENROUTER_API_KEY"], default_model="openai/gpt-4o-mini",
        base_url="https://openrouter.ai/api/v1",
    ),
    "ollama": ProviderPreset(
        key="ollama", label="Ollama (local)", kind="openai",
        env_names=["OLLAMA_API_KEY"], default_model="llama3.1",
        base_url="http://localhost:11434/v1", keyless=True,
    ),
    "openai-compatible": ProviderPreset(
        key="openai-compatible", label="Custom OpenAI-compatible", kind="openai",
        env_names=["LLM_API_KEY", "OPENAI_API_KEY"], default_model="",
        base_url=None,
    ),
}


def list_presets() -> list[ProviderPreset]:
    return list(PRESETS.values())


def _anthropic_content(user):
    """Render a neutral user payload (str or list of parts) for the Anthropic SDK."""
    if isinstance(user, str):
        return user
    out = []
    for p in user:
        if p["type"] == "image":
            out.append({
                "type": "image",
                "source": {"type": "base64", "media_type": p["media_type"], "data": p["data"]},
            })
        else:
            out.append({"type": "text", "text": p["text"]})
    return out


def _openai_content(user):
    """Render a neutral user payload (str or list of parts) for OpenAI-compatible SDKs."""
    if isinstance(user, str):
        return user
    out = []
    for p in user:
        if p["type"] == "image":
            url = f"data:{p['media_type']};base64,{p['data']}"
            out.append({"type": "image_url", "image_url": {"url": url}})
        else:
            out.append({"type": "text", "text": p["text"]})
    return out


def _extract_json(text: str) -> dict:
    """
    Parse a JSON object out of a model response, tolerating code fences and
    leading/trailing prose (some OpenAI-compatible endpoints add both).
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back to the first balanced {...} block.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError("Model did not return parseable JSON")


class BaseProvider:
    """Common surface shared by all providers."""

    def __init__(self, preset: ProviderPreset, model: str, key: KeyResolution):
        self.preset = preset
        self.model = model
        self._key = key
        # Cumulative token usage across all calls in this run (thread-safe:
        # batches may be classified concurrently).
        self._usage_lock = threading.Lock()
        self.usage = {
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }

    @property
    def label(self) -> str:
        return self.preset.label

    def status_line(self) -> str:
        """One line describing the active provider — key is masked, never shown in full."""
        return (
            f"{self.preset.label} · model={self.model} · "
            f"key={mask(self._key.value)} (from {self._key.source})"
        )

    def _record_usage(self, *, input_tokens=0, output_tokens=0,
                      cache_read=0, cache_creation=0) -> None:
        with self._usage_lock:
            self.usage["requests"] += 1
            self.usage["input_tokens"] += int(input_tokens or 0)
            self.usage["output_tokens"] += int(output_tokens or 0)
            self.usage["cache_read_input_tokens"] += int(cache_read or 0)
            self.usage["cache_creation_input_tokens"] += int(cache_creation or 0)

    def usage_summary(self) -> str:
        """Human-readable token/cache totals for the run (no secrets, no pricing guesses)."""
        u = self.usage
        cached = u["cache_read_input_tokens"]
        billed_input = u["input_tokens"] + u["cache_creation_input_tokens"] + cached
        parts = [
            f"{u['requests']} request(s)",
            f"in≈{billed_input:,} tok",
            f"out≈{u['output_tokens']:,} tok",
        ]
        if billed_input:
            pct = 100.0 * cached / billed_input
            parts.append(f"cache hits {pct:.0f}% ({cached:,} tok)")
        return "📊 LLM usage: " + ", ".join(parts)

    def complete_json(self, system: str, user, *,
                      schema: Optional[dict] = None, max_tokens: int = 4096) -> dict:
        raise NotImplementedError


class AnthropicProvider(BaseProvider):
    """Claude via the official Anthropic SDK, with prompt caching + structured output."""

    def __init__(self, preset, model, key, max_retries: int = 4):
        super().__init__(preset, model, key)
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "The 'anthropic' package is required for the Anthropic provider.\n"
                "Install it with: pip install anthropic"
            ) from e
        # api_key may be None only when resolution failed; surface a clear error.
        if not key.value:
            raise RuntimeError(
                "No Anthropic API key found. Set ANTHROPIC_API_KEY, add it to a "
                ".env file, or enter it when prompted."
            )
        # The SDK retries 429/5xx with exponential backoff (default 2); bump it up.
        self._client = anthropic.Anthropic(api_key=key.value, max_retries=max_retries)

    def complete_json(self, system: str, user, *,
                      schema: Optional[dict] = None, max_tokens: int = 4096) -> dict:
        # The category tree lives in `system` and is identical across batches, so
        # caching it makes every batch after the first ~10x cheaper on that prefix.
        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=[{
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": _anthropic_content(user)}],
        )
        # Structured output guarantees parseable JSON, but can't express the
        # recursive schema used for category discovery — omit it in that case.
        if schema is not None:
            kwargs["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
        resp = self._client.messages.create(**kwargs)
        u = resp.usage
        self._record_usage(
            input_tokens=getattr(u, "input_tokens", 0),
            output_tokens=getattr(u, "output_tokens", 0),
            cache_read=getattr(u, "cache_read_input_tokens", 0),
            cache_creation=getattr(u, "cache_creation_input_tokens", 0),
        )
        text = next((b.text for b in resp.content if b.type == "text"), "")
        return _extract_json(text)


class OpenAICompatibleProvider(BaseProvider):
    """OpenAI and every OpenAI-compatible vendor via the official `openai` SDK."""

    def __init__(self, preset, model, key, base_url: Optional[str] = None, max_retries: int = 4):
        super().__init__(preset, model, key)
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "The 'openai' package is required for OpenAI-compatible providers.\n"
                "Install it with: pip install openai"
            ) from e
        resolved_base = base_url or preset.base_url
        if not key.value and not preset.keyless:
            names = " / ".join(preset.env_names)
            raise RuntimeError(
                f"No API key found for {preset.label}. Set {names}, add it to a "
                ".env file, or enter it when prompted."
            )
        # The SDK requires a non-empty key string even for keyless local servers.
        # max_retries gives exponential backoff on 429/5xx.
        self._client = OpenAI(
            api_key=key.value or "not-needed",
            base_url=resolved_base,
            max_retries=max_retries,
            timeout=300.0,  # 5 min per request; avoids premature timeout on busy hosts
        )

    def complete_json(self, system: str, user, *,
                      schema: Optional[dict] = None, max_tokens: int = 4096) -> dict:
        # json_object mode is supported far more widely across OpenAI-compatible
        # vendors than json_schema, so we request that and validate defensively.
        # (`schema` is described in the prompt by the caller; not sent here.)
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": _openai_content(user)},
            ],
        )
        u = getattr(resp, "usage", None)
        cached = 0
        details = getattr(u, "prompt_tokens_details", None) if u else None
        if details is not None:
            cached = getattr(details, "cached_tokens", 0) or 0
        if u is not None:
            self._record_usage(
                input_tokens=(getattr(u, "prompt_tokens", 0) or 0) - cached,
                output_tokens=getattr(u, "completion_tokens", 0),
                cache_read=cached,
            )
        text = resp.choices[0].message.content or ""
        return _extract_json(text)


def build_provider(
    provider_key: str,
    *,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    allow_prompt: bool = True,
    auto_install: bool = True,
    max_retries: int = 4,
    project_dir: Optional[Path] = None,
) -> BaseProvider:
    """
    Construct a ready-to-use provider, resolving the key securely.

    Args:
        provider_key: One of the keys in PRESETS (e.g. "anthropic", "gemini").
        model: Optional model override; falls back to the preset default.
        base_url: Optional endpoint override (for custom/self-hosted vendors).
        allow_prompt: Whether a hidden interactive prompt is allowed.
        auto_install: If True, install the provider's SDK on demand when missing.
        project_dir: Where to look for a .env file.
    """
    preset = PRESETS.get(provider_key)
    if preset is None:
        known = ", ".join(PRESETS)
        raise ValueError(f"Unknown provider '{provider_key}'. Choose one of: {known}")

    chosen_model = (model or preset.default_model or "").strip()
    if not chosen_model:
        raise ValueError(
            f"Provider '{provider_key}' has no default model — pass --model to set one."
        )

    # Install the matching SDK if needed, before we try to import it.
    _ensure_sdk(preset.kind, auto_install=auto_install)

    key = resolve_api_key(
        preset.env_names,
        provider_label=preset.label,
        keyless=preset.keyless,
        allow_prompt=allow_prompt,
        project_dir=project_dir,
    )

    if preset.kind == "anthropic":
        return AnthropicProvider(preset, chosen_model, key, max_retries=max_retries)
    return OpenAICompatibleProvider(preset, chosen_model, key, base_url=base_url, max_retries=max_retries)
