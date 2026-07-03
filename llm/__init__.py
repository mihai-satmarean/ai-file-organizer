# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright (c) 2026 George Zhang — https://github.com/TheYellowDuck

"""LLM backend: bring-your-own-key classification across all major providers."""

from llm.providers import (
    BaseProvider,
    ProviderPreset,
    PRESETS,
    list_presets,
    build_provider,
)
from llm.classifier import LLMClassifier
from llm.discover import discover_categories
from llm.secrets import resolve_api_key, mask, save_key_to_dotenv

__all__ = [
    "BaseProvider",
    "ProviderPreset",
    "PRESETS",
    "list_presets",
    "build_provider",
    "LLMClassifier",
    "discover_categories",
    "resolve_api_key",
    "mask",
    "save_key_to_dotenv",
]
