"""Provider helpers: tolerant JSON parsing, usage accounting, SDK presence check."""

import pytest

from llm import providers
from llm.providers import _extract_json, _ensure_sdk, BaseProvider, ProviderPreset
from llm.secrets import KeyResolution


def test_extract_json_plain():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced():
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_with_prose():
    assert _extract_json('Sure, here you go: {"a": 1} done') == {"a": 1}


def test_extract_json_invalid_raises():
    with pytest.raises(ValueError):
        _extract_json("not json at all")


def test_ensure_sdk_present_is_noop():
    # anthropic is installed in the test env; this must not attempt a pip install.
    _ensure_sdk("anthropic")  # no exception
    _ensure_sdk("unknown-kind")  # unknown kinds are ignored


def test_ensure_sdk_missing_no_autoinstall_is_noop(monkeypatch):
    # With auto_install off, a missing SDK must not trigger pip.
    monkeypatch.setattr(providers.importlib.util, "find_spec", lambda name: None)
    called = {"pip": False}
    monkeypatch.setattr(providers.subprocess, "run",
                        lambda *a, **k: called.__setitem__("pip", True))
    _ensure_sdk("openai", auto_install=False)
    assert called["pip"] is False


def test_usage_summary_reports_cache_hits():
    preset = ProviderPreset("x", "X", "openai", ["X_API_KEY"], "m")
    p = BaseProvider(preset, "m", KeyResolution("sk-x", "env:X_API_KEY"))
    p._record_usage(input_tokens=100, output_tokens=20, cache_creation=0, cache_read=900)
    p._record_usage(input_tokens=50, output_tokens=10, cache_read=950)
    summary = p.usage_summary()
    assert "2 request(s)" in summary
    assert "cache hits" in summary
    # 1850 cached of 2000 billed input ≈ 92%
    assert "92%" in summary
