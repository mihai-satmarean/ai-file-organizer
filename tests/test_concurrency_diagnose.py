# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright (c) 2026 George Zhang — https://github.com/TheYellowDuck

"""Concurrent classification preserves order; retries are wired; diagnose works."""

import re
from pathlib import Path

from categories.tree import CategoryTree
from extractors.base import ExtractedContent
from llm.classifier import LLMClassifier
from llm.providers import AnthropicProvider, ProviderPreset
from llm.secrets import KeyResolution

from conftest import FakeProvider


def _content(name):
    p = Path(name)
    return ExtractedContent(p, p.name, p.suffix, "text", {}, True)


def _numbered_tree(n):
    return CategoryTree.from_dict({"categories": [
        {"name": f"C{i}", "description": f"category {i}"} for i in range(n)
    ]})


def test_concurrency_preserves_order():
    """Each file fK.txt should map to category CK regardless of parallel timing."""
    def payload(user):
        results = []
        for m in re.finditer(r"\[(\d+)\] name=('?\"?)f(\d+)\.txt", user):
            within_idx = int(m.group(1))
            file_num = int(m.group(3))
            results.append({"index": within_idx, "path": f"C{file_num}", "confidence": 0.9})
        return {"results": results}

    tree = _numbered_tree(30)
    clf = LLMClassifier(tree, FakeProvider(payload), batch_size=3, concurrency=8)
    contents = [_content(f"f{i}.txt") for i in range(30)]
    results = clf.classify_batch(contents, show_progress=False)

    assert len(results) == 30
    for i, r in enumerate(results):
        assert r.category_path == f"C{i}", f"order broke at {i}: {r.category_path}"


def test_concurrency_single_batch_is_fine():
    clf = LLMClassifier(_numbered_tree(3), FakeProvider({"results": []}), batch_size=10, concurrency=8)
    results = clf.classify_batch([_content("f0.txt")], show_progress=False)
    assert len(results) == 1


def test_max_retries_wired_to_client():
    preset = ProviderPreset("anthropic", "Anthropic", "anthropic",
                            ["ANTHROPIC_API_KEY"], "claude-opus-4-8")
    p = AnthropicProvider(preset, "claude-opus-4-8", KeyResolution("sk-x", "test"), max_retries=7)
    assert p._client.max_retries == 7


def test_diagnose_returns_ranked_candidates():
    payload = {"candidates": [
        {"path": "C1", "confidence": 0.9, "reason": "best"},
        {"path": "nonexistent/path", "confidence": 0.4, "reason": "fallback"},
        {"path": "C2", "confidence": 0.2, "reason": "maybe"},
        {"path": "C0", "confidence": 0.1, "reason": "unlikely"},
    ]}
    clf = LLMClassifier(_numbered_tree(5), FakeProvider(payload))
    out = clf.diagnose(_content("f1.txt"), top_k=3)
    assert len(out) == 3                       # capped at top_k
    assert out[0]["path"] == "C1"
    assert out[1]["path"] == "Other"           # invalid path normalized
    assert all(0.0 <= c["confidence"] <= 1.0 for c in out)
    assert all(c["reason"] for c in out)
