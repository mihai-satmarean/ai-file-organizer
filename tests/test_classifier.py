# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright (c) 2026 George Zhang — https://github.com/TheYellowDuck

"""LLM classifier: path normalization, Review routing, batch fallback."""

from pathlib import Path

from categories import create_waterloo_example_categories as create_default_categories
from extractors.base import ExtractedContent
from llm.classifier import LLMClassifier

from conftest import FakeProvider


def _content(name, text=""):
    return ExtractedContent(Path(name), name, Path(name).suffix, text, {}, True)


def _classifier(payload, **kw):
    tree = create_default_categories()
    return LLMClassifier(tree, FakeProvider(payload), **kw)


def test_normalize_exact_leaf():
    clf = _classifier({"results": []})
    assert clf._normalize_path("School/University/CS 246") == ("School/University/CS 246", False)


def test_normalize_invalid_to_other():
    clf = _classifier({"results": []})
    assert clf._normalize_path("totally/made/up") == ("Other", True)


def test_normalize_parent_routes_to_other():
    clf = _classifier({"results": []})
    path, is_other = clf._normalize_path("School/University")
    assert path == "School/University/Other"
    assert is_other is True


def test_normalize_case_insensitive():
    clf = _classifier({"results": []})
    assert clf._normalize_path("school/university/cs 246")[0] == "School/University/CS 246"


def test_min_confidence_routes_to_review():
    payload = {"results": [{"index": 0, "path": "School/University/CS 246", "confidence": 0.2}]}
    clf = _classifier(payload, min_confidence=0.6)
    [res] = clf.classify_batch([_content("a.txt")], show_progress=False)
    assert res.category_path == "Review"
    assert res.final_confidence == 0.2


def test_high_confidence_not_reviewed():
    payload = {"results": [{"index": 0, "path": "School/University/CS 246", "confidence": 0.95}]}
    clf = _classifier(payload, min_confidence=0.6)
    [res] = clf.classify_batch([_content("a.txt")], show_progress=False)
    assert res.category_path == "School/University/CS 246"


def test_missing_index_falls_back_to_other():
    # Model omits index 1 → that file must not be lost.
    payload = {"results": [{"index": 0, "path": "Work/Job Search/Resumes", "confidence": 0.9}]}
    clf = _classifier(payload)
    results = clf.classify_batch([_content("a.txt"), _content("b.txt")], show_progress=False)
    assert results[0].category_path == "Work/Job Search/Resumes"
    assert results[1].category_path == "Other"


def test_provider_failure_falls_back(monkeypatch):
    def boom(user):
        raise RuntimeError("api down")
    clf = _classifier(boom)
    results = clf.classify_batch([_content("a.txt"), _content("b.txt")], show_progress=False)
    assert all(r.category_path == "Other" for r in results)


def test_batching_splits_requests():
    payload = lambda user: {"results": []}  # all fall back to Other
    clf = _classifier(payload, batch_size=2)
    contents = [_content(f"f{i}.txt") for i in range(5)]
    clf.classify_batch(contents, show_progress=False)
    # 5 files / batch_size 2 => 3 requests
    assert len(clf.provider.calls) == 3
