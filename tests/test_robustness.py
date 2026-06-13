"""Scanner excludes, batch split-retry, usage estimate, and the spend guard."""

import re

import pytest

import llm
from config import OrganizerConfig
from main import FileOrganizer
from organizer.scanner import DirectoryScanner
from categories import create_default_categories
from extractors.base import ExtractedContent
from llm.classifier import LLMClassifier
from pathlib import Path

from conftest import FakeProvider


# ---- scanner excludes ----------------------------------------------------

def test_scanner_ignore_patterns(tmp_path):
    (tmp_path / "keep.txt").write_text("x")
    (tmp_path / "secret_notes.txt").write_text("x")
    (tmp_path / "app.log").write_text("x")
    sc = DirectoryScanner(ignore_patterns=["secret*", "*.log"])
    res = sc.scan(tmp_path, recursive=False)
    assert {p.name for p in res.file_paths} == {"keep.txt"}


# ---- batch split-retry ---------------------------------------------------

def _content(name):
    p = Path(name)
    return ExtractedContent(p, p.name, p.suffix, "some text", {}, True)


def test_batch_splits_until_it_succeeds():
    # A model that fails on multi-file batches but works one-at-a-time.
    def flaky(user):
        if len(re.findall(r"name=", user)) > 1:
            raise RuntimeError("batch too large / truncated")
        return {"results": [{"index": 0, "path": "Documents", "confidence": 0.9}]}

    clf = LLMClassifier(create_default_categories(), FakeProvider(flaky), batch_size=4)
    contents = [_content(f"f{i}.txt") for i in range(4)]
    results = clf.classify_batch(contents, show_progress=False)

    # Every file got classified by the model (not dumped into 'Other' by failure).
    assert all(r.category_path != "Other" for r in results)
    # It actually split (more calls than the single initial batch).
    assert len(clf.provider.calls) > 4


def test_single_file_failure_falls_back_to_other():
    def always_fail(user):
        raise RuntimeError("down")
    clf = LLMClassifier(create_default_categories(), FakeProvider(always_fail), batch_size=4)
    results = clf.classify_batch([_content("a.txt")], show_progress=False)
    assert results[0].category_path == "Other"


# ---- usage estimate ------------------------------------------------------

def test_estimate_usage():
    clf = LLMClassifier(create_default_categories(), FakeProvider({"results": []}), batch_size=2)
    tokens, requests = clf.estimate_usage([_content(f"f{i}.txt") for i in range(5)])
    assert tokens > 0
    assert requests == 3  # ceil(5 / 2)


# ---- spend guard ---------------------------------------------------------

def test_max_files_guard_blocks_oversized_run(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "build_provider", lambda *a, **k: FakeProvider({"results": []}))
    inp = tmp_path / "in"
    inp.mkdir()
    (inp / "a.txt").write_text("a")
    (inp / "b.txt").write_text("b")
    cfg = OrganizerConfig(
        backend="llm", llm_provider="fake", llm_max_files=1,
        input_directories=[str(inp)], output_directory=str(tmp_path / "out"),
    )
    with pytest.raises(ValueError, match="max-files"):
        FileOrganizer(cfg).organize(show_progress=False, assume_yes=True)
