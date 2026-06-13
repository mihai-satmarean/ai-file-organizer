"""End-to-end: scan -> classify (stub LLM) -> move -> journal -> undo restores."""

import json
import re
import subprocess
import shutil
from pathlib import Path

import pytest

import llm
from config import OrganizerConfig
from main import FileOrganizer

from conftest import FakeProvider


CATEGORIES = {"categories": [
    {"name": "Alpha", "description": "files starting with a"},
    {"name": "Beta", "description": "files starting with b"},
]}


def _smart_payload(user: str) -> dict:
    """Classify each listed file by its filename so the result is order-independent."""
    results = []
    for m in re.finditer(r"\[(\d+)\] name=(.+)", user):
        idx = int(m.group(1))
        name = m.group(2).strip().strip("'\"")
        if name.startswith("a"):
            path = "Alpha"
        elif name.startswith("b"):
            path = "Beta"
        else:
            path = "Other"
        results.append({"index": idx, "path": path, "confidence": 0.9})
    return {"results": results}


@pytest.fixture
def project(tmp_path, monkeypatch):
    inp = tmp_path / "in"
    out = tmp_path / "out"
    inp.mkdir()
    for name in ("apple.txt", "avocado.txt", "banana.txt", "cherry.txt"):
        (inp / name).write_text(name)
    cats = tmp_path / "categories.json"
    cats.write_text(json.dumps(CATEGORIES))

    # Stub the LLM provider so no network/key is needed.
    monkeypatch.setattr(llm, "build_provider", lambda *a, **k: FakeProvider(_smart_payload))

    cfg = OrganizerConfig(
        backend="llm", llm_provider="fake", llm_vision=False,
        custom_categories_file=str(cats),
        input_directories=[str(inp)], output_directory=str(out),
    )
    return cfg, inp, out


def test_end_to_end_move_and_journal(project):
    cfg, inp, out = project
    result = FileOrganizer(cfg).organize(show_progress=False, assume_yes=True)["organization_result"]

    # Files landed in the right folders.
    assert (out / "Alpha" / "apple.txt").is_file()
    assert (out / "Alpha" / "avocado.txt").is_file()
    assert (out / "Beta" / "banana.txt").is_file()
    assert (out / "Other" / "cherry.txt").is_file()
    # Sources were moved, not copied.
    assert not (inp / "apple.txt").exists()

    # Crash-safe journal recorded every move.
    assert result.journal_path and result.journal_path.is_file()
    lines = [l for l in result.journal_path.read_text().splitlines() if l.strip()]
    assert len(lines) == 4
    assert all("source" in json.loads(l) and "destination" in json.loads(l) for l in lines)

    # Undo + log written automatically.
    assert result.undo_path and result.undo_path.is_file()
    assert result.log_path and result.log_path.is_file()


def test_undo_script_actually_restores(project):
    cfg, inp, out = project
    result = FileOrganizer(cfg).organize(show_progress=False, assume_yes=True)["organization_result"]

    assert not (inp / "apple.txt").exists()  # moved away
    bash = shutil.which("bash")
    assert bash, "bash required for this test"
    proc = subprocess.run([bash, str(result.undo_path)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    # Originals are back where they started.
    for name in ("apple.txt", "avocado.txt", "banana.txt", "cherry.txt"):
        assert (inp / name).is_file()
    assert not (out / "Alpha" / "apple.txt").exists()


def test_guard_blocks_output_in_input(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "build_provider", lambda *a, **k: FakeProvider({"results": []}))
    inp = tmp_path / "in"
    inp.mkdir()
    (inp / "x.txt").write_text("x")
    cfg = OrganizerConfig(backend="llm", llm_provider="fake",
                          input_directories=[str(inp)], output_directory=str(inp / "organized"))
    with pytest.raises(ValueError):
        FileOrganizer(cfg).organize(show_progress=False, assume_yes=True)
