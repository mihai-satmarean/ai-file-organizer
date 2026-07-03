# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright (c) 2026 George Zhang — https://github.com/TheYellowDuck

"""File mover: move, copy, dedupe, dry-run, and undo-script generation."""

from pathlib import Path

from organizer.mover import FileMover


def _make_file(path: Path, text="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_move_places_file(tmp_path):
    src = _make_file(tmp_path / "src" / "a.txt")
    out = tmp_path / "out"
    mover = FileMover(out, create_log=False)
    op = mover.organize_file(src, "School/Math", 0.9)
    assert op.success
    assert (out / "School/Math/a.txt").is_file()
    assert not src.exists()  # moved, not copied


def test_copy_keeps_original(tmp_path):
    src = _make_file(tmp_path / "src" / "a.txt")
    out = tmp_path / "out"
    mover = FileMover(out, copy_instead_of_move=True, create_log=False)
    mover.organize_file(src, "Work", 0.8)
    assert src.exists()
    assert (out / "Work/a.txt").is_file()


def test_dedupe_rename(tmp_path):
    out = tmp_path / "out"
    for _ in range(2):
        src = _make_file(tmp_path / "src" / "a.txt")
        FileMover(out, handle_duplicates="rename", create_log=False).organize_file(src, "X", 0.5)
    names = sorted(p.name for p in (out / "X").iterdir())
    assert names == ["a.txt", "a_1.txt"]


def test_dedupe_skip(tmp_path):
    out = tmp_path / "out"
    src1 = _make_file(tmp_path / "src1" / "a.txt")
    FileMover(out, create_log=False).organize_file(src1, "X", 0.5)
    src2 = _make_file(tmp_path / "src2" / "a.txt")
    op = FileMover(out, handle_duplicates="skip", create_log=False).organize_file(src2, "X", 0.5)
    assert not op.success
    assert "skip" in (op.error_message or "").lower()
    assert src2.exists()  # left in place


def test_dry_run_moves_nothing(tmp_path):
    src = _make_file(tmp_path / "src" / "a.txt")
    out = tmp_path / "out"
    op = FileMover(out, dry_run=True, create_log=False).organize_file(src, "X", 0.5)
    assert op.success
    assert src.exists()
    assert not (out / "X" / "a.txt").exists()


def test_undo_script_lists_moves(tmp_path):
    src = _make_file(tmp_path / "src" / "a.txt")
    out = tmp_path / "out"
    mover = FileMover(out, create_log=False)
    result = mover.organize_files([(src, "X", 0.9)], show_progress=False)
    script = mover.create_undo_script(result)
    assert script.is_file()
    body = script.read_text()
    assert body.startswith("#!/bin/bash")
    assert "mv " in body
    assert "a.txt" in body
