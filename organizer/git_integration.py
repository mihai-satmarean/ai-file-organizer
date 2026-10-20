# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright (c) 2026 George Zhang — https://github.com/TheYellowDuck

"""
Git integration for organized folders.

Provides repo detection, init with optional LFS, safety pre-checks,
and batch commits after file-organization runs.
"""

import subprocess
from collections import Counter
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from organizer.mover import MoveOperation


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class GitStatus:
    """Snapshot of the git state for a folder."""

    is_repo: bool
    has_uncommitted: bool
    branch: str
    lfs_enabled: bool
    repo_root: Optional[Path] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_git(
    *args: str,
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    """Run a git command, returning the CompletedProcess.

    Raises ``FileNotFoundError`` when git is not on ``$PATH``.
    """
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _git_available() -> bool:
    """Return True when the git binary is reachable."""
    try:
        subprocess.run(
            ["git", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _lfs_in_gitattributes(repo_root: Path) -> bool:
    """Check whether .gitattributes contains at least one LFS filter line."""
    ga = repo_root / ".gitattributes"
    if not ga.is_file():
        return False
    try:
        return "filter=lfs" in ga.read_text(encoding="utf-8")
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_git_repo(folder: Path) -> GitStatus:
    """Check whether *folder* lives inside a git repository.

    Returns a populated :class:`GitStatus`; when git is missing or the
    folder is not a repo, sensible defaults are used.
    """
    folder = Path(folder).resolve()

    if not _git_available():
        return GitStatus(
            is_repo=False,
            has_uncommitted=False,
            branch="",
            lfs_enabled=False,
        )

    # Is it inside a work-tree?
    proc = _run_git("rev-parse", "--show-toplevel", cwd=folder)
    if proc.returncode != 0:
        return GitStatus(
            is_repo=False,
            has_uncommitted=False,
            branch="",
            lfs_enabled=False,
        )

    repo_root = Path(proc.stdout.strip())

    # Uncommitted changes (staged or unstaged)
    porcelain = _run_git("status", "--porcelain", cwd=folder)
    has_uncommitted = bool(porcelain.stdout.strip())

    # Current branch
    branch_proc = _run_git("branch", "--show-current", cwd=folder)
    branch = branch_proc.stdout.strip() if branch_proc.returncode == 0 else ""

    # LFS
    lfs_enabled = _lfs_in_gitattributes(repo_root)

    return GitStatus(
        is_repo=True,
        has_uncommitted=has_uncommitted,
        branch=branch,
        lfs_enabled=lfs_enabled,
        repo_root=repo_root,
    )


def git_init(
    folder: Path,
    lfs_patterns: Optional[list[str]] = None,
) -> Path:
    """Initialise a git repository in *folder*.

    When *lfs_patterns* is provided (e.g. ``['*.pdf', '*.docx']``),
    ``git lfs install`` is executed and a ``.gitattributes`` file is
    written and committed.

    Returns:
        The repository root (always *folder* itself).

    Raises:
        RuntimeError: if ``git init`` fails.
    """
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)

    proc = _run_git("init", cwd=folder)
    if proc.returncode != 0:
        raise RuntimeError(f"git init failed: {proc.stderr.strip()}")

    if lfs_patterns:
        lfs_proc = _run_git("lfs", "install", cwd=folder)
        if lfs_proc.returncode != 0:
            raise RuntimeError(f"git lfs install failed: {lfs_proc.stderr.strip()}")

        ga_path = folder / ".gitattributes"
        lines = [f"{pat} filter=lfs diff=lfs merge=lfs -text" for pat in lfs_patterns]
        ga_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        _run_git("add", ".gitattributes", cwd=folder)

    _run_git("add", ".", cwd=folder)

    # Only commit if there is something staged (an empty folder has nothing).
    staged = _run_git("diff", "--cached", "--name-only", cwd=folder)
    if staged.stdout.strip():
        _run_git("commit", "-m", "init: set up repository", cwd=folder)

    return folder


def _build_commit_message(operations: list[MoveOperation]) -> str:
    """Derive a commit message from a list of move operations.

    Format example::

        organize: 33 files (5 CV, 8 Oferte, 12 Financiare, 3 Archives, 5 Other)
    """
    counts: Counter[str] = Counter()
    for op in operations:
        top_level = Path(op.category_path).parts[0] if op.category_path else "Other"
        counts[top_level] += 1

    total = sum(counts.values())

    # Sort categories by count descending, then alphabetically for ties.
    sorted_cats = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    parts = ", ".join(f"{count} {cat}" for cat, count in sorted_cats)
    return f"organize: {total} files ({parts})"


def _repo_root(folder: Path) -> Optional[Path]:
    """Return the git repo root for *folder*, or None."""
    proc = _run_git("rev-parse", "--show-toplevel", cwd=folder)
    if proc.returncode != 0:
        return None
    return Path(proc.stdout.strip())


def git_commit_batch(
    folder: Path,
    operations: list[MoveOperation],
    dry_run: bool = False,
    message: Optional[str] = None,
) -> Optional[str]:
    """Commit only the files touched by the organisation run.

    When the organised folder is a subdirectory of a larger repo (e.g.
    ``docs-skylite/_arhiva`` inside the ``acda`` repo), we stage only
    the source and destination paths from the operations — never
    ``git add --all`` which would pull in unrelated changes.

    Args:
        folder: the organised folder (or any path inside the repo).
        operations: list of :class:`MoveOperation` objects from the mover.
        dry_run: if ``True``, return the planned commit message without
            touching git.
        message: override the auto-generated commit message.

    Returns:
        The commit SHA on success, the planned message when *dry_run* is
        ``True``, or ``None`` if there was nothing to commit.
    """
    folder = Path(folder).resolve()
    msg = message or _build_commit_message(operations)

    if dry_run:
        return msg

    repo_root = _repo_root(folder)
    if repo_root is None:
        return None

    # Stage only the specific files that were moved/renamed.
    paths_to_stage: list[str] = []
    for op in operations:
        if op.success:
            paths_to_stage.append(str(op.source_path))
            paths_to_stage.append(str(op.destination_path))

    if not paths_to_stage:
        return None

    # git add the specific paths (source is deleted, dest is new)
    add_proc = _run_git("add", "--", *paths_to_stage, cwd=repo_root)
    if add_proc.returncode != 0:
        # Fallback: some source paths may no longer exist (moved away).
        # Stage the whole organised folder instead (still scoped).
        _run_git("add", "--all", str(folder), cwd=repo_root)

    # Bail out when the index is clean (nothing to commit)
    diff = _run_git("diff", "--cached", "--quiet", cwd=repo_root)
    if diff.returncode == 0:
        return None

    commit_proc = _run_git("commit", "-m", msg, cwd=repo_root)
    if commit_proc.returncode != 0:
        raise RuntimeError(f"git commit failed: {commit_proc.stderr.strip()}")

    sha_proc = _run_git("rev-parse", "HEAD", cwd=repo_root)
    if sha_proc.returncode != 0:
        raise RuntimeError(f"git rev-parse HEAD failed: {sha_proc.stderr.strip()}")

    return sha_proc.stdout.strip()


def git_pre_check(folder: Path) -> tuple[bool, str]:
    """Pre-flight safety check before organising inside a git repo.

    Only checks for uncommitted changes *within the folder being
    organised*, not the entire repo.  This matters when the folder is
    a subdirectory of a larger monorepo (e.g. ``docs-skylite/_arhiva``
    inside the ``acda`` project).

    Returns:
        ``(ok, reason)`` — ``ok`` is ``False`` when there are pre-existing
        uncommitted changes inside *folder*.
    """
    folder = Path(folder).resolve()

    if not _git_available():
        return True, "git not installed; skipping pre-check"

    status = detect_git_repo(folder)

    if not status.is_repo:
        return True, "not a git repository; no constraints"

    # Check only files under this folder, not the whole repo.
    porcelain = _run_git("status", "--porcelain", "--", str(folder), cwd=folder)
    if porcelain.stdout.strip():
        dirty_count = len(porcelain.stdout.strip().splitlines())
        return False, (
            f"there are {dirty_count} uncommitted change(s) inside {folder.name}/; "
            "commit or stash them first so the organise commit stays clean"
        )

    return True, "working tree is clean"
