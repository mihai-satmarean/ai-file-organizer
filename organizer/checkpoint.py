# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
Per-file checkpoint manifest for resumable organization runs.

Writes a JSON progress file into the INPUT directory after each batch so that
a crash at file 800 of 1685 can be resumed without re-processing the first 800.

Safety: every save goes through a tmp-file + os.replace atomic rename so a
crash mid-write never corrupts the checkpoint.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

CHECKPOINT_FILE = ".organizer-progress.json"


class OrganizeCheckpoint:
    """Per-file checkpoint manifest for resumable organization runs."""

    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir = Path(input_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.checkpoint_path = self.input_dir / CHECKPOINT_FILE
        self.manifest: dict[str, dict] = {}
        self._load()

    # ---- persistence ---------------------------------------------------------

    def _load(self):
        if self.checkpoint_path.is_file():
            data = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
            self.manifest = data.get("files", {})

    def _save(self):
        data = {
            "input_dir": str(self.input_dir),
            "output_dir": str(self.output_dir),
            "updated_at": datetime.now().isoformat(),
            "total": len(self.manifest),
            "done": sum(1 for v in self.manifest.values() if v["status"] == "done"),
            "failed": sum(1 for v in self.manifest.values() if v["status"] == "failed"),
            "pending": sum(1 for v in self.manifest.values() if v["status"] == "pending"),
            "files": self.manifest,
        }
        tmp = self.checkpoint_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(str(tmp), str(self.checkpoint_path))

    # ---- registration and queries --------------------------------------------

    def register_files(self, file_paths: list[Path]):
        """Register all files to process. Already-known files keep their status."""
        for fp in file_paths:
            rel = str(fp.resolve().relative_to(self.input_dir))
            if rel not in self.manifest:
                self.manifest[rel] = {"status": "pending", "timestamp": None}
        self._save()

    def get_pending(self) -> list[str]:
        """Return relative paths of files that still need processing."""
        return [
            k for k, v in self.manifest.items()
            if v["status"] in ("pending", "failed")
        ]

    # ---- status updates ------------------------------------------------------

    def mark_done(
        self,
        rel_path: str,
        category: str,
        new_name: Optional[str],
        confidence: float,
    ):
        self.manifest[rel_path] = {
            "status": "done",
            "category": category,
            "new_name": new_name,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
        }

    def mark_failed(self, rel_path: str, error: str):
        self.manifest[rel_path] = {
            "status": "failed",
            "error": error,
            "timestamp": datetime.now().isoformat(),
        }

    def save_batch(self):
        """Flush checkpoint to disk — call after each batch."""
        self._save()

    # ---- reporting -----------------------------------------------------------

    def summary(self) -> str:
        done = sum(1 for v in self.manifest.values() if v["status"] == "done")
        failed = sum(1 for v in self.manifest.values() if v["status"] == "failed")
        pending = sum(1 for v in self.manifest.values() if v["status"] == "pending")
        total = len(self.manifest)
        return f"Progress: {done}/{total} done, {failed} failed, {pending} pending"

    @property
    def is_complete(self) -> bool:
        return bool(self.manifest) and all(
            v["status"] == "done" for v in self.manifest.values()
        )

    def clear(self):
        """Remove checkpoint file to start fresh."""
        if self.checkpoint_path.is_file():
            self.checkpoint_path.unlink()
        self.manifest = {}
