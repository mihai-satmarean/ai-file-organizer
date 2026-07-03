# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright (c) 2026 George Zhang — https://github.com/TheYellowDuck

"""
File mover for organizing files into the category structure.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import shutil
import json
import os


@dataclass
class MoveOperation:
    """Represents a single file move operation."""
    source_path: Path
    destination_path: Path
    category_path: str
    confidence: float
    success: bool = False
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OrganizationResult:
    """Result of organizing files."""
    total_files: int
    successful_moves: int
    failed_moves: int
    skipped_files: int
    operations: list[MoveOperation]
    output_directory: Path
    interrupted: bool = False
    undo_path: Optional[Path] = None
    log_path: Optional[Path] = None
    journal_path: Optional[Path] = None

    def get_summary(self) -> str:
        """Get a summary of the organization."""
        lines = [
            "Organization Summary",
            "=" * 40,
            f"Total files processed: {self.total_files}",
            f"Successfully moved: {self.successful_moves}",
            f"Failed: {self.failed_moves}",
            f"Skipped: {self.skipped_files}",
            f"Output directory: {self.output_directory}",
        ]
        
        if self.failed_moves > 0:
            lines.append("\nFailed operations:")
            for op in self.operations:
                if not op.success:
                    lines.append(f"  - {op.source_path.name}: {op.error_message}")
        
        return "\n".join(lines)


class FileMover:
    """
    Handles moving files into the organized directory structure.
    """
    
    def __init__(
        self,
        output_directory: Path,
        dry_run: bool = False,
        copy_instead_of_move: bool = False,
        handle_duplicates: str = "rename",  # "rename", "skip", "overwrite"
        create_log: bool = True,
        create_undo: bool = True,
    ):
        """
        Initialize the file mover.

        Args:
            output_directory: Base directory for organized files.
            dry_run: If True, don't actually move files.
            copy_instead_of_move: If True, copy files instead of moving.
            handle_duplicates: How to handle files with same name.
            create_log: Whether to create a log file of operations.
            create_undo: Whether to write an undo script after organizing.
        """
        self.output_directory = Path(output_directory)
        self.dry_run = dry_run
        self.copy_instead_of_move = copy_instead_of_move
        self.handle_duplicates = handle_duplicates
        self.create_log = create_log
        self.create_undo = create_undo

        self._operations: list[MoveOperation] = []
        self._journal_handle = None  # open file appended to as each move succeeds
        self._journal_path: Optional[Path] = None
    
    def organize_file(
        self,
        source_path: Path,
        category_path: str,
        confidence: float
    ) -> MoveOperation:
        """
        Move/copy a single file to its category folder.
        
        Args:
            source_path: Path to the source file.
            category_path: Category path like "School/University/CS246".
            confidence: Classification confidence score.
            
        Returns:
            MoveOperation with the result.
        """
        source_path = Path(source_path)
        
        # Build destination path
        dest_dir = self.output_directory / category_path
        dest_path = dest_dir / source_path.name
        
        # Handle duplicates
        if dest_path.exists():
            if self.handle_duplicates == "skip":
                operation = MoveOperation(
                    source_path=source_path,
                    destination_path=dest_path,
                    category_path=category_path,
                    confidence=confidence,
                    success=False,
                    error_message="File already exists (skipped)"
                )
                self._operations.append(operation)
                return operation
            
            elif self.handle_duplicates == "rename":
                dest_path = self._get_unique_path(dest_path)
            
            # "overwrite" just proceeds normally
        
        operation = MoveOperation(
            source_path=source_path,
            destination_path=dest_path,
            category_path=category_path,
            confidence=confidence
        )
        
        if self.dry_run:
            operation.success = True
            operation.error_message = "(dry run - no actual move)"
        else:
            try:
                # Create destination directory
                dest_dir.mkdir(parents=True, exist_ok=True)
                
                # Move or copy
                if self.copy_instead_of_move:
                    shutil.copy2(source_path, dest_path)
                else:
                    shutil.move(str(source_path), str(dest_path))
                
                operation.success = True
                
            except PermissionError:
                operation.success = False
                operation.error_message = "Permission denied"
            except FileNotFoundError:
                operation.success = False
                operation.error_message = "Source file not found"
            except Exception as e:
                operation.success = False
                operation.error_message = str(e)
        
        self._operations.append(operation)
        return operation
    
    def organize_files(
        self,
        classifications: list[tuple[Path, str, float]],
        show_progress: bool = True
    ) -> OrganizationResult:
        """
        Organize multiple files based on their classifications.
        
        Args:
            classifications: List of (file_path, category_path, confidence) tuples.
            show_progress: Whether to show progress bar.
            
        Returns:
            OrganizationResult with all operations.
        """
        self._operations = []
        interrupted = False

        # Open a crash-safe journal: each successful move is recorded immediately,
        # so an interrupted run still has a complete record to undo from.
        self._open_journal()

        iterable = classifications
        if show_progress:
            try:
                from tqdm import tqdm
                iterable = tqdm(classifications, desc="Organizing files", total=len(classifications))
            except ImportError:
                pass

        try:
            for file_path, category_path, confidence in iterable:
                op = self.organize_file(file_path, category_path, confidence)
                if op.success and not self.dry_run:
                    self._append_journal(op)
        except KeyboardInterrupt:
            interrupted = True

        self._close_journal()

        # Create result
        successful = sum(1 for op in self._operations if op.success)
        failed = sum(1 for op in self._operations if not op.success and "skip" not in (op.error_message or ""))
        skipped = sum(1 for op in self._operations if "skip" in (op.error_message or "").lower())

        result = OrganizationResult(
            total_files=len(self._operations),
            successful_moves=successful,
            failed_moves=failed,
            skipped_files=skipped,
            operations=self._operations.copy(),
            output_directory=self.output_directory,
            interrupted=interrupted,
            journal_path=self._journal_path,
        )

        # Finalize: write the log and undo script even if we were interrupted,
        # so a partial run is still fully recoverable.
        if not self.dry_run:
            if self.create_log:
                result.log_path = self._write_log(result)
            if self.create_undo and not self.copy_instead_of_move:
                result.undo_path = self.create_undo_script(result)

        return result
    
    def _get_unique_path(self, path: Path) -> Path:
        """Get a unique path by adding a number suffix."""
        if not path.exists():
            return path
        
        stem = path.stem
        suffix = path.suffix
        parent = path.parent
        
        counter = 1
        while True:
            new_name = f"{stem}_{counter}{suffix}"
            new_path = parent / new_name
            if not new_path.exists():
                return new_path
            counter += 1
            
            if counter > 1000:  # Safety limit
                raise ValueError(f"Too many duplicates for {path}")
    
    # ---- crash-safe journal --------------------------------------------------

    def _open_journal(self) -> None:
        """Open an append-only journal so each completed move is durably recorded."""
        self._journal_handle = None
        self._journal_path = None
        if self.dry_run or not self.create_log:
            return
        try:
            log_dir = self.output_directory / "_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._journal_path = log_dir / f"journal_{timestamp}.jsonl"
            self._journal_handle = open(self._journal_path, "a", encoding="utf-8")
        except OSError:
            self._journal_handle = None
            self._journal_path = None

    def _append_journal(self, op: MoveOperation) -> None:
        if self._journal_handle is None:
            return
        try:
            self._journal_handle.write(json.dumps({
                "source": str(op.source_path),
                "destination": str(op.destination_path),
                "category": op.category_path,
                "copy": self.copy_instead_of_move,
            }) + "\n")
            self._journal_handle.flush()
            os.fsync(self._journal_handle.fileno())  # survive a hard crash
        except OSError:
            pass

    def _close_journal(self) -> None:
        if self._journal_handle is not None:
            try:
                self._journal_handle.close()
            except OSError:
                pass
            self._journal_handle = None

    def _write_log(self, result: OrganizationResult) -> Path:
        """Write a log file of operations. Returns the log file path."""
        log_dir = self.output_directory / "_logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"organization_log_{timestamp}.json"

        log_data = {
            "timestamp": timestamp,
            "summary": {
                "total_files": result.total_files,
                "successful_moves": result.successful_moves,
                "failed_moves": result.failed_moves,
                "skipped_files": result.skipped_files,
            },
            "operations": [
                {
                    "source": str(op.source_path),
                    "destination": str(op.destination_path),
                    "category": op.category_path,
                    "confidence": op.confidence,
                    "success": op.success,
                    "error": op.error_message,
                }
                for op in result.operations
            ]
        }
        
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

        return log_file

    def create_undo_script(self, result: OrganizationResult) -> Path:
        """
        Create a script to undo the organization.
        
        Returns:
            Path to the undo script.
        """
        script_dir = self.output_directory / "_logs"
        script_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_file = script_dir / f"undo_{timestamp}.sh"
        
        lines = [
            "#!/bin/bash",
            "# Undo script for file organization",
            f"# Generated: {timestamp}",
            "",
            "set -e",
            "",
        ]
        
        for op in result.operations:
            if op.success and not self.copy_instead_of_move:
                # Create command to move file back
                source = str(op.destination_path).replace("'", "'\\''")
                dest = str(op.source_path).replace("'", "'\\''")
                lines.append(f"mv '{source}' '{dest}'")
        
        lines.append("")
        lines.append('echo "Undo complete!"')
        
        with open(script_file, 'w') as f:
            f.write('\n'.join(lines))
        
        # Make executable
        os.chmod(script_file, 0o755)
        
        return script_file
