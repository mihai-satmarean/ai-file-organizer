# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright (c) 2026 George Zhang — https://github.com/TheYellowDuck

"""
Directory scanner for discovering files to organize.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Iterator, Optional
import os


@dataclass
class ScanResult:
    """Result of scanning a directory."""
    total_files: int
    supported_files: int
    skipped_files: int
    file_paths: list[Path]
    skipped_paths: list[Path]
    errors: list[str]


class DirectoryScanner:
    """
    Scans directories to find files for organization.
    """
    
    # Default directories to skip
    DEFAULT_SKIP_DIRS = {
        '.git', '.svn', '.hg',  # Version control
        'node_modules', '__pycache__', '.venv', 'venv', 'env',  # Dependencies
        '.idea', '.vscode', '.vs',  # IDE folders
        '$RECYCLE.BIN', 'System Volume Information',  # System folders
        '.Trash', '.Trashes',  # Trash
        'Thumbs.db', '.DS_Store',  # System files (as dirs)
    }
    
    # Default files to skip
    DEFAULT_SKIP_FILES = {
        '.DS_Store', 'Thumbs.db', 'desktop.ini',
        '.gitignore', '.gitattributes',
    }
    
    # Default extensions to skip
    DEFAULT_SKIP_EXTENSIONS = {
        '.tmp', '.temp', '.swp', '.swo',  # Temporary files
        '.lock', '.pid',  # Lock files
    }
    
    def __init__(
        self,
        skip_dirs: Optional[set[str]] = None,
        skip_files: Optional[set[str]] = None,
        skip_extensions: Optional[set[str]] = None,
        skip_hidden: bool = True,
        max_file_size_mb: float = 100,
        supported_extensions: Optional[set[str]] = None,
        ignore_patterns: Optional[list[str]] = None,
    ):
        """
        Initialize the scanner.

        Args:
            skip_dirs: Directory names to skip (merged with defaults).
            skip_files: File names to skip (merged with defaults).
            skip_extensions: Extensions to skip (merged with defaults).
            skip_hidden: Whether to skip hidden files/folders (starting with .).
            max_file_size_mb: Maximum file size to process in MB.
            supported_extensions: If provided, only include these extensions.
            ignore_patterns: Glob patterns (matched against filename and full path)
                of files to skip — e.g. from --exclude or a .organizerignore file.
        """
        self.skip_dirs = self.DEFAULT_SKIP_DIRS | (skip_dirs or set())
        self.skip_files = self.DEFAULT_SKIP_FILES | (skip_files or set())
        self.skip_extensions = self.DEFAULT_SKIP_EXTENSIONS | (skip_extensions or set())
        self.skip_hidden = skip_hidden
        self.max_file_size = max_file_size_mb * 1024 * 1024  # Convert to bytes
        self.supported_extensions = supported_extensions
        self.ignore_patterns = ignore_patterns or []
    
    def scan(
        self, 
        directory: Path, 
        recursive: bool = True
    ) -> ScanResult:
        """
        Scan a directory for files.
        
        Args:
            directory: Path to the directory to scan.
            recursive: Whether to scan subdirectories.
            
        Returns:
            ScanResult with discovered files.
        """
        directory = Path(directory)
        
        if not directory.exists():
            return ScanResult(
                total_files=0,
                supported_files=0,
                skipped_files=0,
                file_paths=[],
                skipped_paths=[],
                errors=[f"Directory does not exist: {directory}"]
            )
        
        if not directory.is_dir():
            return ScanResult(
                total_files=0,
                supported_files=0,
                skipped_files=0,
                file_paths=[],
                skipped_paths=[],
                errors=[f"Path is not a directory: {directory}"]
            )
        
        file_paths = []
        skipped_paths = []
        errors = []
        
        try:
            for file_path in self._iterate_files(directory, recursive):
                skip_reason = self._should_skip(file_path)
                
                if skip_reason:
                    skipped_paths.append(file_path)
                else:
                    file_paths.append(file_path)
                    
        except Exception as e:
            errors.append(f"Error scanning directory: {str(e)}")
        
        return ScanResult(
            total_files=len(file_paths) + len(skipped_paths),
            supported_files=len(file_paths),
            skipped_files=len(skipped_paths),
            file_paths=file_paths,
            skipped_paths=skipped_paths,
            errors=errors
        )
    
    def _iterate_files(
        self, 
        directory: Path, 
        recursive: bool
    ) -> Iterator[Path]:
        """Iterate through files in a directory."""
        try:
            for entry in os.scandir(directory):
                try:
                    if entry.is_file(follow_symlinks=False):
                        yield Path(entry.path)
                    elif entry.is_dir(follow_symlinks=False) and recursive:
                        # Check if directory should be skipped
                        dir_name = entry.name
                        
                        if dir_name in self.skip_dirs:
                            continue
                        if self.skip_hidden and dir_name.startswith('.'):
                            continue
                        
                        # Recurse into subdirectory
                        yield from self._iterate_files(Path(entry.path), recursive)
                        
                except PermissionError:
                    continue
                except Exception:
                    continue
                    
        except PermissionError:
            pass
        except Exception:
            pass
    
    def _should_skip(self, file_path: Path) -> Optional[str]:
        """
        Check if a file should be skipped.
        
        Returns:
            Reason for skipping, or None if file should be processed.
        """
        # Check filename
        if file_path.name in self.skip_files:
            return "skip_file"
        
        # Check hidden
        if self.skip_hidden and file_path.name.startswith('.'):
            return "hidden"
        
        # Check extension
        extension = file_path.suffix.lower()
        if extension in self.skip_extensions:
            return "skip_extension"
        
        # Check supported extensions filter
        if self.supported_extensions and extension not in self.supported_extensions:
            return "unsupported_extension"

        # Check user-provided ignore globs (filename or full path).
        if self.ignore_patterns:
            from fnmatch import fnmatch
            name = file_path.name
            full = str(file_path)
            for pattern in self.ignore_patterns:
                if fnmatch(name, pattern) or fnmatch(full, pattern):
                    return "excluded"
        
        # Check file size
        try:
            if file_path.stat().st_size > self.max_file_size:
                return "too_large"
        except Exception:
            return "stat_error"
        
        return None
    
    def scan_multiple(
        self, 
        directories: list[Path],
        recursive: bool = True
    ) -> ScanResult:
        """
        Scan multiple directories.
        
        Args:
            directories: List of directory paths to scan.
            recursive: Whether to scan subdirectories.
            
        Returns:
            Combined ScanResult.
        """
        all_file_paths = []
        all_skipped_paths = []
        all_errors = []
        
        for directory in directories:
            result = self.scan(directory, recursive)
            all_file_paths.extend(result.file_paths)
            all_skipped_paths.extend(result.skipped_paths)
            all_errors.extend(result.errors)
        
        return ScanResult(
            total_files=len(all_file_paths) + len(all_skipped_paths),
            supported_files=len(all_file_paths),
            skipped_files=len(all_skipped_paths),
            file_paths=all_file_paths,
            skipped_paths=all_skipped_paths,
            errors=all_errors
        )
