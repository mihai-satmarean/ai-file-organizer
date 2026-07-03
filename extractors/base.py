# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright (c) 2026 George Zhang — https://github.com/TheYellowDuck

"""
Base content extractor interface.
All extractors inherit from this class.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import mimetypes


@dataclass
class ExtractedContent:
    """Container for extracted file content."""
    file_path: Path
    file_name: str
    extension: str
    text_content: str
    metadata: dict
    extraction_success: bool
    error_message: Optional[str] = None
    
    @property
    def combined_content(self) -> str:
        """Combine filename and content for classification."""
        # Filename often contains useful classification hints
        name_without_ext = self.file_path.stem
        # Replace common separators with spaces
        clean_name = name_without_ext.replace('_', ' ').replace('-', ' ').replace('.', ' ')
        
        # Combine: weighted towards content but filename included
        parts = [clean_name]
        if self.text_content:
            # Limit content length to avoid overwhelming the embedding
            content_preview = self.text_content[:2000]
            parts.append(content_preview)
        
        return " ".join(parts)
    
    @property
    def is_empty(self) -> bool:
        """Check if no meaningful content was extracted."""
        return not self.text_content.strip()


class BaseExtractor(ABC):
    """Abstract base class for content extractors."""
    
    # File extensions this extractor handles
    SUPPORTED_EXTENSIONS: set[str] = set()
    
    # MIME types this extractor handles
    SUPPORTED_MIME_TYPES: set[str] = set()
    
    @classmethod
    def can_handle(cls, file_path: Path) -> bool:
        """Check if this extractor can handle the given file."""
        extension = file_path.suffix.lower()
        if extension in cls.SUPPORTED_EXTENSIONS:
            return True
        
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type and mime_type in cls.SUPPORTED_MIME_TYPES:
            return True
        
        return False
    
    @abstractmethod
    def extract(self, file_path: Path) -> ExtractedContent:
        """
        Extract content from the file.
        
        Args:
            file_path: Path to the file to extract content from.
            
        Returns:
            ExtractedContent object with the extracted text and metadata.
        """
        pass
    
    def _create_error_result(self, file_path: Path, error: str) -> ExtractedContent:
        """Helper to create an error result."""
        return ExtractedContent(
            file_path=file_path,
            file_name=file_path.name,
            extension=file_path.suffix.lower(),
            text_content="",
            metadata={},
            extraction_success=False,
            error_message=error
        )
    
    def _create_success_result(
        self, 
        file_path: Path, 
        text_content: str,
        metadata: Optional[dict] = None
    ) -> ExtractedContent:
        """Helper to create a success result."""
        return ExtractedContent(
            file_path=file_path,
            file_name=file_path.name,
            extension=file_path.suffix.lower(),
            text_content=text_content,
            metadata=metadata or {},
            extraction_success=True
        )
