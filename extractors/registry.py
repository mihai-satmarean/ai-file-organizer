"""
Content extractor registry.
Manages all available extractors and routes files to the appropriate one.
"""

from pathlib import Path
from typing import Optional
from extractors.base import BaseExtractor, ExtractedContent
from extractors.text import TextExtractor
from extractors.pdf import PDFExtractor
from extractors.docx import DocxExtractor
from extractors.image import ImageExtractor, ImageExtractorNoOCR
from extractors.spreadsheet import SpreadsheetExtractor
from extractors.pptx import PowerPointExtractor


class ExtractorRegistry:
    """
    Registry of all available content extractors.
    Automatically selects the right extractor for each file type.
    """
    
    def __init__(self, enable_ocr: bool = True):
        """
        Initialize the registry with available extractors.
        
        Args:
            enable_ocr: Whether to enable OCR for images (slower but more accurate)
        """
        self.extractors: list[BaseExtractor] = [
            TextExtractor(),
            PDFExtractor(),
            DocxExtractor(),
            ImageExtractor() if enable_ocr else ImageExtractorNoOCR(),
            SpreadsheetExtractor(),
            PowerPointExtractor(),
        ]
        
        # Build extension lookup for quick matching
        self._extension_map: dict[str, BaseExtractor] = {}
        for extractor in self.extractors:
            for ext in extractor.SUPPORTED_EXTENSIONS:
                self._extension_map[ext.lower()] = extractor
    
    def get_extractor(self, file_path: Path) -> Optional[BaseExtractor]:
        """Get the appropriate extractor for a file."""
        extension = file_path.suffix.lower()
        
        # Quick lookup by extension
        if extension in self._extension_map:
            return self._extension_map[extension]
        
        # Fallback to checking each extractor
        for extractor in self.extractors:
            if extractor.can_handle(file_path):
                return extractor
        
        return None
    
    def can_extract(self, file_path: Path) -> bool:
        """Check if any extractor can handle this file."""
        return self.get_extractor(file_path) is not None
    
    def extract(self, file_path: Path) -> ExtractedContent:
        """
        Extract content from a file using the appropriate extractor.
        
        Args:
            file_path: Path to the file to extract content from.
            
        Returns:
            ExtractedContent object. If no extractor is available,
            returns a result with filename-only content.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return ExtractedContent(
                file_path=file_path,
                file_name=file_path.name,
                extension=file_path.suffix.lower(),
                text_content="",
                metadata={},
                extraction_success=False,
                error_message="File not found"
            )
        
        extractor = self.get_extractor(file_path)
        
        if extractor:
            return extractor.extract(file_path)
        
        # No extractor available - return filename-only content
        return ExtractedContent(
            file_path=file_path,
            file_name=file_path.name,
            extension=file_path.suffix.lower(),
            text_content="",  # Will use filename via combined_content
            metadata={"unsupported_type": True},
            extraction_success=True  # Not an error, just no content
        )
    
    def get_supported_extensions(self) -> set[str]:
        """Get all supported file extensions."""
        extensions = set()
        for extractor in self.extractors:
            extensions.update(extractor.SUPPORTED_EXTENSIONS)
        return extensions
    
    def print_supported_formats(self) -> None:
        """Print all supported file formats."""
        print("Supported file formats:")
        print("=" * 40)
        for extractor in self.extractors:
            extractor_name = extractor.__class__.__name__
            extensions = sorted(extractor.SUPPORTED_EXTENSIONS)
            print(f"\n{extractor_name}:")
            print(f"  {', '.join(extensions)}")
