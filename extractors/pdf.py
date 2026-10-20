# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright (c) 2026 George Zhang — https://github.com/TheYellowDuck

"""
PDF content extractor using PyPDF2.
"""

from pathlib import Path
from .base import BaseExtractor, ExtractedContent

try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False


def _detect_digital_signature(reader) -> dict:
    """
    Detect cryptographic digital signatures in a PDF.

    Returns a dict with:
      - digital_signature: bool  — True if at least one /Sig field has a value (/V not None)
      - digital_signature_count: int — number of completed sig fields
      - digital_signature_empty_fields: int — sig placeholder fields without a value

    Uses the AcroForm field approach: a /Sig field exists iff there is a form
    field with /FT == /Sig; it is *completed* iff /V is not None.
    A plain image of a signature on the page produces no /Sig field.
    """
    try:
        fields = reader.get_fields() or {}
    except Exception:
        return {"digital_signature": False, "digital_signature_count": 0,
                "digital_signature_empty_fields": 0}

    signed = 0
    empty = 0
    for field in fields.values():
        if field.get("/FT") == "/Sig":
            if field.get("/V") is not None:
                signed += 1
            else:
                empty += 1

    return {
        "digital_signature": signed > 0,
        "digital_signature_count": signed,
        "digital_signature_empty_fields": empty,
    }


class PDFExtractor(BaseExtractor):
    """Extract text content from PDF files."""
    
    SUPPORTED_EXTENSIONS = {'.pdf'}
    SUPPORTED_MIME_TYPES = {'application/pdf'}
    
    # Maximum pages to extract
    MAX_PAGES = 100
    
    def extract(self, file_path: Path) -> ExtractedContent:
        """Extract text content from a PDF file."""
        if not PYPDF2_AVAILABLE:
            return self._create_error_result(
                file_path,
                "PyPDF2 not installed. Run: pip install PyPDF2"
            )
        
        try:
            reader = PdfReader(str(file_path))
            
            # Extract metadata
            metadata = {}
            if reader.metadata:
                metadata = {
                    "title": reader.metadata.get('/Title', ''),
                    "author": reader.metadata.get('/Author', ''),
                    "subject": reader.metadata.get('/Subject', ''),
                    "creator": reader.metadata.get('/Creator', ''),
                }
                # Clean None values
                metadata = {k: v for k, v in metadata.items() if v}
            
            num_pages = len(reader.pages)
            metadata["pages"] = num_pages

            # Digital signature detection (cryptographic /Sig fields only;
            # image-based signatures are not detectable this way)
            metadata.update(_detect_digital_signature(reader))
            
            # Extract text from pages
            text_parts = []
            pages_to_extract = min(num_pages, self.MAX_PAGES)
            
            for i in range(pages_to_extract):
                try:
                    page = reader.pages[i]
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                except Exception as e:
                    # Skip problematic pages but continue
                    continue
            
            if num_pages > self.MAX_PAGES:
                text_parts.append(f"\n[...{num_pages - self.MAX_PAGES} more pages not extracted...]")
            
            text_content = '\n\n'.join(text_parts)
            text_content = self._clean_pdf_text(text_content)
            
            # Include metadata in searchable content
            if metadata.get('title'):
                text_content = f"Title: {metadata['title']}\n\n{text_content}"
            
            return self._create_success_result(
                file_path,
                text_content,
                metadata
            )
            
        except Exception as e:
            return self._create_error_result(file_path, f"PDF extraction error: {str(e)}")
    
    def _clean_pdf_text(self, text: str) -> str:
        """Clean up extracted PDF text."""
        # Remove excessive whitespace
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
