# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright (c) 2026 George Zhang — https://github.com/TheYellowDuck

"""Content extraction package for the file organizer."""

from extractors.base import BaseExtractor, ExtractedContent
from extractors.registry import ExtractorRegistry
from extractors.text import TextExtractor
from extractors.pdf import PDFExtractor
from extractors.docx import DocxExtractor
from extractors.image import ImageExtractor, ImageExtractorNoOCR
from extractors.spreadsheet import SpreadsheetExtractor
from extractors.pptx import PowerPointExtractor

__all__ = [
    'BaseExtractor',
    'ExtractedContent',
    'ExtractorRegistry',
    'TextExtractor',
    'PDFExtractor',
    'DocxExtractor',
    'ImageExtractor',
    'ImageExtractorNoOCR',
    'SpreadsheetExtractor',
    'PowerPointExtractor',
]
