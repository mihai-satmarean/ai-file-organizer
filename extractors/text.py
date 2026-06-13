"""
Text file extractor for plain text, markdown, and code files.
"""

from pathlib import Path
import chardet
from .base import BaseExtractor, ExtractedContent


class TextExtractor(BaseExtractor):
    """Extract content from plain text and code files."""
    
    SUPPORTED_EXTENSIONS = {
        # Plain text
        '.txt', '.text', '.md', '.markdown', '.rst', '.rtf',
        # Code files
        '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.h', '.hpp',
        '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala',
        '.html', '.htm', '.css', '.scss', '.sass', '.less',
        '.json', '.yaml', '.yml', '.xml', '.toml', '.ini', '.cfg', '.conf',
        '.sh', '.bash', '.zsh', '.fish', '.ps1', '.bat', '.cmd',
        '.sql', '.graphql', '.proto',
        '.r', '.R', '.m', '.mat',  # R and MATLAB
        '.tex', '.bib',  # LaTeX
        '.csv', '.tsv',  # Data files
        '.log', '.diff', '.patch',
        # Config/misc
        '.env', '.gitignore', '.dockerignore', 'Makefile', 'Dockerfile',
        '.vue', '.svelte',
    }
    
    SUPPORTED_MIME_TYPES = {
        'text/plain',
        'text/markdown',
        'text/x-python',
        'text/javascript',
        'text/html',
        'text/css',
        'application/json',
        'application/xml',
        'text/csv',
    }
    
    # Maximum file size to read (10 MB)
    MAX_FILE_SIZE = 10 * 1024 * 1024
    
    def extract(self, file_path: Path) -> ExtractedContent:
        """Extract text content from a text-based file."""
        try:
            # Check file size
            file_size = file_path.stat().st_size
            if file_size > self.MAX_FILE_SIZE:
                return self._create_error_result(
                    file_path, 
                    f"File too large: {file_size / 1024 / 1024:.1f} MB"
                )
            
            if file_size == 0:
                return self._create_success_result(
                    file_path,
                    "",
                    {"size": 0, "encoding": "unknown"}
                )
            
            # Try to detect encoding
            with open(file_path, 'rb') as f:
                raw_data = f.read()
            
            detected = chardet.detect(raw_data)
            encoding = detected.get('encoding', 'utf-8') or 'utf-8'
            
            # Decode content
            try:
                text_content = raw_data.decode(encoding)
            except (UnicodeDecodeError, LookupError):
                # Fallback to utf-8 with error handling
                text_content = raw_data.decode('utf-8', errors='replace')
            
            # Clean up content
            text_content = self._clean_text(text_content)
            
            return self._create_success_result(
                file_path,
                text_content,
                {
                    "size": file_size,
                    "encoding": encoding,
                    "confidence": detected.get('confidence', 0),
                    "lines": text_content.count('\n') + 1
                }
            )
            
        except PermissionError:
            return self._create_error_result(file_path, "Permission denied")
        except Exception as e:
            return self._create_error_result(file_path, str(e))
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove null bytes
        text = text.replace('\x00', '')
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # Remove excessive whitespace while preserving structure
        lines = text.split('\n')
        cleaned_lines = [line.rstrip() for line in lines]
        return '\n'.join(cleaned_lines)
