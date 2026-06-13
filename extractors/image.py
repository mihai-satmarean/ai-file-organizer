"""
Image content extractor using OCR (pytesseract).
Also handles basic image metadata for classification.
"""

from pathlib import Path
from .base import BaseExtractor, ExtractedContent

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


class ImageExtractor(BaseExtractor):
    """Extract text content from images using OCR."""
    
    SUPPORTED_EXTENSIONS = {
        '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif', '.webp'
    }
    SUPPORTED_MIME_TYPES = {
        'image/png', 'image/jpeg', 'image/gif', 'image/bmp', 
        'image/tiff', 'image/webp'
    }
    
    # Whether to actually perform OCR (can be slow)
    ENABLE_OCR = True
    
    # Maximum image dimension for OCR
    MAX_DIMENSION = 4000
    
    def extract(self, file_path: Path) -> ExtractedContent:
        """Extract text from image using OCR and gather metadata."""
        if not PIL_AVAILABLE:
            return self._create_error_result(
                file_path,
                "Pillow not installed. Run: pip install Pillow"
            )
        
        try:
            with Image.open(str(file_path)) as img:
                # Get image metadata
                metadata = {
                    "width": img.width,
                    "height": img.height,
                    "format": img.format,
                    "mode": img.mode,
                }
                
                # Check for EXIF data
                exif_data = {}
                try:
                    if hasattr(img, '_getexif') and img._getexif():
                        exif = img._getexif()
                        # Extract useful EXIF fields
                        EXIF_TAGS = {
                            271: 'make',
                            272: 'model', 
                            306: 'datetime',
                            36867: 'datetime_original',
                        }
                        for tag_id, tag_name in EXIF_TAGS.items():
                            if tag_id in exif:
                                exif_data[tag_name] = str(exif[tag_id])
                except Exception:
                    pass
                
                if exif_data:
                    metadata["exif"] = exif_data
                
                # Perform OCR if available and enabled
                text_content = ""
                if self.ENABLE_OCR and TESSERACT_AVAILABLE:
                    try:
                        # Resize large images
                        process_img = img
                        if max(img.width, img.height) > self.MAX_DIMENSION:
                            ratio = self.MAX_DIMENSION / max(img.width, img.height)
                            new_size = (int(img.width * ratio), int(img.height * ratio))
                            process_img = img.resize(new_size, Image.Resampling.LANCZOS)
                        
                        # Convert to RGB if necessary
                        if process_img.mode not in ('RGB', 'L'):
                            process_img = process_img.convert('RGB')
                        
                        # Run OCR
                        text_content = pytesseract.image_to_string(process_img)
                        text_content = text_content.strip()
                        
                        if text_content:
                            metadata["ocr_performed"] = True
                            metadata["ocr_chars"] = len(text_content)
                        
                    except Exception as e:
                        metadata["ocr_error"] = str(e)
                
                # If no OCR text, use filename-based classification
                # The combined_content property will use the filename
                
                return self._create_success_result(
                    file_path,
                    text_content,
                    metadata
                )
                
        except Exception as e:
            return self._create_error_result(file_path, f"Image extraction error: {str(e)}")


class ImageExtractorNoOCR(ImageExtractor):
    """Image extractor without OCR - faster, just uses metadata."""
    ENABLE_OCR = False
