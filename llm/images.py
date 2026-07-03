# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright (c) 2026 George Zhang — https://github.com/TheYellowDuck

"""
Prepare image files for vision-capable models.

Loads an image, downscales it (vision token cost scales with resolution), and
returns a provider-neutral image part. Returns None whenever the image can't be
used, so the caller can fall back to filename/OCR text.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Optional

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


# Extensions we treat as images worth sending to a vision model.
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}


def is_image_path(path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def load_image_part(
    path,
    *,
    max_dim: int = 1024,
    quality: int = 80,
    max_source_bytes: int = 25_000_000,
) -> Optional[dict]:
    """
    Return a neutral image part {"type":"image","media_type":...,"data":<base64>}
    for the given file, or None if it can't be prepared.

    Everything is re-encoded to JPEG after downscaling to bound the payload, so
    a 12 MP photo doesn't cost a fortune in image tokens.
    """
    if not _PIL_AVAILABLE:
        return None

    p = Path(path)
    try:
        if not p.is_file() or p.stat().st_size > max_source_bytes:
            return None
        with Image.open(p) as im:
            im = im.convert("RGB")          # flattens RGBA/palette; first frame of GIFs
            im.thumbnail((max_dim, max_dim))  # in-place downscale, keeps aspect ratio
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=quality)
        data = base64.standard_b64encode(buf.getvalue()).decode("ascii")
        return {"type": "image", "media_type": "image/jpeg", "data": data}
    except Exception:
        # Corrupt/unsupported image, permission error, etc. — fall back to text.
        return None
