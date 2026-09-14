"""
Small utility functions for the Business Card OCR app.
"""

import os
import base64
from pathlib import Path

from config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB


def is_allowed_file(filename: str) -> bool:
    """Check if the file extension is in the allowed set."""
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXTENSIONS


def validate_file_size(file_bytes: bytes) -> bool:
    """Return True if the file is within the maximum allowed size."""
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    return len(file_bytes) <= max_bytes


def encode_image_to_base64(image_path: str) -> str:
    """Read an image file and return its base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_file_extension(filename: str) -> str:
    """Return the lowercase file extension including the dot."""
    return Path(filename).suffix.lower()


def clean_ocr_text(text: str) -> str:
    """Basic cleanup of raw OCR text: strip whitespace, collapse blank lines."""
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            cleaned.append(stripped)
    return "\n".join(cleaned)
