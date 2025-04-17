"""
Configuration package for the Pattern Analysis System.
"""

from .config import (
    BASE_DIR,
    UPLOAD_DIR,
    THUMBNAIL_DIR,
    METADATA_DIR,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_MIN_SIMILARITY,
    MAX_SEARCH_RESULTS
)

__all__ = [
    'BASE_DIR',
    'UPLOAD_DIR',
    'THUMBNAIL_DIR',
    'METADATA_DIR',
    'DEFAULT_SEARCH_LIMIT',
    'DEFAULT_MIN_SIMILARITY',
    'MAX_SEARCH_RESULTS'
]
