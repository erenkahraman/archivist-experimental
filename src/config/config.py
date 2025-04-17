"""
Configuration module that exposes all config variables.
"""

# Re-export all configuration from the various config files
from pathlib import Path
import os

from .prompts import GEMINI_CONFIG, IMAGE_SIZE, THUMBNAIL_QUALITY, IMAGE_FORMATS

# Define paths and other constants
BASE_DIR = Path(__file__).resolve().parents[2]
UPLOAD_DIR = BASE_DIR / 'uploads'
THUMBNAIL_DIR = BASE_DIR / 'thumbnails'
METADATA_DIR = BASE_DIR / 'metadata'

# Search settings
DEFAULT_SEARCH_LIMIT = 10
DEFAULT_MIN_SIMILARITY = 0.3
MAX_SEARCH_RESULTS = 50

# Expose all config vars as attributes of this module
__all__ = [
    'GEMINI_CONFIG',
    'BASE_DIR',
    'UPLOAD_DIR',
    'THUMBNAIL_DIR',
    'METADATA_DIR',
    'DEFAULT_SEARCH_LIMIT',
    'DEFAULT_MIN_SIMILARITY',
    'MAX_SEARCH_RESULTS',
    'IMAGE_SIZE',
    'THUMBNAIL_QUALITY',
    'IMAGE_FORMATS'
] 