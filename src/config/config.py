"""
Configuration module that exposes all config variables.
This file helps maintain backward compatibility with code that imports `config`.
"""

# Re-export all configuration from the various config files
from pathlib import Path
import os

from .elasticsearch_config import (
    ELASTICSEARCH_HOSTS,
    ELASTICSEARCH_CLOUD_ID,
    ELASTICSEARCH_API_KEY,
    ELASTICSEARCH_USERNAME,
    ELASTICSEARCH_PASSWORD,
    INDEX_NAME,
    INDEX_SHARDS,
    INDEX_REPLICAS,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_MIN_SIMILARITY,
    ENABLE_CACHE,
    CACHE_TTL
)

from .prompts import GEMINI_CONFIG

# Define paths and other constants
BASE_DIR = Path(__file__).resolve().parents[2]
UPLOAD_DIR = BASE_DIR / 'uploads'
THUMBNAIL_DIR = BASE_DIR / 'thumbnails'
METADATA_DIR = BASE_DIR / 'metadata'

# Expose all config vars as attributes of this module
__all__ = [
    'ELASTICSEARCH_HOSTS',
    'ELASTICSEARCH_CLOUD_ID',
    'ELASTICSEARCH_API_KEY',
    'ELASTICSEARCH_USERNAME',
    'ELASTICSEARCH_PASSWORD',
    'INDEX_NAME',
    'INDEX_SHARDS',
    'INDEX_REPLICAS',
    'DEFAULT_SEARCH_LIMIT',
    'DEFAULT_MIN_SIMILARITY',
    'ENABLE_CACHE',
    'CACHE_TTL',
    'GEMINI_CONFIG',
    'BASE_DIR',
    'UPLOAD_DIR',
    'THUMBNAIL_DIR',
    'METADATA_DIR'
] 