from .prompts import GEMINI_CONFIG
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

# Import config as a module for backward compatibility
from . import config

__all__ = [
    'config',  # Add config module to exports
    'GEMINI_CONFIG',
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
    'CACHE_TTL'
]
