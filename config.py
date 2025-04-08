"""
Configuration settings for the Archivist application.
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
THUMBNAIL_DIR = BASE_DIR / "thumbnails"
METADATA_DIR = BASE_DIR / "metadata"
CACHE_DIR = BASE_DIR / "cache"
INDEX_PATH = BASE_DIR / "vectors"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [UPLOAD_DIR, THUMBNAIL_DIR, METADATA_DIR, CACHE_DIR, INDEX_PATH, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# API settings
DEBUG = os.environ.get("DEBUG", "false").lower() == "true"
API_HOST = "0.0.0.0"  # Allow access from all IPs
API_PORT = int(os.environ.get("API_PORT", "8000"))

# Image processing settings
IMAGE_SIZE = 256  # Size for thumbnails
EMBEDDING_DIM = 512
BATCH_SIZE = 32

# Model configurations
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# Vector store configurations
N_CLUSTERS = 10
INDEX_TYPE = "IVFFlat"
NLIST = 100  # Number of clusters for IVF index

# Search settings
DEFAULT_SEARCH_LIMIT = 20
DEFAULT_MIN_SIMILARITY = 0.1

# Elasticsearch settings
ELASTICSEARCH_HOSTS = os.environ.get("ELASTICSEARCH_HOSTS", "http://localhost:9200").split(",")
ELASTICSEARCH_INDEX = os.environ.get("ELASTICSEARCH_INDEX", "archivist")
ELASTICSEARCH_CLOUD_ID = os.environ.get("ELASTICSEARCH_CLOUD_ID")
ELASTICSEARCH_API_KEY = os.environ.get("ELASTICSEARCH_API_KEY")
ELASTICSEARCH_USERNAME = os.environ.get("ELASTICSEARCH_USERNAME")
ELASTICSEARCH_PASSWORD = os.environ.get("ELASTICSEARCH_PASSWORD")

# Cache settings
ENABLE_CACHE = os.environ.get("ENABLE_CACHE", "false").lower() == "true"
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL = int(os.environ.get("CACHE_TTL", 3600))  # Default cache expiration: 1 hour 