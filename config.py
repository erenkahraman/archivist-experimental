import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
THUMBNAIL_DIR = BASE_DIR / "thumbnails"
METADATA_DIR = BASE_DIR / "metadata"

# Create directories if they don't exist
UPLOAD_DIR.mkdir(exist_ok=True)
THUMBNAIL_DIR.mkdir(exist_ok=True)
METADATA_DIR.mkdir(exist_ok=True)

# Model settings
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
IMAGE_SIZE = 256  # Size for thumbnails
EMBEDDING_DIM = 512
BATCH_SIZE = 32

# File paths
INDEX_PATH = BASE_DIR / "vectors"
INDEX_PATH.mkdir(exist_ok=True)

# Vector store configurations
N_CLUSTERS = 5
INDEX_TYPE = "IVFFlat"
NLIST = 100  # Number of clusters for IVF index

# Search settings
DEFAULT_MIN_SIMILARITY = 0.1  # Default minimum similarity threshold for search results

# API settings
API_HOST = "0.0.0.0"  # Allow access from all IPs
API_PORT = 5000

# Elasticsearch settings
ELASTICSEARCH_HOSTS = ["http://localhost:9200"]  # Default local Elasticsearch instance
ELASTICSEARCH_INDEX = os.environ.get("ELASTICSEARCH_INDEX", "archivist")
ELASTICSEARCH_CLOUD_ID = None
ELASTICSEARCH_API_KEY = None
ELASTICSEARCH_USERNAME = None
ELASTICSEARCH_PASSWORD = None

# Search settings
DEFAULT_SEARCH_LIMIT = 20
DEFAULT_MIN_SIMILARITY = 0.1

# Cache settings
ENABLE_CACHE = os.environ.get("ENABLE_CACHE", "false").lower() == "true"
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL = int(os.environ.get("CACHE_TTL", 3600))  # Default cache expiration: 1 hour

# Model configurations
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
IMAGE_SIZE = 256  # Size for thumbnails
EMBEDDING_DIM = 512
BATCH_SIZE = 32

# Vector store configurations
N_CLUSTERS = 10  # Changed from 100 to 10 as the maximum number of clusters
INDEX_TYPE = "IVFFlat"
NLIST = 100  # Number of clusters for IVF index 