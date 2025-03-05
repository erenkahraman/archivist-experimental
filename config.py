import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
THUMBNAIL_DIR = BASE_DIR / "thumbnails"

# Create directories if they don't exist
UPLOAD_DIR.mkdir(exist_ok=True)
THUMBNAIL_DIR.mkdir(exist_ok=True)

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

# API settings
API_HOST = "0.0.0.0"  # Tüm IP'lerden erişime izin ver
API_PORT = 5001

# Model configurations
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
IMAGE_SIZE = 256  # Size for thumbnails
EMBEDDING_DIM = 512
BATCH_SIZE = 32

# File paths
BASE_DIR = Path(__file__).parent
THUMBNAIL_DIR = BASE_DIR / "thumbnails"
INDEX_PATH = BASE_DIR / "vectors"
THUMBNAIL_DIR.mkdir(exist_ok=True)
INDEX_PATH.mkdir(exist_ok=True)

# Vector store configurations
N_CLUSTERS = 10  # Changed from 100 to 10 as the maximum number of clusters
INDEX_TYPE = "IVFFlat"
NLIST = 100  # Number of clusters for IVF index

# API configurations
API_HOST = "0.0.0.0"
API_PORT = 5001  # Changed from 5000 to 5001 