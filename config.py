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

# API settings
API_HOST = "0.0.0.0"  # Allow access from all IPs
API_PORT = 5000

# Model configurations
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
IMAGE_SIZE = 256  # Size for thumbnails
EMBEDDING_DIM = 512
BATCH_SIZE = 32

# Vector store configurations
N_CLUSTERS = 10  # Changed from 100 to 10 as the maximum number of clusters
INDEX_TYPE = "IVFFlat"
NLIST = 100  # Number of clusters for IVF index 