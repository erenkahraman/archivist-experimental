import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
THUMBNAIL_DIR = BASE_DIR / "thumbnails"

# Create directories if they don't exist
UPLOAD_DIR.mkdir(exist_ok=True)
THUMBNAIL_DIR.mkdir(exist_ok=True)

# File paths
INDEX_PATH = BASE_DIR / "vectors"
INDEX_PATH.mkdir(exist_ok=True)

# API settings
API_HOST = "0.0.0.0"  # Allow access from all IPs
API_PORT = 5000

# Model configurations
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
IMAGE_SIZE = 256  # Size for thumbnails
EMBEDDING_DIM = 512
BATCH_SIZE = 32

# Vector store configurations
N_CLUSTERS = 10  # Number of clusters for color analysis
INDEX_TYPE = "IVFFlat"
NLIST = 100  # Number of clusters for IVF index