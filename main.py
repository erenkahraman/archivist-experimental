from src.api import start_api
from pathlib import Path
import config

def main():
    # Create necessary directories
    config.THUMBNAIL_DIR.mkdir(exist_ok=True)
    config.INDEX_PATH.mkdir(exist_ok=True)
    
    # Start the API server
    start_api()

if __name__ == "__main__":
    main() 