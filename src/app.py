import logging
from flask import Flask
from flask_cors import CORS
import os
import sys
from pathlib import Path

# Add the project root to the path for imports to work
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dotenv
from src.config.config import UPLOAD_DIR, THUMBNAIL_DIR, METADATA_DIR

# Load environment variables from .env file if it exists
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Ensured directory exists: {directory}")

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Enable CORS for all routes with comprehensive permissions
    CORS(app, resources={
        r"/*": {
            "origins": ["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5000", "http://127.0.0.1:5000"],
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
            "allow_headers": ["Content-Type", "Authorization", "X-Requested-With", "Accept", "Origin"],
            "expose_headers": ["Content-Type", "Authorization", "X-Total-Count"],
            "supports_credentials": True,
            "max_age": 86400
        }
    })
    
    # Add a root route
    @app.route('/')
    def home():
        return 'Archivist server is running'
    
    # Import API blueprint
    from src.api import api
    
    # Register API routes with prefix
    app.register_blueprint(api, url_prefix='/api')
    
    # Ensure required directories exist
    ensure_directories()
    
    return app

def ensure_directories():
    """Ensure all required directories exist."""
    directories = [
        UPLOAD_DIR,
        THUMBNAIL_DIR,
        METADATA_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")

# Create the application instance
app = create_app()

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=8000) 