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

from src.utils.logging_config import configure_logging

# Load environment variables from .env file if it exists
dotenv.load_dotenv()

# Configure logging at application startup
configure_logging()

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
            "origins": "*",  # Allow all origins in development
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
            "allow_headers": ["Content-Type", "Authorization", "X-Requested-With", "Accept", "Origin"],
            "expose_headers": ["Content-Type", "Authorization", "X-Total-Count"],
            "supports_credentials": True,
            "max_age": 86400
        }
    })
    
    # Add CORS preflight route handler for all routes
    @app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
    @app.route('/<path:path>', methods=['OPTIONS'])
    def options_handler(path):
        return '', 200
    
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

def start_app():
    """Start the Flask application."""
    app = create_app()
    # Ensure required directories exist
    ensure_dir(os.path.join(UPLOAD_DIR))
    ensure_dir(os.path.join(THUMBNAIL_DIR))
    ensure_dir(os.path.join(METADATA_DIR))
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=8080)

if __name__ == '__main__':
    start_app() 