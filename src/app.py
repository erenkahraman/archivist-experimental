import logging
from flask import Flask
from flask_cors import CORS
from pathlib import Path
import os
import config
from .logging_config import configure_logging
from .search_engine import SearchEngine
import dotenv

# Load environment variables from .env file if it exists
dotenv.load_dotenv()

# Configure logging at application startup
configure_logging()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Suppress less important logs
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.setLevel(logging.WARNING)

# Get Gemini API key from environment variable
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    # Mask API key for secure logging
    masked_key = GEMINI_API_KEY[:4] + "..." + GEMINI_API_KEY[-4:] if len(GEMINI_API_KEY) >= 8 else "INVALID_KEY"
    logger.info(f"Gemini API key found in environment variables: {masked_key}")
else:
    logger.warning("No Gemini API key found in environment variables")
    GEMINI_API_KEY = None  # Don't use a default key

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Enable CORS for all routes
    CORS(app, resources={
        r"/*": {
            "origins": ["http://localhost:3000"],
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type"],
            "supports_credentials": True
        }
    })
    
    # Add a root route
    @app.route('/')
    def home():
        return 'Archivist server is running'
    
    # Import API routes
    from . import api
    
    # Register API routes with prefix
    app.register_blueprint(api.app, url_prefix='/api')
    
    # Ensure required directories exist
    ensure_directories()
    
    return app

def ensure_directories():
    """Ensure all required directories exist."""
    directories = [
        config.UPLOAD_DIR,
        config.THUMBNAIL_DIR,
        config.METADATA_DIR,
        Path(config.BASE_DIR) / "catalogs"  # Add catalogs directory for PantoneAnalyzer
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")

def start_app():
    """Start the Flask application."""
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=8000)

if __name__ == '__main__':
    start_app() 