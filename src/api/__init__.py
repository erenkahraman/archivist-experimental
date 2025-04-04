from flask import Blueprint, jsonify, Flask
from flask_cors import CORS
import logging
import os
import dotenv
import sys

# Add the project root to the Python path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.search_engine import SearchEngine

# Load environment variables from .env file
dotenv.load_dotenv()

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Control logging verbosity
DEBUG = os.environ.get('DEBUG', 'false').lower() == 'true'

# Set default logging to WARNING for production unless DEBUG is enabled
if not DEBUG:
    # Only show warnings and errors in production
    logging.getLogger().setLevel(logging.WARNING)  # Root logger
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('src').setLevel(logging.WARNING)
    # Only keep ERROR level logs for most libraries
    logging.getLogger('urllib3').setLevel(logging.ERROR)
    logging.getLogger('requests').setLevel(logging.ERROR)

# Get Gemini API key from environment variable
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("No Gemini API key found in environment variables")
else:
    # Mask API key for secure logging
    masked_key = GEMINI_API_KEY[:4] + "..." + GEMINI_API_KEY[-4:] if len(GEMINI_API_KEY) >= 8 else "INVALID_KEY"
    logger.info(f"Using Gemini API key: {masked_key}")

# Initialize search engine with Gemini API key - shared instance
search_engine = SearchEngine(gemini_api_key=GEMINI_API_KEY)

# Export the search_engine instance for use in app.py
__all__ = ['api', 'search_engine', 'create_app', 'DEBUG']

# Create a Flask Blueprint
api = Blueprint('api', __name__)

# Register basic error handlers
from werkzeug.exceptions import BadRequest

@api.errorhandler(BadRequest)
def handle_bad_request(e):
    return jsonify({'error': str(e)}), 400

# Basic test route
@api.route('/')
def home():
    return 'API is running'

# Import routes after api is created to avoid circular imports
from .routes import image_routes, search_routes, settings_routes

def create_app():
    """
    Create and configure the Flask application
    """
    app = Flask(__name__)
    
    # Configure CORS to allow requests from frontend
    CORS(app, resources={
        r"/api/*": {
            "origins": [
                "http://localhost:3000", 
                "http://localhost:5000",
                "http://127.0.0.1:3000",
                "http://127.0.0.1:5000"
            ],
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": [
                "Content-Type", 
                "Authorization", 
                "Accept",
                "Cache-Control"
            ]
        }
    })
    
    # Register the API blueprint
    app.register_blueprint(api, url_prefix='/api')
    
    return app
