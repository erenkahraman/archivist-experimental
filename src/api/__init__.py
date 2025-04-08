from flask import Blueprint, jsonify
import logging
import os
import dotenv
import sys
from pathlib import Path

# Add the project root to the Python path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Load environment variables from .env file
dotenv.load_dotenv()

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Debug flag
DEBUG = os.environ.get('DEBUG', 'False').lower() in ('true', '1', 't', 'yes')

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

# Create API blueprint
api = Blueprint('api', __name__)

# Import SearchEngine to set up search functionality
try:
    # Import here to avoid circular imports
    from ..search_engine import SearchEngine
    
    # Initialize search engine
    search_engine = SearchEngine()
    
    if not search_engine.es_client or not search_engine.es_client.is_connected():
        logger.warning("Elasticsearch connection failed. Some search features will be limited.")
    else:
        logger.info("Elasticsearch connection successful.")
    
    # Log initialization state
    logger.info(f"API initialized with DEBUG={DEBUG}")
except Exception as e:
    logger.error(f"Error initializing search engine: {e}", exc_info=True)
    search_engine = None
    logger.warning("API initialized without search engine.")

# Register basic error handlers
from werkzeug.exceptions import BadRequest

@api.errorhandler(BadRequest)
def handle_bad_request(e):
    return jsonify({'error': str(e)}), 400

@api.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404

@api.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}", exc_info=True)
    return jsonify({"error": "Internal server error"}), 500

# Basic test route
@api.route('/')
def home():
    return 'API is running'

# Import routes after api is created to avoid circular imports
from .routes import image_routes, search_routes, settings_routes
