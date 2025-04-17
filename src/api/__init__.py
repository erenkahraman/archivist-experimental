from flask import Blueprint, jsonify, Flask
from flask_cors import CORS
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

# Import necessary modules for initialization
try:
    # Import pattern analyzer
    from src.core.pattern_analyzer import PatternAnalyzer
    from src.search import search_engine
    
    # Ensure directories exist
    UPLOAD_DIR = Path(__file__).parent.parent.parent / "uploads"
    THUMBNAIL_DIR = Path(__file__).parent.parent.parent / "thumbnails"
    METADATA_DIR = Path(__file__).parent.parent.parent / "metadata"
    
    for dir_path in [UPLOAD_DIR, THUMBNAIL_DIR, METADATA_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {dir_path}")
    
    # Initialize PatternAnalyzer
    try:
        pattern_analyzer = PatternAnalyzer(api_key=GEMINI_API_KEY)
        logger.info("PatternAnalyzer initialized successfully")
        
        # Load any existing metadata
        metadata = pattern_analyzer.get_all_metadata()
        if metadata:
            # Configure search engine with metadata
            search_engine.set_metadata(metadata)
            logger.info(f"Search engine configured with {len(metadata)} items")
        else:
            logger.info("No existing metadata found to load into search engine")
    except Exception as e:
        logger.error(f"Failed to initialize PatternAnalyzer: {str(e)}")
    
    # Log initialization state
    logger.info(f"API initialized with DEBUG={DEBUG}")
except ImportError as e:
    logger.error(f"Error importing core modules: {str(e)}")
    logger.warning("API initialized without core functionality.")
except Exception as e:
    logger.error(f"Error initializing core components: {str(e)}", exc_info=True)
    logger.warning("API initialized with limited functionality.")

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
from .routes import search_routes, upload_routes, image_routes

# Register route blueprints with the main api blueprint
from .routes.search_routes import search_blueprint
from .routes.upload_routes import upload_blueprint
from .routes.image_routes import image_blueprint

api.register_blueprint(search_blueprint)
api.register_blueprint(upload_blueprint, url_prefix='/upload/')
api.register_blueprint(image_blueprint, url_prefix='/images/')

def create_app():
    """
    Create and configure the Flask application
    """
    app = Flask(__name__)
    
    # Configure CORS to allow requests from frontend
    CORS(app, resources={
        r"/*": {
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
                "Cache-Control",
                "X-Requested-With",
                "Origin"
            ],
            "supports_credentials": True,
            "max_age": 86400
        }
    })
    
    # Register the API blueprint
    app.register_blueprint(api, url_prefix='/api')
    
    return app
