from flask import request, jsonify, send_from_directory, Blueprint
from pathlib import Path
import os
from werkzeug.utils import secure_filename
from src.core.search_engine import SearchEngine
import config
from werkzeug.exceptions import BadRequest
import dotenv
import logging
import time
import uuid

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables from .env file
dotenv.load_dotenv()

# Create a Flask Blueprint
api = Blueprint('api', __name__)

# Get Gemini API key from environment variable
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("No Gemini API key found in environment variables")
else:
    # Mask API key for secure logging
    masked_key = GEMINI_API_KEY[:4] + "..." + GEMINI_API_KEY[-4:] if len(GEMINI_API_KEY) >= 8 else "INVALID_KEY"
    logger.info(f"Using Gemini API key: {masked_key}")

# Initialize search engine with Gemini API key
search_engine = SearchEngine(gemini_api_key=GEMINI_API_KEY)

# Configure upload folder
UPLOAD_FOLDER = Path(__file__).parent.parent.parent / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@api.route('/thumbnails/<path:filename>')
def serve_thumbnail(filename):
    """Serve thumbnail images"""
    return send_from_directory(config.THUMBNAIL_DIR, filename)

@api.errorhandler(BadRequest)
def handle_bad_request(e):
    return jsonify({'error': str(e)}), 400

# Add a new endpoint to set the Gemini API key
@api.route('/set-gemini-key', methods=['POST'])
def set_gemini_key():
    """
    Set or update the Gemini API key
    
    Expects:
        - api_key: The Gemini API key
        
    Returns:
        - JSON response with success or error message
    """
    try:
        data = request.json
        if not data or 'api_key' not in data:
            return jsonify({'error': 'API key is required'}), 400
            
        api_key = data['api_key']
        
        if not api_key or len(api_key.strip()) == 0:
            return jsonify({'error': 'API key cannot be empty'}), 400
            
        # Log securely with masked key
        masked_key = api_key[:4] + "..." + api_key[-4:] if len(api_key) >= 8 else "INVALID_KEY"
        logger.info(f"Setting new Gemini API key: {masked_key}")
        
        # Update the API key in the search engine
        search_engine.set_gemini_api_key(api_key)
        
        return jsonify({'status': 'success', 'message': 'Gemini API key updated successfully'}), 200
    except Exception as e:
        logger.error(f"Error setting Gemini API key: {str(e)}")
        return jsonify({'error': 'Failed to update API key'}), 500

@api.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle file upload with proper validation and error handling.
    
    Expects:
        - file: The image file to upload
        
    Returns:
        JSON response with metadata
    """
    try:
        # Check if the post request has the file part
        if 'file' not in request.files:
            logger.error("No file part in request")
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['file']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            logger.error("No selected file")
            return jsonify({'error': 'No selected file'}), 400
            
        if file and allowed_file(file.filename):
            # Generate a unique filename to avoid collisions
            import uuid
            import time
            
            # Get file extension
            ext = os.path.splitext(file.filename)[1].lower()
            
            # Create a unique filename with timestamp
            unique_filename = f"{uuid.uuid4()}_{int(time.time())}{ext}"
            
            # Save the file
            file_path = config.UPLOAD_DIR / unique_filename
            file.save(file_path)
            
            logger.info(f"File saved to: {file_path}")
            
            # Process the image
            metadata = search_engine.process_image(file_path)
            
            if metadata:
                # Ensure the metadata has the correct path
                if 'original_path' not in metadata or not metadata['original_path']:
                    metadata['original_path'] = str(file_path)
                
                # Also add a relative path for frontend use
                if 'path' not in metadata:
                    metadata['path'] = f"uploads/{unique_filename}"
                
                logger.info(f"Image processed successfully: {metadata.get('original_path')}")
                return jsonify(metadata), 200
            else:
                logger.error(f"Failed to process image: {file_path}")
                return jsonify({'error': 'Failed to process image'}), 500
        else:
            logger.error(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Invalid file type'}), 400
    except Exception as e:
        print(f"Upload error: {str(e)}")
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api.route('/images', methods=['GET'])
def get_images():
    try:
        # Only return valid metadata
        valid_metadata = {
            path: data for path, data in search_engine.metadata.items()
            if data and 'thumbnail_path' in data and 'patterns' in data
        }
        return jsonify(list(valid_metadata.values()))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/search', methods=['POST'])
def search():
    """
    Advanced search endpoint that handles complex queries with pattern and color matching.
    
    Accepts the following parameters:
    - query: Main search query (required)
      - Can include color terms (e.g., "red", "blue") 
      - Can include pattern terms (e.g., "paisley", "floral")
      - Can be compound queries (e.g., "red paisley", "blue floral")
    - limit: Maximum number of results (optional, default: 20)
    - min_similarity: Minimum similarity score threshold (optional, default: 0.1)
    
    Returns:
    - List of metadata with similarity scores, sorted by relevance
    """
    try:
        # Get search parameters
        data = request.json or {}
        query = data.get('query', '').strip()
        limit = min(int(data.get('limit', 20)), 100)  # Cap at 100 results
        min_similarity = max(0.0, min(float(data.get('min_similarity', 0.1)), 1.0))  # Between 0 and 1
        
        # Validate query
        if not query:
            return jsonify({'error': 'Search query is required'}), 400
            
        # Log search request
        logger.info(f"Search request: query='{query}', limit={limit}, min_similarity={min_similarity}")
        
        # Perform search using the enhanced search function
        results = search_engine.search(query, k=limit)
        
        # Filter by minimum similarity if specified
        if min_similarity > 0:
            results = [r for r in results if r.get('similarity', 0) >= min_similarity]
        
        # Format the results for the response
        formatted_results = []
        for result in results:
            # Create a clean copy without internal fields
            item = {
                'id': result.get('id'),
                'filename': result.get('filename'),
                'path': result.get('path'),
                'thumbnail_path': result.get('thumbnail_path'),
                'similarity': result.get('similarity', 0.0),
                'timestamp': result.get('timestamp'),
            }
            
            # Include pattern information
            if 'patterns' in result:
                item['pattern'] = {
                    'primary': result['patterns'].get('primary_pattern', 'Unknown'),
                    'confidence': result['patterns'].get('pattern_confidence', 0.0),
                    'secondary': [p.get('name') for p in result['patterns'].get('secondary_patterns', [])],
                    'elements': [e.get('name') if isinstance(e, dict) else str(e) for e in result['patterns'].get('elements', [])]
                }
                
                # Include style keywords
                item['style_keywords'] = result['patterns'].get('style_keywords', [])
                
                # Include prompt if available
                if 'prompt' in result['patterns'] and 'final_prompt' in result['patterns']['prompt']:
                    item['prompt'] = result['patterns']['prompt']['final_prompt']
            
            # Include color information
            if 'colors' in result:
                item['colors'] = []
                for color in result['colors'].get('dominant_colors', [])[:5]:  # Top 5 colors
                    item['colors'].append({
                        'name': color.get('name', ''),
                        'hex': color.get('hex', ''),
                        'proportion': color.get('proportion', 0.0)
                    })
            
            # Add score components if available (for debugging)
            if 'pattern_score' in result and 'color_score' in result and 'other_score' in result:
                item['score_components'] = {
                    'pattern_score': result.get('pattern_score', 0.0),
                    'color_score': result.get('color_score', 0.0),
                    'other_score': result.get('other_score', 0.0)
                }
                
            formatted_results.append(item)
        
        return jsonify({
            'query': query,
            'result_count': len(formatted_results),
            'results': formatted_results
        })
        
    except Exception as e:
        logger.error(f"Error in search: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@api.route('/generate-prompt', methods=['POST'])
def generate_prompt():
    """
    Generate a prompt for an image
    
    Expects:
        - image_path: Path to the image
        
    Returns:
        - JSON response with prompt
    """
    try:
        data = request.json
        if not data or 'image_path' not in data:
            return jsonify({'error': 'Image path is required'}), 400
            
        image_path = data['image_path']
        
        # Get the metadata for the image
        metadata = search_engine.metadata.get(image_path)
        if not metadata:
            return jsonify({'error': 'Image not found'}), 404
            
        # Get the prompt from the metadata
        prompt = metadata.get('patterns', {}).get('prompt', {}).get('final_prompt', '')
        if not prompt:
            prompt = "Unable to generate prompt for this image"
            
        return jsonify({
            'prompt': prompt,
            'image_path': image_path
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Basic test route
@api.route('/')
def home():
    return 'API is running' 