from flask import Flask, request, jsonify, send_from_directory, Blueprint
from flask_cors import CORS
from pathlib import Path
import os
from werkzeug.utils import secure_filename
from .search_engine import SearchEngine
import config
from werkzeug.exceptions import BadRequest
import dotenv
import logging
from .analyzers.pantone_analyzer import PantoneAnalyzer
import time
import uuid

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables from .env file
dotenv.load_dotenv()

# Create a Flask Blueprint
app = Blueprint('api', __name__)

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

# Initialize Pantone analyzer
pantone_analyzer = PantoneAnalyzer()

# Configure upload folder
UPLOAD_FOLDER = Path(__file__).parent.parent / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/thumbnails/<path:filename>')
def serve_thumbnail(filename):
    """Serve thumbnail images"""
    return send_from_directory(config.THUMBNAIL_DIR, filename)

@app.errorhandler(BadRequest)
def handle_bad_request(e):
    return jsonify({'error': str(e)}), 400

# Add a new endpoint to set the Gemini API key
@app.route('/set-gemini-key', methods=['POST'])
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

@app.route('/upload', methods=['POST'])
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

@app.route('/images', methods=['GET'])
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

@app.route('/search', methods=['POST'])
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

@app.route('/generate-prompt', methods=['POST'])
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

@app.route('/pantone/catalogs', methods=['GET'])
def get_pantone_catalogs():
    """
    Get list of available Pantone catalogs
    
    Returns:
        JSON response with catalog names
    """
    try:
        catalogs = pantone_analyzer.get_available_catalogs()
        return jsonify({
            'status': 'success',
            'catalogs': catalogs
        }), 200
    except Exception as e:
        logger.error(f"Error getting Pantone catalogs: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/pantone/catalogs/info', methods=['GET'])
def get_pantone_catalogs_info():
    """
    Get detailed information about available Pantone catalogs
    
    Returns:
        JSON response with detailed catalog information
    """
    try:
        catalog_info = pantone_analyzer.get_catalog_info()
        return jsonify({
            'status': 'success',
            'catalog_info': catalog_info
        }), 200
    except Exception as e:
        logger.error(f"Error getting Pantone catalog info: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/pantone/upload-catalog', methods=['POST'])
def upload_pantone_catalog():
    """
    Upload a new Pantone catalog file
    
    Expects:
        - file: The catalog file (.cat format)
        - catalog_name: (Optional) Name for the catalog
        
    Returns:
        JSON response with upload status
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['file']
        if not file or not file.filename:
            return jsonify({'error': 'No selected file'}), 400
            
        if not file.filename.endswith('.cat'):
            return jsonify({'error': 'Invalid file format. Only .cat files are supported'}), 400

        # Create catalogs directory if it doesn't exist
        catalogs_dir = Path(__file__).parent.parent / "catalogs"
        catalogs_dir.mkdir(exist_ok=True)
        
        # Create a secure filename
        filename = secure_filename(file.filename)
        temp_file_path = catalogs_dir / f"temp_{int(time.time())}_{filename}"
        
        # Save the file to a temporary location first
        file.save(temp_file_path)
        logger.info(f"Saved catalog file to temporary location: {temp_file_path}")
        
        # Get catalog name from form data or use filename
        catalog_name = request.form.get('catalog_name')
        if not catalog_name:
            catalog_name = os.path.splitext(filename)[0]
        
        logger.info(f"Processing catalog: {catalog_name}")
            
        # Process the catalog
        result = pantone_analyzer.upload_catalog(str(temp_file_path), catalog_name)
        
        # Clean up the temporary file
        try:
            os.remove(temp_file_path)
        except Exception as e:
            logger.warning(f"Failed to remove temporary file {temp_file_path}: {str(e)}")
        
        if result.get('success'):
            return jsonify({
                'status': 'success',
                'message': result.get('message', f'Catalog uploaded successfully: {catalog_name}'),
                'catalog_name': catalog_name,
                'colors_count': result.get('colors_count', 0)
            }), 200
        else:
            logger.error(f"Failed to process catalog: {result.get('error')}")
            return jsonify({
                'status': 'error',
                'message': result.get('error', 'Failed to process catalog')
            }), 400
            
    except Exception as e:
        logger.error(f"Error in upload_pantone_catalog: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/pantone/convert', methods=['POST'])
def convert_to_pantone():
    """
    Convert RGB colors to Pantone colors
    
    Expects:
        - colors: List of RGB colors to convert
        - catalog_name: (Optional) Name of catalog to use
        
    Returns:
        JSON response with Pantone color matches
    """
    try:
        data = request.json
        if not data or 'colors' not in data:
            return jsonify({'error': 'Colors are required'}), 400
            
        colors = data['colors']
        catalog_name = data.get('catalog_name')
        
        # Convert colors to Pantone
        result = pantone_analyzer.analyze_colors(colors, catalog_name)
        
        return jsonify({
            'status': 'success',
            'colors': result
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/pantone/analyze-image/<path:image_path>', methods=['GET'])
def analyze_image_pantone(image_path):
    """
    Analyze an image and convert its dominant colors to Pantone
    
    Expects:
        - image_path: Path to the image
        - catalog_name: (Optional) Name of catalog to use (query parameter)
        
    Returns:
        JSON response with Pantone color matches
    """
    try:
        # Check if image_path is valid
        if not image_path or image_path == 'undefined':
            logger.error(f"Invalid image path: {image_path}")
            return jsonify({
                'status': 'error',
                'message': 'Invalid image path. Please select a valid image.'
            }), 400
            
        catalog_name = request.args.get('catalog_name')
        
        # Log the request
        logger.info(f"Analyzing image for Pantone colors: {image_path}, catalog: {catalog_name or 'All catalogs'}")
        
        # Fix the image path if it doesn't include the full path
        # If the path doesn't start with the expected directory structure, prepend it
        if not image_path.startswith('/'):
            # Ensure the path has the correct format
            if not image_path.startswith('Users/'):
                # Check if the path is just a filename
                if '/' not in image_path:
                    # Try to find the file in the uploads directory
                    possible_paths = [
                        f"Users/erenkahraman/Desktop/code/archivist-experimental/uploads/{image_path}",
                        f"uploads/{image_path}",
                        image_path
                    ]
                    
                    # Try each path until we find one that works
                    for path in possible_paths:
                        logger.info(f"Trying path: {path}")
                        metadata = search_engine.get_image_metadata(path)
                        if metadata and 'colors' in metadata:
                            image_path = path
                            logger.info(f"Found valid path: {image_path}")
                            break
                else:
                    image_path = f"Users/erenkahraman/Desktop/code/archivist-experimental/uploads/{os.path.basename(image_path)}"
            
        logger.info(f"Using normalized image path: {image_path}")
        
        # Get image metadata
        metadata = search_engine.get_image_metadata(image_path)
        if not metadata or 'colors' not in metadata:
            logger.error(f"Image colors not found for: {image_path}")
            return jsonify({
                'status': 'error',
                'message': 'Image colors not found. The image may not exist or has not been processed.'
            }), 404
            
        # Get dominant colors
        colors = metadata['colors'].get('dominant_colors', [])
        if not colors:
            logger.error(f"No dominant colors found for image: {image_path}")
            return jsonify({'error': 'No dominant colors found in image'}), 404
            
        # Check if the specified catalog exists
        if catalog_name and catalog_name not in pantone_analyzer.get_available_catalogs():
            logger.error(f"Specified catalog not found: {catalog_name}")
            return jsonify({
                'status': 'error',
                'message': f"Catalog '{catalog_name}' not found. Available catalogs: {', '.join(pantone_analyzer.get_available_catalogs())}"
            }), 400
        
        # Convert to Pantone
        pantone_colors = pantone_analyzer.analyze_colors(colors, catalog_name)
        
        logger.info(f"Converted {len(colors)} colors to Pantone using catalog: {catalog_name or 'All catalogs'}")
        
        return jsonify({
            'status': 'success',
            'image_path': image_path,
            'pantone_colors': pantone_colors,
            'catalog_used': catalog_name or 'All catalogs'
        }), 200
    except Exception as e:
        logger.error(f"Error in analyze_image_pantone: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# Basic test route
@app.route('/')
def home():
    return 'API is running' 