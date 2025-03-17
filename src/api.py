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

# Load environment variables from .env file
dotenv.load_dotenv()

# Create a Flask Blueprint
app = Blueprint('api', __name__)

# Get Gemini API key from environment variable
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("No Gemini API key found in environment variables")

# Initialize search engine with Gemini API key
search_engine = SearchEngine(gemini_api_key=GEMINI_API_KEY)

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
        
        # Update the API key in the search engine
        search_engine.set_gemini_api_key(api_key)
        
        return jsonify({'status': 'success', 'message': 'Gemini API key updated successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle file upload with proper validation and error handling.
    
    Returns:
        JSON response with metadata or error
    """
    try:
        if 'file' not in request.files:
            raise BadRequest('No file part')
            
        file = request.files['file']
        if not file or not file.filename:
            raise BadRequest('No selected file')
            
        if not allowed_file(file.filename):
            raise BadRequest('File type not allowed')

        # Create a secure filename
        filename = secure_filename(file.filename)
        file_path = config.UPLOAD_DIR / filename

        # Check if file already exists
        if file_path.exists():
            # Add timestamp to filename to make it unique
            import time
            timestamp = int(time.time())
            name, ext = os.path.splitext(filename)
            filename = f"{name}_{timestamp}{ext}"
            file_path = config.UPLOAD_DIR / filename

        # Save the file
        file.save(str(file_path))
        print(f"File saved to: {file_path}")

        # Process image
        metadata = search_engine.process_image(file_path)
        if metadata is None:
            # If processing failed, try to remove the file
            try:
                os.remove(str(file_path))
            except:
                pass
            return jsonify({'error': 'Failed to process image'}), 500

        return jsonify(metadata), 200

    except Exception as e:
        print(f"Upload error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/images', methods=['GET'])
def get_images():
    try:
        # Sadece geçerli metadata'yı döndür
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
    Enhanced search endpoint with advanced filtering options.
    
    Accepts the following parameters:
    - query: Main search query (required)
    - filters: Dictionary of filters to apply (optional)
      - pattern_type: Filter by pattern type
      - color: Filter by color
      - style: Filter by style
    - sort: Sort method (optional, default: 'relevance')
      - 'relevance': Sort by search relevance
      - 'newest': Sort by upload date (newest first)
      - 'oldest': Sort by upload date (oldest first)
    - limit: Maximum number of results (optional, default: 50)
    """
    try:
        # Get search parameters
        data = request.json or {}
        query = data.get('query', '').strip()
        filters = data.get('filters', {})
        sort_method = data.get('sort', 'relevance')
        limit = min(int(data.get('limit', 50)), 100)  # Cap at 100 results
        
        # Log search request
        print(f"Search request: query='{query}', filters={filters}, sort={sort_method}, limit={limit}")
        
        # Perform basic search
        results = search_engine.search(query, k=limit)
        
        # Apply additional filters if provided
        if filters:
            filtered_results = []
            for result in results:
                # Check if result matches all filters
                matches_all_filters = True
                
                # Pattern type filter
                if 'pattern_type' in filters and filters['pattern_type']:
                    pattern_type = filters['pattern_type'].lower()
                    if 'patterns' in result and 'category' in result['patterns']:
                        if pattern_type not in result['patterns']['category'].lower():
                            matches_all_filters = False
                
                # Color filter
                if 'color' in filters and filters['color']:
                    color = filters['color'].lower()
                    if 'colors' in result and 'dominant_colors' in result['colors']:
                        color_match = False
                        for color_data in result['colors']['dominant_colors']:
                            if color in color_data['name'].lower():
                                color_match = True
                                break
                        if not color_match:
                            matches_all_filters = False
                
                # Style filter
                if 'style' in filters and filters['style']:
                    style = filters['style'].lower()
                    if 'patterns' in result and 'style' in result['patterns']:
                        style_match = False
                        for style_key, style_data in result['patterns']['style'].items():
                            if isinstance(style_data, dict) and 'type' in style_data:
                                if style in style_data['type'].lower():
                                    style_match = True
                                    break
                        if not style_match:
                            matches_all_filters = False
                
                # Add to filtered results if it matches all filters
                if matches_all_filters:
                    filtered_results.append(result)
            
            # Replace results with filtered results
            results = filtered_results
        
        # Apply sorting
        if sort_method == 'newest':
            results.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        elif sort_method == 'oldest':
            results.sort(key=lambda x: x.get('timestamp', 0))
        
        return jsonify(results)
        
    except Exception as e:
        print(f"Search error: {str(e)}")
        import traceback
        traceback.print_exc()
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

# Basic test route
@app.route('/')
def home():
    return 'API is running' 