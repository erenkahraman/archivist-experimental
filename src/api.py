from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pathlib import Path
import os
from werkzeug.utils import secure_filename
from .search_engine import SearchEngine
import config
from werkzeug.exceptions import BadRequest

app = Flask(__name__)

# Basic test route
@app.route('/')
def home():
    return 'Server is running'

# Test route that returns JSON
@app.route('/api/test')
def test():
    return jsonify({'status': 'ok'})

# Configure CORS for development
CORS(app, resources={
    r"/*": {  # Apply to all routes
        "origins": ["http://localhost:3000"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "supports_credentials": True
    }
})

# Initialize search engine
search_engine = SearchEngine()

# Configure upload folder
UPLOAD_FOLDER = Path(__file__).parent.parent / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/thumbnails/<path:filename>')
def serve_thumbnail(filename):
    """Serve thumbnail images"""
    return send_from_directory(config.THUMBNAIL_DIR, filename)

@app.errorhandler(BadRequest)
def handle_bad_request(e):
    return jsonify({'error': str(e)}), 400

@app.route('/api/upload', methods=['POST'])
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

@app.route('/api/images', methods=['GET'])
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

@app.route('/api/search', methods=['POST'])
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
                    style_match = False
                    
                    # Check in style keywords
                    if 'patterns' in result and 'style_keywords' in result['patterns']:
                        for keyword in result['patterns']['style_keywords']:
                            if style in keyword.lower():
                                style_match = True
                                break
                    
                    # Check in layout, texture, etc.
                    for attr in ['layout', 'texture_type', 'cultural_influence', 'historical_period']:
                        if 'patterns' in result and attr in result['patterns']:
                            if 'type' in result['patterns'][attr] and style in result['patterns'][attr]['type'].lower():
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
            # Sort by timestamp (newest first)
            results.sort(key=lambda x: x.get('timestamp', '0'), reverse=True)
        elif sort_method == 'oldest':
            # Sort by timestamp (oldest first)
            results.sort(key=lambda x: x.get('timestamp', '0'))
        # Default is 'relevance' which is already sorted by similarity
        
        # Add metadata about search results
        response = {
            'query': query,
            'filters': filters,
            'sort': sort_method,
            'total_results': len(results),
            'results': results
        }
        
        return jsonify(response)
    except Exception as e:
        print(f"Search error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete/<filename>', methods=['DELETE'])
def delete_image(filename):
    try:
        # Delete original file
        original_path = config.UPLOAD_DIR / filename
        if original_path.exists():
            original_path.unlink()

        # Delete thumbnail
        thumbnail_path = config.THUMBNAIL_DIR / filename
        if thumbnail_path.exists():
            thumbnail_path.unlink()

        # Remove from metadata - search by filename in metadata
        for path in list(search_engine.metadata.keys()):
            if Path(path).name == filename:
                del search_engine.metadata[path]
                search_engine.save_metadata()
                break

        return jsonify({'success': True})
    except Exception as e:
        print(f"Delete error: {str(e)}")  # Add debug logging
        return jsonify({'error': str(e)}), 500

def ensure_directories():
    """Ensure required directories exist."""
    directories = [
        UPLOAD_FOLDER,
        config.THUMBNAIL_DIR,  # Make sure this exists
        config.INDEX_PATH
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        # Add debug logging
        print(f"Ensured directory exists: {directory}")

def start_api():
    """Start the Flask API server."""
    ensure_directories()
    app.run(
        host='0.0.0.0',  # Allow external connections
        port=8000,
        debug=True
    ) 