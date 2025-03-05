from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pathlib import Path
import os
from werkzeug.utils import secure_filename
from .search_engine import SearchEngine
import config

app = Flask(__name__)

# Basic test route
@app.route('/')
def home():
    return 'Server is running'

# Test route that returns JSON
@app.route('/api/test')
def test():
    return jsonify({'status': 'ok'})

# Configure CORS properly
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "supports_credentials": True
    },
    r"/thumbnails/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET"],
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

@app.route('/thumbnails/<path:filename>')
def serve_thumbnail(filename):
    """Serve thumbnail images."""
    return send_from_directory(config.THUMBNAIL_DIR, filename)

@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Güvenli dosya adı oluştur
        filename = Path(file.filename).name
        file_path = config.UPLOAD_DIR / filename

        # Dosyayı kaydet
        file.save(str(file_path))
        print(f"File saved to: {file_path}")  # Debug log

        # Process image
        metadata = search_engine.process_image(file_path)
        if metadata is None:
            return jsonify({'error': 'Failed to process image'}), 500

        return jsonify(metadata), 200

    except Exception as e:
        print(f"Upload error: {str(e)}")  # Debug log
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
    try:
        query = request.json.get('query', '')
        results = search_engine.search(query)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete/<path:filepath>', methods=['DELETE'])
def delete_image(filepath):
    try:
        # Convert URL-encoded path to system path
        file_path = Path(filepath)
        
        # Delete original file
        original_path = config.UPLOAD_DIR / file_path.name
        if original_path.exists():
            original_path.unlink()

        # Delete thumbnail
        thumbnail_path = config.THUMBNAIL_DIR / file_path.name
        if thumbnail_path.exists():
            thumbnail_path.unlink()

        # Remove from metadata
        if str(file_path) in search_engine.metadata:
            del search_engine.metadata[str(file_path)]
            search_engine.save_metadata()

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def ensure_directories():
    """Ensure required directories exist."""
    directories = [
        UPLOAD_FOLDER,
        config.THUMBNAIL_DIR,
        config.INDEX_PATH
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Ensured directory exists: {directory}")

def start_api():
    """Start the Flask API server."""
    ensure_directories()
    app.run(
        host=config.API_HOST,
        port=config.API_PORT,
        debug=True
    ) 