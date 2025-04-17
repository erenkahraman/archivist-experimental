"""
API routes for image upload and processing
"""
from flask import Blueprint, request, jsonify, make_response
import os
import sys
import logging
from werkzeug.utils import secure_filename
from pathlib import Path
import uuid
import time

# Add the project root to the path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.pattern_analyzer import PatternAnalyzer
from src.search import search_engine
from src.config.config import UPLOAD_DIR

# Set up logging
logger = logging.getLogger(__name__)

# Create blueprint
upload_blueprint = Blueprint('upload', __name__)

# Initialize pattern analyzer
pattern_analyzer = PatternAnalyzer(api_key=os.environ.get("GEMINI_API_KEY"))

def allowed_file(filename):
    """Check if file has an allowed extension"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@upload_blueprint.route('/', methods=['OPTIONS'])
def options():
    """Handle preflight CORS requests"""
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "http://localhost:3000")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
    response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
    return response

@upload_blueprint.route('/', methods=['POST'])
def upload_file():
    """
    Upload a single image and process it
    
    Returns:
        JSON response with upload status and metadata
    """
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        
        # Check if file has a name
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        # Check if file is allowed
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
            
        # Generate secure filename with timestamp
        filename = secure_filename(file.filename)
        name, ext = os.path.splitext(filename)
        timestamp = int(time.time())
        unique_filename = f"{name}_{timestamp}{ext}"
        
        # Create upload path
        upload_path = Path(UPLOAD_DIR)
        upload_path.mkdir(exist_ok=True, parents=True)
        
        # Save file
        file_path = upload_path / unique_filename
        file.save(file_path)
        
        logger.info(f"File saved to {file_path}")
        
        # Process image
        metadata = pattern_analyzer.process_image(file_path)
        
        if not metadata:
            return jsonify({
                'error': 'Failed to process image', 
                'file': unique_filename,
                'path': str(file_path)
            }), 500
            
        # Update search engine metadata
        all_metadata = pattern_analyzer.get_all_metadata()
        search_engine.set_metadata(all_metadata)
        
        # Return success
        return jsonify({
            'success': True,
            'file': unique_filename,
            'metadata': metadata
        })
        
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500 