"""
API routes for system status and health checks.
"""
from flask import Blueprint, jsonify
import os
import sys
import logging
from pathlib import Path
import time

# Add the project root to the path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.config import UPLOAD_DIR, THUMBNAIL_DIR, METADATA_DIR
from src.search import search_engine

# Set up logging
logger = logging.getLogger(__name__)

# Create blueprint
status_blueprint = Blueprint('status', __name__)

@status_blueprint.route('/', methods=['GET'])
def get_status():
    """
    Get the current system status
    
    Returns:
        JSON response with system status information
    """
    try:
        # Check if directories exist
        upload_exists = Path(UPLOAD_DIR).exists()
        thumbnail_exists = Path(THUMBNAIL_DIR).exists()
        metadata_exists = Path(METADATA_DIR).exists()
        
        # Count files in directories
        upload_count = len(list(Path(UPLOAD_DIR).glob('*'))) if upload_exists else 0
        thumbnail_count = len(list(Path(THUMBNAIL_DIR).glob('*'))) if thumbnail_exists else 0
        metadata_count = len(search_engine.metadata) if hasattr(search_engine, 'metadata') else 0
        
        # Build status response
        status = {
            'status': 'ok',
            'time': time.time(),
            'directories': {
                'upload': {
                    'exists': upload_exists,
                    'path': str(UPLOAD_DIR),
                    'file_count': upload_count
                },
                'thumbnail': {
                    'exists': thumbnail_exists,
                    'path': str(THUMBNAIL_DIR),
                    'file_count': thumbnail_count
                },
                'metadata': {
                    'exists': metadata_exists,
                    'path': str(METADATA_DIR)
                }
            },
            'metadata_count': metadata_count,
            'version': '0.1.0'
        }
        
        return jsonify(status), 200
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        return jsonify({'error': str(e)}), 500 