from flask import request, jsonify, send_from_directory
import os
from pathlib import Path
import uuid
import time
import logging
from werkzeug.utils import secure_filename
import config
from .. import api, search_engine, DEBUG

logger = logging.getLogger(__name__)

# Configure upload folder
UPLOAD_FOLDER = Path(__file__).parent.parent.parent.parent / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@api.route('/thumbnails/<path:filename>')
def serve_thumbnail(filename):
    """Serve thumbnail images"""
    return send_from_directory(config.THUMBNAIL_DIR, filename)

@api.route('/images/<path:filename>')
def serve_image(filename):
    """Serve full-size images"""
    return send_from_directory(config.UPLOAD_DIR, filename)

@api.route('/images', methods=['GET'])
def get_images():
    try:
        # Get limit parameter (default to 20)
        limit = request.args.get('limit', default=20, type=int)
        # Get offset parameter (default to 0)
        offset = request.args.get('offset', default=0, type=int)
        
        # Only return valid metadata
        valid_metadata = {
            path: data for path, data in search_engine.metadata.items()
            if data and 'thumbnail_path' in data and 'patterns' in data
        }
        
        # Convert to list for pagination
        all_images = list(valid_metadata.values())
        
        # Sort by timestamp (newest first)
        all_images.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        
        # Apply pagination
        paginated_images = all_images[offset:offset + limit]
        
        return jsonify(paginated_images)
    except Exception as e:
        logger.error(f"Error getting images: {str(e)}")
        return jsonify({'error': str(e)}), 500

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
            # Get file extension
            ext = os.path.splitext(file.filename)[1].lower()
            
            # Create a unique filename with timestamp
            unique_filename = f"{uuid.uuid4()}_{int(time.time())}{ext}"
            
            # Save the file
            file_path = config.UPLOAD_DIR / unique_filename
            file.save(file_path)
            
            if DEBUG:
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
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api.route('/delete/<path:filename>', methods=['DELETE', 'OPTIONS'])
def delete_image(filename):
    """Delete an image and its associated metadata"""
    # Handle OPTIONS request for CORS preflight
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        # Look for the file in uploads directory
        file_path = config.UPLOAD_DIR / filename
        thumbnail_path = config.THUMBNAIL_DIR / filename
        
        # Delete files if they exist
        if file_path.exists():
            os.remove(file_path)
            if DEBUG:
                logger.info(f"Deleted file: {file_path}")
            
        if thumbnail_path.exists():
            os.remove(thumbnail_path)
            if DEBUG:
                logger.info(f"Deleted thumbnail: {thumbnail_path}")
            
        # Remove metadata and clean up elasticsearch
        success = search_engine.delete_image(filename)
        
        if success:
            logger.info(f"Successfully deleted image and metadata for: {filename}")
        else:
            logger.warning(f"Deleted files but couldn't find metadata for: {filename}")
        
        return jsonify({'status': 'success'}), 200
    except Exception as e:
        logger.error(f"Delete error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api.route('/cleanup-metadata', methods=['POST', 'OPTIONS'])
def cleanup_metadata():
    """Clean up metadata for missing files to fix gallery display issues"""
    # Handle OPTIONS request for CORS preflight
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        # Run the cleanup operation
        cleaned_count = search_engine.cleanup_missing_files()
        
        # Return results
        return jsonify({
            'status': 'success', 
            'cleaned_entries': cleaned_count,
            'message': f"Cleaned up {cleaned_count} missing file entries"
        }), 200
    except Exception as e:
        logger.error(f"Cleanup error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api.route('/purge-all', methods=['POST', 'OPTIONS'])
def purge_all_images():
    """Delete ALL images and metadata to start fresh"""
    # Handle OPTIONS request for CORS preflight
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        logger.info("PURGING ALL IMAGES AND METADATA")
        
        # First, clear all metadata
        search_engine.metadata.clear()
        search_engine.save_metadata()
        
        # Delete all files in the uploads directory
        for file in config.UPLOAD_DIR.glob("*"):
            if file.is_file():
                os.remove(file)
        
        # Delete all files in the thumbnails directory
        for file in config.THUMBNAIL_DIR.glob("*"):
            if file.is_file():
                os.remove(file)
        
        logger.info("All images and metadata purged successfully")
        return jsonify({'status': 'success', 'message': 'All images and metadata purged'}), 200
    except Exception as e:
        logger.error(f"Error purging images: {str(e)}")
        return jsonify({'error': str(e)}), 500 