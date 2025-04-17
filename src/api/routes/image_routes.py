from flask import request, jsonify, send_from_directory, Blueprint
import os
import sys
from pathlib import Path
import uuid
import time
import logging
from werkzeug.utils import secure_filename

# Add the project root to the path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.config import UPLOAD_DIR, THUMBNAIL_DIR
from src.core.pattern_analyzer import PatternAnalyzer
from src.search import search_engine

# Get API key from environment
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Initialize pattern analyzer
pattern_analyzer = PatternAnalyzer(api_key=GEMINI_API_KEY)

# Set up logging
logger = logging.getLogger(__name__)
DEBUG = os.environ.get('DEBUG', 'False').lower() in ('true', '1', 't')

# Create blueprint
image_blueprint = Blueprint('image', __name__)

def _build_cors_preflight_response():
    """Helper function for CORS preflight responses"""
    response = jsonify({})
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,PUT,DELETE,OPTIONS")
    return response

def allowed_file(filename):
    """Check if file has an allowed extension"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@image_blueprint.route('/thumbnails/<path:filename>')
def serve_thumbnail(filename):
    """Serve thumbnail images"""
    try:
        # Check if the file exists in the thumbnail directory
        thumbnail_path = Path(THUMBNAIL_DIR) / filename
        if not thumbnail_path.exists():
            # Try to create the thumbnail if original exists
            original_path = Path(UPLOAD_DIR) / filename
            if original_path.exists():
                logger.info(f"Creating missing thumbnail for {filename}")
                thumbnail = pattern_analyzer.create_thumbnail(original_path)
                if thumbnail:
                    return send_from_directory(THUMBNAIL_DIR, filename)
            
            # Return a 404 if we can't find or create the thumbnail
            logger.warning(f"Thumbnail not found: {filename}")
            return jsonify({"error": "Thumbnail not found"}), 404
            
        return send_from_directory(THUMBNAIL_DIR, filename)
    except Exception as e:
        logger.error(f"Error serving thumbnail {filename}: {str(e)}")
        return jsonify({"error": str(e)}), 500

@image_blueprint.route('/<path:filename>')
def serve_image(filename):
    """Serve full-size images"""
    return send_from_directory(UPLOAD_DIR, filename)

@image_blueprint.route('/', methods=['GET'])
def get_images():
    try:
        # Get limit parameter (default to 20)
        limit = request.args.get('limit', default=20, type=int)
        # Get offset parameter (default to 0)
        offset = request.args.get('offset', default=0, type=int)
        
        # Check if metadata is available
        if not search_engine.metadata:
            logger.warning("No metadata available")
            return jsonify({'message': 'No images available', 'images': []}), 200
        
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
        return jsonify({'error': str(e), 'images': []}), 500

@image_blueprint.route('/upload', methods=['POST'])
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
            file_path = Path(UPLOAD_DIR) / unique_filename
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

@image_blueprint.route('/delete/<path:filename>', methods=['DELETE', 'OPTIONS', 'POST'])
def delete_image(filename):
    """Delete an image and its associated metadata"""
    # Handle OPTIONS request for CORS preflight
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()
        
    try:
        # Look for the file in uploads directory
        file_path = Path(UPLOAD_DIR) / filename
        thumbnail_path = Path(THUMBNAIL_DIR) / filename
        
        deleted_original = False
        deleted_thumbnail = False
        
        # Delete files if they exist
        if file_path.exists():
            os.remove(file_path)
            deleted_original = True
            logger.info(f"Deleted original file: {file_path}")
        else:
            # Try to find the file by basename in uploads directory
            basename = os.path.basename(filename)
            for file in Path(UPLOAD_DIR).glob(f"*{basename}*"):
                if file.is_file():
                    os.remove(file)
                    deleted_original = True
                    logger.info(f"Deleted original file: {file}")
                    break
            
        if thumbnail_path.exists():
            os.remove(thumbnail_path)
            deleted_thumbnail = True
            logger.info(f"Deleted thumbnail: {thumbnail_path}")
        else:
            # Try to find the thumbnail by basename
            basename = os.path.basename(filename)
            for file in Path(THUMBNAIL_DIR).glob(f"*{basename}*"):
                if file.is_file():
                    os.remove(file)
                    deleted_thumbnail = True
                    logger.info(f"Deleted thumbnail: {file}")
                    break
            
        # Remove metadata and clean up elasticsearch
        success = search_engine.delete_image(filename)
        
        if success:
            logger.info(f"Successfully deleted image and metadata for: {filename}")
        else:
            logger.warning(f"Deleted files but couldn't find metadata for: {filename}")
        
        # Return detailed status
        return jsonify({
            'status': 'success',
            'deleted_original': deleted_original,
            'deleted_thumbnail': deleted_thumbnail,
            'deleted_metadata': success
        }), 200
    except Exception as e:
        logger.error(f"Delete error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@image_blueprint.route('/cleanup-metadata', methods=['POST', 'OPTIONS'])
def cleanup_metadata():
    """Clean up metadata for missing files to fix gallery display issues"""
    # Handle OPTIONS request for CORS preflight
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()
        
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

@image_blueprint.route('/purge-all', methods=['POST', 'OPTIONS'])
def purge_all_images():
    """Delete ALL images and metadata to start fresh"""
    # Handle OPTIONS request for CORS preflight
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()
        
    try:
        logger.info("PURGING ALL IMAGES AND METADATA")
        
        # Clear search engine metadata
        search_engine.clear_metadata()
        
        # Reset pattern analyzer metadata
        pattern_analyzer.metadata = {}
        pattern_analyzer.save_metadata()
        
        # Delete all files in the uploads directory
        for file in Path(UPLOAD_DIR).glob("*"):
            if file.is_file():
                os.remove(file)
        
        # Delete all files in the thumbnails directory
        for file in Path(THUMBNAIL_DIR).glob("*"):
            if file.is_file():
                os.remove(file)
        
        logger.info("All images and metadata purged successfully")
        return jsonify({'status': 'success', 'message': 'All images and metadata purged'}), 200
    except Exception as e:
        logger.error(f"Error purging images: {str(e)}")
        return jsonify({'error': str(e)}), 500

@image_blueprint.route('/cleanup-elasticsearch', methods=['POST', 'OPTIONS'])
def cleanup_elasticsearch():
    """Clean up Elasticsearch index after purging all images"""
    try:
        # Check CORS preflight
        if request.method == 'OPTIONS':
            return _build_cors_preflight_response()

        logger.info("CLEANUP ELASTICSEARCH ENDPOINT IS NOW DEPRECATED - Using in-memory search only")

        # Return success since we're not using Elasticsearch anymore
        return jsonify({'status': 'success', 'message': 'Using in-memory search only, no Elasticsearch cleanup needed'}), 200
    except Exception as e:
        logger.error(f"Error in cleanup endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@image_blueprint.route('/repair-thumbnails', methods=['POST', 'OPTIONS'])
def repair_thumbnails():
    """Repair missing thumbnails and synchronize metadata with actual files"""
    # Handle OPTIONS request for CORS preflight
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()
        
    try:
        results = {
            'thumbnails_recreated': 0,
            'entries_cleaned': 0,
            'new_images_added': 0
        }
        
        # First, clean up missing file entries
        results['entries_cleaned'] = search_engine.cleanup_missing_files()
        
        # Then recreate any missing thumbnails for existing images
        for file_path in Path(UPLOAD_DIR).glob("*"):
            if not file_path.is_file():
                continue
                
            filename = file_path.name
            thumbnail_path = Path(THUMBNAIL_DIR) / filename
            
            if not thumbnail_path.exists():
                logger.info(f"Recreating missing thumbnail for: {filename}")
                try:
                    thumbnail = search_engine.create_thumbnail(file_path)
                    if thumbnail:
                        results['thumbnails_recreated'] += 1
                except Exception as e:
                    logger.error(f"Error recreating thumbnail for {filename}: {e}")
            
            # Check if this image is in metadata, if not, add it
            rel_path = str(file_path.relative_to(Path(UPLOAD_DIR)))
            found = False
            
            for meta_path, metadata in search_engine.metadata.items():
                if metadata.get('filename') == filename or meta_path == rel_path:
                    found = True
                    break
                    
            if not found:
                # Process the new image
                logger.info(f"Adding missing image to metadata: {filename}")
                try:
                    metadata = search_engine.process_image(file_path)
                    if metadata:
                        results['new_images_added'] += 1
                except Exception as e:
                    logger.error(f"Error processing missing image {filename}: {e}")
        
        return jsonify({
            'status': 'success',
            'message': f"Repair complete: {results['thumbnails_recreated']} thumbnails recreated, {results['entries_cleaned']} entries cleaned, {results['new_images_added']} new images added",
            'results': results
        }), 200
    except Exception as e:
        logger.error(f"Error repairing thumbnails: {str(e)}")
        return jsonify({'error': str(e)}), 500

@image_blueprint.route('/reindex-embeddings', methods=['POST', 'OPTIONS'])
def reindex_embeddings():
    """
    Regenerate embeddings for all images.
    
    Optional parameters:
    - force: Set to 'true' to regenerate embeddings for all images, even if they already exist
    """
    try:
        # Handle OPTIONS request for CORS preflight
        if request.method == 'OPTIONS':
            return _build_cors_preflight_response()
            
        # Check if force flag is set
        force = False
        if request.is_json and request.json:
            force = request.json.get('force', False)
        else:
            force = request.form.get('force', 'false').lower() == 'true'
            
        logger.info(f"Reindexing all embeddings with force={force}")
        
        # Process reindexing in background thread
        def reindex_task():
            try:
                stats = search_engine.reindex_all_with_embeddings(force=force)
                logger.info(f"Reindexing complete: {stats}")
            except Exception as e:
                logger.error(f"Error in reindexing task: {str(e)}")
                
        # Start thread for background processing
        import threading
        thread = threading.Thread(target=reindex_task)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'status': 'processing',
            'message': 'Reindexing started in background thread'
        }), 202
        
    except Exception as e:
        logger.error(f"Error in reindex_embeddings: {str(e)}")
        return jsonify({'error': str(e)}), 500

@image_blueprint.route('/mark-missing/<path:filename>', methods=['POST', 'OPTIONS', 'DELETE'])
def mark_image_as_missing(filename):
    """Mark an image as missing without deleting the actual file"""
    # Handle OPTIONS request for CORS preflight
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()
        
    try:
        # Simply update the metadata for this image
        if search_engine and hasattr(search_engine, 'delete_image'):
            success = search_engine.delete_image(filename)
            if success:
                logger.info(f"Successfully marked image as missing: {filename}")
                return jsonify({
                    'status': 'success',
                    'message': f"Image {filename} marked as missing"
                }), 200
            else:
                logger.warning(f"Could not find metadata for: {filename}")
                return jsonify({
                    'status': 'warning',
                    'message': f"Could not find metadata for: {filename}"
                }), 404
        else:
            return jsonify({
                'error': 'Search engine unavailable'
            }), 500
    except Exception as e:
        logger.error(f"Error marking image as missing: {str(e)}")
        return jsonify({'error': str(e)}), 500 