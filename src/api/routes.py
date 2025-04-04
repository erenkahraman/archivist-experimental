from flask import request, jsonify, send_from_directory, Blueprint, make_response
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

# Import Elasticsearch client if available
try:
    from src.search.elasticsearch_client import ElasticsearchClient
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Control logging verbosity
DEBUG = os.environ.get('DEBUG', 'false').lower() == 'true'

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

# Initialize Elasticsearch client if available
es_client = None
if ELASTICSEARCH_AVAILABLE:
    # Get Elasticsearch configuration from environment variables
    ES_HOSTS = os.environ.get('ELASTICSEARCH_HOSTS', 'http://localhost:9200').split(',')
    ES_CLOUD_ID = os.environ.get('ELASTICSEARCH_CLOUD_ID')
    ES_API_KEY = os.environ.get('ELASTICSEARCH_API_KEY')
    ES_USERNAME = os.environ.get('ELASTICSEARCH_USERNAME')
    ES_PASSWORD = os.environ.get('ELASTICSEARCH_PASSWORD')
    
    # Initialize Elasticsearch client
    try:
        es_client = ElasticsearchClient(
            hosts=ES_HOSTS,
            cloud_id=ES_CLOUD_ID,
            api_key=ES_API_KEY,
            username=ES_USERNAME,
            password=ES_PASSWORD
        )
        
        if es_client.is_connected():
            logger.info("Connected to Elasticsearch successfully")
            # Create the index if it doesn't exist
            es_client.create_index()
        else:
            logger.warning("Elasticsearch client initialized but not connected")
            es_client = None
    except Exception as e:
        logger.error(f"Failed to initialize Elasticsearch client: {str(e)}")
        es_client = None

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
    Upload and process a new image file.
    
    Expects:
        - file: The image file to upload (multipart/form-data)
        
    Returns:
        - JSON response with processed image metadata
    """
    try:
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        
        file = request.files['file']
        
        # If the user does not select a file, the browser submits an empty file
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if the file is allowed
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
                
                # Index to Elasticsearch if available
                if es_client and es_client.is_connected():
                    try:
                        # Make sure the metadata has an ID
                        if 'id' not in metadata:
                            metadata['id'] = metadata.get('path', str(unique_filename))
                            
                        # Index the document
                        indexed = es_client.index_document(metadata)
                        if indexed:
                            logger.info(f"Image indexed to Elasticsearch: {metadata.get('id')}")
                        else:
                            logger.warning(f"Failed to index image to Elasticsearch: {metadata.get('id')}")
                    except Exception as es_err:
                        logger.error(f"Error indexing to Elasticsearch: {str(es_err)}")
                
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
        
        # Only log this message on initial load or in extreme debug scenarios
        # if DEBUG:
        #     logger.info(f"Returning {len(paginated_images)} images (of {len(all_images)} total)")
        
        return jsonify(paginated_images)
    except Exception as e:
        logger.error(f"Error getting images: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api.route('/search', methods=['POST', 'OPTIONS'])
def search():
    """
    Search for images based on a query.
    
    Parameters:
        query: The search query
        limit: Maximum number of results to return (default: 20)
        min_similarity: Minimum similarity threshold (default: 0.1)
        force_memory_search: Force in-memory search even if Elasticsearch is available (default: false)
        
    Returns:
        JSON response with search results
    """
    # Handle CORS preflight request
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
        
    try:
        # Get parameters from request
        data = request.get_json() or {}
        query = data.get('query', '').strip()
        limit = int(data.get('limit', 20))
        min_similarity = float(data.get('min_similarity', 0.1))
        force_memory_search = data.get('force_memory_search', False)
        
        # Check if query is provided
        if not query:
            return jsonify({
                'error': 'Missing query parameter',
                'status': 'error'
            }), 400
            
        logger.info(f"Search request: query='{query}', limit={limit}, min_similarity={min_similarity}")
        
        # Perform search using Elasticsearch if available and not forced to use memory search
        start_time = time.time()
        search_method = "in-memory"
        
        if es_client and es_client.is_connected() and not force_memory_search:
            logger.info(f"Using Elasticsearch for search: '{query}'")
            search_method = "elasticsearch"
            results = es_client.search(query, limit, min_similarity)
        else:
            logger.info(f"Using in-memory search for: '{query}'")
            results = search_engine.search(query, limit)
            
        # Filter results based on minimum similarity threshold
        results = [r for r in results if r.get('similarity', 0) >= min_similarity]
        
        # Format results for API response
        formatted_results = []
        for result in results:
            # Format metadata
            formatted_result = format_search_result(result)
            formatted_results.append(formatted_result)
        
        execution_time = time.time() - start_time
        
        # Return results
        return jsonify({
            'query': query,
            'result_count': len(formatted_results),
            'min_similarity': min_similarity,
            'execution_time': execution_time,
            'search_method': search_method,
            'results': formatted_results
        })
        
    except Exception as e:
        logger.error(f"Error in search: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

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
        logger.error(f"Error generating prompt: {str(e)}")
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
            
        # Get metadata before removing it
        metadata = None
        if filename in search_engine.metadata:
            metadata = search_engine.metadata.get(filename)
            
        # Remove metadata if it exists
        if filename in search_engine.metadata:
            del search_engine.metadata[filename]
            if DEBUG:
                logger.info(f"Removed metadata for: {filename}")
            
        # Save metadata
        search_engine.save_metadata()
        
        # Delete from Elasticsearch if available
        if es_client and es_client.is_connected() and metadata:
            try:
                # Try to delete using ID if available, otherwise use filename
                doc_id = metadata.get('id', filename)
                deleted = es_client.delete_document(doc_id)
                
                if deleted:
                    logger.info(f"Deleted document from Elasticsearch: {doc_id}")
                else:
                    logger.warning(f"Document not found in Elasticsearch: {doc_id}")
            except Exception as es_err:
                logger.error(f"Error deleting from Elasticsearch: {str(es_err)}")
        
        return jsonify({'status': 'success'}), 200
    except Exception as e:
        logger.error(f"Delete error: {str(e)}")
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
        
        # If Elasticsearch is available, delete the index and recreate it
        if es_client and es_client.is_connected():
            try:
                # Delete the index
                if es_client.client.indices.exists(index=es_client.index_name):
                    es_client.client.indices.delete(index=es_client.index_name)
                    logger.info(f"Deleted Elasticsearch index: {es_client.index_name}")
                    
                # Recreate the index
                es_client.create_index()
                logger.info(f"Recreated Elasticsearch index: {es_client.index_name}")
            except Exception as es_err:
                logger.error(f"Error purging Elasticsearch: {str(es_err)}")
        
        logger.info("All images and metadata purged successfully")
        return jsonify({'status': 'success', 'message': 'All images and metadata purged'}), 200
    except Exception as e:
        logger.error(f"Error purging images: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Basic test route
@api.route('/')
def home():
    return 'API is running'

@api.route('/admin/index-to-elasticsearch', methods=['POST'])
def index_to_elasticsearch():
    """
    Admin endpoint to index all existing metadata into Elasticsearch.
    
    This is useful for initially populating Elasticsearch with existing data.
    
    Returns:
        JSON response with results
    """
    try:
        if not es_client:
            return jsonify({
                'status': 'error', 
                'message': 'Elasticsearch client not available'
            }), 400
        
        if not es_client.is_connected():
            return jsonify({
                'status': 'error', 
                'message': 'Elasticsearch client not connected'
            }), 400
        
        # Get metadata from search engine
        metadata = search_engine.metadata
        
        if not metadata:
            return jsonify({
                'status': 'warning', 
                'message': 'No metadata available to index'
            }), 200
        
        # Prepare documents for bulk indexing
        documents = []
        for path, data in metadata.items():
            # Skip any items without valid data
            if not data:
                continue
                
            # Use ID or path as document ID
            if 'id' not in data:
                data['id'] = path
                
            documents.append(data)
        
        logger.info(f"Indexing {len(documents)} documents to Elasticsearch")
        
        # Bulk index documents
        success = es_client.bulk_index(documents)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': f'Successfully indexed {len(documents)} documents to Elasticsearch'
            }), 200
        else:
            return jsonify({
                'status': 'error', 
                'message': 'Failed to index documents to Elasticsearch'
            }), 500
            
    except Exception as e:
        logger.error(f"Error indexing to Elasticsearch: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error', 
            'message': f'Error: {str(e)}'
        }), 500

@api.route('/admin/rebuild-elasticsearch-index', methods=['POST'])
def rebuild_elasticsearch_index():
    """
    Admin endpoint to rebuild the Elasticsearch index with new mappings.
    
    This is useful after updating analyzers or mappings.
    All documents will be reindexed with the new mappings.
    
    Returns:
        JSON response with results
    """
    try:
        if not es_client:
            return jsonify({
                'status': 'error', 
                'message': 'Elasticsearch client not available'
            }), 400
        
        if not es_client.is_connected():
            return jsonify({
                'status': 'error', 
                'message': 'Elasticsearch client not connected'
            }), 400
        
        logger.info("Rebuilding Elasticsearch index with updated mappings")
        
        # Rebuild index
        success = es_client.rebuild_index()
        
        if success:
            # Also invalidate cache
            if hasattr(search_engine, 'cache'):
                search_engine.cache.invalidate_all()
                
            return jsonify({
                'status': 'success',
                'message': 'Successfully rebuilt Elasticsearch index with new mappings'
            }), 200
        else:
            return jsonify({
                'status': 'error', 
                'message': 'Failed to rebuild Elasticsearch index'
            }), 500
            
    except Exception as e:
        logger.error(f"Error rebuilding Elasticsearch index: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error: {str(e)}'
        }), 500

@api.route('/admin/cleanup-metadata', methods=['POST'])
def cleanup_metadata():
    """
    Admin endpoint to check for and cleanup orphaned metadata.
    
    This ensures consistency between:
    1. Files on disk
    2. In-memory metadata
    3. Elasticsearch index
    
    Returns:
        JSON response with results
    """
    try:
        from pathlib import Path
        
        # Check if Elasticsearch is available
        es_available = es_client and es_client.is_connected()
        
        # Structure to store results
        results = {
            'orphaned_metadata': [],
            'orphaned_es_docs': [],
            'fixed_count': 0,
            'total_files': 0,
            'total_metadata': len(search_engine.metadata) if search_engine.metadata else 0,
            'total_es_docs': 0
        }
        
        # Get all actual files in upload directory
        upload_dir = config.UPLOAD_DIR
        existing_files = set()
        
        for file_path in upload_dir.glob('*'):
            if file_path.is_file():
                # Store the relative path
                rel_path = str(file_path.relative_to(upload_dir))
                existing_files.add(rel_path)
                results['total_files'] += 1
        
        logger.info(f"Found {results['total_files']} files on disk")
        
        # Find orphaned metadata (metadata with no corresponding file)
        if search_engine.metadata:
            for path, data in list(search_engine.metadata.items()):
                # Check if the path exists in the list of actual files
                if path not in existing_files:
                    results['orphaned_metadata'].append({
                        'path': path, 
                        'id': data.get('id', 'unknown')
                    })
                    
                    # Remove from in-memory metadata
                    del search_engine.metadata[path]
                    results['fixed_count'] += 1
                    logger.info(f"Removed orphaned metadata entry: {path}")
        
        # Check Elasticsearch if available
        if es_available:
            # Get total document count
            count_result = es_client.client.count(index=es_client.index_name)
            results['total_es_docs'] = count_result["count"]
            
            # Use scroll API to get all documents
            query_body = {
                "query": {
                    "match_all": {}
                },
                "_source": ["id", "path", "filename"]
            }
            
            scroll_response = es_client.client.search(
                index=es_client.index_name, 
                body=query_body,
                scroll="2m",
                size=100
            )
            
            scroll_id = scroll_response["_scroll_id"]
            hits = scroll_response["hits"]["hits"]
            
            while len(hits) > 0:
                for hit in hits:
                    doc = hit["_source"]
                    doc_id = hit["_id"]
                    
                    # Check if file exists
                    file_exists = False
                    
                    if "path" in doc:
                        file_path = Path(upload_dir) / doc["path"]
                        if file_path.exists():
                            file_exists = True
                    
                    if not file_exists and "filename" in doc:
                        # Try to find by filename (less reliable)
                        file_path = Path(upload_dir) / doc["filename"]
                        if file_path.exists():
                            file_exists = True
                    
                    if not file_exists:
                        # Document doesn't correspond to a file
                        results['orphaned_es_docs'].append({
                            'id': doc_id,
                            'path': doc.get('path', 'unknown'),
                            'filename': doc.get('filename', 'unknown')
                        })
                        
                        # Delete from Elasticsearch
                        es_client.delete_document(doc_id)
                        results['fixed_count'] += 1
                        logger.info(f"Deleted orphaned Elasticsearch document: {doc_id}")
                
                # Get next batch
                scroll_response = es_client.client.scroll(
                    scroll_id=scroll_id,
                    scroll="2m"
                )
                
                scroll_id = scroll_response["_scroll_id"]
                hits = scroll_response["hits"]["hits"]
            
            # Clear the scroll to free resources
            es_client.client.clear_scroll(scroll_id=scroll_id)
        
        # Save the updated metadata
        search_engine.save_metadata()
        
        # Invalidate cache
        if hasattr(search_engine, 'cache'):
            search_engine.cache.invalidate_all()
        
        # Prepare summary
        orphaned_count = len(results['orphaned_metadata']) + len(results['orphaned_es_docs'])
        
        if orphaned_count > 0:
            message = f"Cleaned up {orphaned_count} orphaned metadata entries"
        else:
            message = "No orphaned metadata found - everything is consistent"
        
        return jsonify({
            'status': 'success',
            'message': message,
            'results': results
        }), 200
        
    except Exception as e:
        logger.error(f"Error during metadata cleanup: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error: {str(e)}'
        }), 500
        
@api.route('/admin/diagnose-paisley-issue', methods=['GET'])
def diagnose_paisley_issue():
    """
    Special diagnostic endpoint to understand the "paisley" search issue.
    
    This will:
    1. Check direct search results for "paisley" 
    2. Show how results match (which fields)
    3. Check if files exist
    
    Returns:
        JSON response with diagnostic information
    """
    try:
        results = {
            'paisley_matches': [],
            'all_files': [],
            'metadata_count': len(search_engine.metadata) if search_engine.metadata else 0,
            'es_count': 0,
            'diagnosis': ""
        }
        
        # Get all files
        upload_dir = config.UPLOAD_DIR
        for file_path in upload_dir.glob('*'):
            if file_path.is_file():
                results['all_files'].append(str(file_path.name))
        
        # Get metadata count
        if es_client and es_client.is_connected():
            # Get total document count
            count_result = es_client.client.count(index=es_client.index_name)
            results['es_count'] = count_result["count"]
            
            # Perform search for "paisley"
            search_results = es_client.search(query="paisley", limit=20)
            
            # Analyze each result
            for result in search_results:
                match_info = {
                    'id': result.get('id', 'unknown'),
                    'filename': result.get('filename', 'unknown'),
                    'path': result.get('path', 'unknown'),
                    'match_fields': [],
                    'file_exists': False,
                    'score': result.get('similarity', 0),
                    'raw_score': result.get('_score', 0)
                }
                
                # Check if file exists
                if 'path' in result:
                    file_path = Path(upload_dir) / result['path']
                    match_info['file_exists'] = file_path.exists()
                
                # Check which fields matched
                if 'patterns' in result:
                    patterns = result['patterns']
                    
                    # Check main theme
                    if 'main_theme' in patterns and 'paisley' in patterns['main_theme'].lower():
                        match_info['match_fields'].append('main_theme')
                    
                    # Check primary pattern
                    if 'primary_pattern' in patterns and 'paisley' in patterns['primary_pattern'].lower():
                        match_info['match_fields'].append('primary_pattern')
                    
                    # Check secondary patterns
                    if 'secondary_patterns' in patterns:
                        for pattern in patterns['secondary_patterns']:
                            if isinstance(pattern, dict) and 'name' in pattern and 'paisley' in pattern['name'].lower():
                                match_info['match_fields'].append('secondary_patterns')
                                break
                    
                    # Check content details
                    if 'content_details' in patterns:
                        for detail in patterns['content_details']:
                            if isinstance(detail, dict) and 'name' in detail and 'paisley' in detail['name'].lower():
                                match_info['match_fields'].append('content_details')
                                break
                    
                    # Check prompt
                    if 'prompt' in patterns and 'final_prompt' in patterns['prompt'] and 'paisley' in patterns['prompt']['final_prompt'].lower():
                        match_info['match_fields'].append('prompt')
                    
                    # Check style keywords
                    if 'style_keywords' in patterns and any('paisley' in kw.lower() for kw in patterns['style_keywords']):
                        match_info['match_fields'].append('style_keywords')
                
                results['paisley_matches'].append(match_info)
        
        # Diagnose the issue
        if len(results['paisley_matches']) > len(results['all_files']):
            results['diagnosis'] = "There are more search results than actual files. This indicates orphaned metadata in the search index."
        elif any(not match['file_exists'] for match in results['paisley_matches']):
            results['diagnosis'] = "Some search results point to files that don't exist on disk. These need to be cleaned up."
        elif results['metadata_count'] != len(results['all_files']):
            results['diagnosis'] = "The number of metadata entries doesn't match the number of files. This suggests inconsistency."
        elif results['es_count'] != len(results['all_files']):
            results['diagnosis'] = "The number of Elasticsearch documents doesn't match the number of files. This suggests inconsistency."
        else:
            results['diagnosis'] = "Unclear why 'paisley' matches aren't showing in the gallery. This might be a frontend issue or a cache problem."
        
        return jsonify({
            'status': 'success',
            'message': 'Diagnostic information gathered',
            'results': results
        }), 200
        
    except Exception as e:
        logger.error(f"Error during paisley diagnosis: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error: {str(e)}'
        }), 500

# Add helper function at the top before routes
def format_search_result(result):
    """
    Format a search result for API response
    
    Args:
        result: Raw search result
        
    Returns:
        Formatted result dictionary
    """
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
        patterns = result['patterns']
        
        # New detailed fields
        item['pattern'] = {
            'main_theme': patterns.get('main_theme', 'Unknown'),
            'main_theme_confidence': patterns.get('main_theme_confidence', 0.0),
            'primary': patterns.get('primary_pattern', 'Unknown'),
            'confidence': patterns.get('pattern_confidence', 0.0),
            'secondary': [p.get('name') for p in patterns.get('secondary_patterns', []) if isinstance(p, dict) and 'name' in p],
            'elements': [e.get('name') if isinstance(e, dict) else str(e) for e in patterns.get('elements', [])],
        }
        
        # Content details
        item['content_details'] = []
        for content in patterns.get('content_details', []):
            if isinstance(content, dict) and 'name' in content:
                item['content_details'].append({
                    'name': content.get('name', ''),
                    'confidence': content.get('confidence', 0.0)
                })
        
        # Stylistic attributes
        item['stylistic_attributes'] = []
        for style in patterns.get('stylistic_attributes', []):
            if isinstance(style, dict) and 'name' in style:
                item['stylistic_attributes'].append({
                    'name': style.get('name', ''),
                    'confidence': style.get('confidence', 0.0)
                })
        
        # Include style keywords
        item['style_keywords'] = patterns.get('style_keywords', [])
        
        # Include prompt if available
        if 'prompt' in patterns and 'final_prompt' in patterns['prompt']:
            item['prompt'] = patterns['prompt']['final_prompt']
    
    # Include color information
    if 'colors' in result:
        item['colors'] = []
        for color in result['colors'].get('dominant_colors', [])[:5]:  # Top 5 colors
            item['colors'].append({
                'name': color.get('name', ''),
                'hex': color.get('hex', ''),
                'proportion': color.get('proportion', 0.0)
            })
    
    # Add score components if available
    if all(k in result for k in ['theme_score', 'content_score', 'style_score', 'pattern_score', 'color_score', 'other_score']):
        item['score_components'] = {
            'theme_score': result.get('theme_score', 0.0),
            'content_score': result.get('content_score', 0.0),
            'style_score': result.get('style_score', 0.0),
            'pattern_score': result.get('pattern_score', 0.0),
            'color_score': result.get('color_score', 0.0),
            'other_score': result.get('other_score', 0.0)
        }
    
    return item 