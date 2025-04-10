from flask import Blueprint, jsonify, request
import logging
import os
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add the project root to the path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.config import DEFAULT_SEARCH_LIMIT, DEFAULT_MIN_SIMILARITY
from src.search.search_engine import search_engine
from src.utils.embedding_utils import get_embedding_for_image_id

# Set up logging
logger = logging.getLogger(__name__)
DEBUG = os.environ.get('DEBUG', 'False').lower() in ('true', '1', 't')

# Use a different name for the blueprint to avoid conflicts
search_blueprint = Blueprint('search', __name__)

# Helper function to enhance search results with metadata
def enhance_search_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Enhance search results with additional metadata and formatting
    
    Args:
        results: List of search results from Elasticsearch
        
    Returns:
        Enhanced search results with normalized pattern information
    """
    enhanced_results = []
    
    for result in results:
        # Create a copy to avoid modifying the original
        enhanced = result.copy()
        
        # Make sure pattern information is properly formatted
        if 'patterns' in enhanced:
            patterns = enhanced['patterns']
            
            # If patterns is a dict, it's already properly formatted
            if isinstance(patterns, dict):
                # Ensure primary pattern is accessible in simplified format for UI display
                if 'primary_pattern' in patterns and patterns['primary_pattern']:
                    if not enhanced.get('pattern'):
                        enhanced['pattern'] = {
                            'primary': patterns['primary_pattern'],
                            'confidence': patterns.get('pattern_confidence', 0.95)
                        }
            # If it's a string (nested field), convert to proper format
            elif isinstance(patterns, str):
                try:
                    # Try to parse if it's a JSON string
                    parsed_patterns = json.loads(patterns)
                    enhanced['patterns'] = parsed_patterns
                    
                    # Set simplified pattern info
                    if not enhanced.get('pattern') and parsed_patterns.get('primary_pattern'):
                        enhanced['pattern'] = {
                            'primary': parsed_patterns['primary_pattern'],
                            'confidence': parsed_patterns.get('pattern_confidence', 0.95)
                        }
                except:
                    # Not JSON, just use as primary pattern
                    enhanced['patterns'] = {
                        'primary_pattern': patterns,
                        'pattern_confidence': 0.95
                    }
                    
                    # Set simplified pattern info
                    if not enhanced.get('pattern'):
                        enhanced['pattern'] = {
                            'primary': patterns,
                            'confidence': 0.95
                        }
        else:
            # No patterns at all, create default
            enhanced['patterns'] = {
                'primary_pattern': 'Unknown',
                'pattern_confidence': 0.5
            }
            
            # Set simplified pattern info for UI
            if not enhanced.get('pattern'):
                enhanced['pattern'] = {
                    'primary': 'Unknown',
                    'confidence': 0.5
                }
                
        # Make sure colors are properly formatted
        if 'colors' in enhanced and isinstance(enhanced['colors'], dict):
            # Already properly formatted, do nothing
            pass
        else:
            # Colors missing or malformatted, create default
            if 'colors' not in enhanced or not enhanced['colors']:
                enhanced['colors'] = {
                    'dominant_colors': []
                }
            elif isinstance(enhanced['colors'], list):
                enhanced['colors'] = {
                    'dominant_colors': enhanced['colors']
                }
        
        # Add to enhanced results
        enhanced_results.append(enhanced)
    
    return enhanced_results

@search_blueprint.route('/search', methods=['GET', 'POST'])
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
        # Log request details for debugging
        logger.info(f"Search request received. Method: {request.method}")
        if request.method == 'GET':
            logger.info(f"GET params: {request.args}")
        elif request.method == 'POST':
            logger.info(f"POST data: {request.json}")
        
        # Get search parameters from either JSON body (POST) or query params (GET)
        if request.method == 'POST':
            data = request.json or {}
            query = data.get('query', '').strip()
            try:
                limit = min(int(data.get('limit', 20)), 100)  # Cap at 100 results
            except (ValueError, TypeError):
                limit = 20
                
            try:
                min_similarity = max(0.0, min(float(data.get('min_similarity', 0.1)), 1.0))  # Between 0 and 1
            except (ValueError, TypeError):
                min_similarity = 0.1
        else:  # GET
            query = request.args.get('query', '').strip()
            try:
                limit = min(int(request.args.get('limit', 20)), 100)  # Cap at 100 results
            except (ValueError, TypeError):
                limit = 20
                
            try:
                min_similarity = max(0.0, min(float(request.args.get('min_similarity', 0.1)), 1.0))  # Between 0 and 1
            except (ValueError, TypeError):
                min_similarity = 0.1
        
        # Validate query
        if not query:
            return jsonify({'error': 'Search query is required'}), 400
            
        # Check if Elasticsearch is available
        if not search_engine.use_elasticsearch or not search_engine.es_client.is_connected():
            logger.warning("Elasticsearch is not available, falling back to direct metadata search")
            # Use the direct metadata search instead
            results = search_engine.metadata_search(query, limit, min_similarity)
            
            # Enhance results with additional metadata
            enhanced_results = enhance_search_results(results)
            
            # Build response
            response = {
                "query": query,
                "result_count": len(enhanced_results),
                "results": enhanced_results,
                "search_method": "metadata_direct"  # Indicate search method used
            }
            
            logger.info(f"Direct metadata search returned {len(enhanced_results)} results for query: '{query}'")
            return jsonify(response)
            
        # Log search request
        if DEBUG:
            logger.info(f"Search request: query='{query}', limit={limit}, min_similarity={min_similarity}")
        
        try:
            # Perform search with Elasticsearch
            results = search_engine.es_client.search(query, limit, min_similarity)
            
            # If Elasticsearch returned no results, try direct metadata search as fallback
            if not results:
                logger.info(f"Elasticsearch returned no results for '{query}', trying direct metadata search")
                results = search_engine.metadata_search(query, limit, min_similarity)
            
            # Enhance results with additional metadata
            enhanced_results = enhance_search_results(results)
            
            # Build response
            response = {
                "query": query,
                "result_count": len(enhanced_results),
                "results": enhanced_results
            }
            
            logger.info(f"Search returned {len(enhanced_results)} results for query: '{query}'")
            return jsonify(response)
        
        except Exception as e:
            # Handle any exception during search by falling back to direct metadata search
            logger.error(f"Search engine error: {str(e)}, falling back to direct metadata search")
            
            # Use direct metadata search as final fallback
            results = search_engine.metadata_search(query, limit, min_similarity)
            enhanced_results = enhance_search_results(results)
            
            response = {
                "query": query,
                "result_count": len(enhanced_results),
                "results": enhanced_results,
                "search_method": "metadata_direct_fallback"  # Indicate search method used
            }
            
            logger.info(f"Fallback direct metadata search returned {len(enhanced_results)} results for query: '{query}'")
            return jsonify(response)
            
    except Exception as e:
        logger.error(f"Error in search: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@search_blueprint.route('/generate-prompt', methods=['POST'])
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

@search_blueprint.route('/similar/<path:filename>', methods=['GET', 'POST'])
def similar_by_id(filename):
    """
    Find images similar to a specific image by its ID or filename using Elasticsearch.
    
    The similarity is based on the image's patterns, colors, and other visual features.
    
    Parameters (in URL path):
    - filename: The filename or ID of the reference image
    
    Query parameters:
    - limit: Maximum number of results (optional, default: 20)
    - min_similarity: Minimum similarity score threshold (optional, default: 0.1)
    - sort: Whether to sort by similarity score (optional, default: true)
    - use_embedding: Whether to use vector embedding for similarity (optional, default: true)
    - use_text: Whether to use text for similarity (optional, default: true)
    - image_weight: Weight for image similarity in hybrid search (optional, default: 0.7)
    - text_weight: Weight for text similarity in hybrid search (optional, default: 0.3)
    
    Returns:
    - List of similar images with their metadata and similarity scores
    """
    try:
        # Support both query parameters and JSON body
        if request.method == 'POST' and request.is_json:
            data = request.json
            limit = min(int(data.get('limit', DEFAULT_SEARCH_LIMIT)), 100)
            min_similarity = max(0.0, min(float(data.get('min_similarity', DEFAULT_MIN_SIMILARITY)), 1.0))
            sort_by_similarity = data.get('sort', True)
            use_embedding = data.get('use_embedding', True)
            use_text = data.get('use_text', True)
            image_weight = float(data.get('image_weight', 0.7))
            text_weight = float(data.get('text_weight', 0.3))
        else:
            # Get query parameters
            limit = min(int(request.args.get('limit', DEFAULT_SEARCH_LIMIT)), 100)  # Cap at 100 results
            min_similarity = max(0.0, min(float(request.args.get('min_similarity', DEFAULT_MIN_SIMILARITY)), 1.0))
            sort_by_similarity = request.args.get('sort', 'true').lower() == 'true'
            
            # New parameters for controlling search behavior
            use_embedding = request.args.get('use_embedding', 'true').lower() == 'true'
            use_text = request.args.get('use_text', 'true').lower() == 'true'
            image_weight = float(request.args.get('image_weight', 0.7))
            text_weight = float(request.args.get('text_weight', 0.3))
        
        if DEBUG:
            logger.info(f"Similar search request for: {filename}, limit={limit}, min_similarity={min_similarity}, "
                       f"sort={sort_by_similarity}, use_embedding={use_embedding}, use_text={use_text}, "
                       f"image_weight={image_weight}, text_weight={text_weight}")
        
        # Verify Elasticsearch is available
        if not search_engine.es_client or not search_engine.es_client.is_connected():
            logger.error("Elasticsearch not available for similarity search")
            return jsonify({
                'error': 'Elasticsearch is required for similarity search and is not available',
                'results': [], 
                'result_count': 0
            }), 503
            
        # First, get metadata for the reference image
        ref_metadata = None
        ref_id = None
        
        # Clean up the filename - sometimes it comes with URL encoding or extra parameters
        clean_filename = filename.split('?')[0]
        
        # Try different ways to look up the reference image
        try:
            # First try to get it from Elasticsearch by ID
            ref_metadata = search_engine.es_client.get_document(clean_filename)
            if ref_metadata:
                ref_id = clean_filename
                logger.info(f"Found reference image: {ref_id} ({ref_metadata.get('filename', 'unknown')})")
        except Exception as e:
            logger.warning(f"Error getting reference image metadata: {str(e)}")
            
        if not ref_metadata:
            # Try alternative lookup by filename
            try:
                ref_id = clean_filename
                # Search for documents with matching filename
                search_results = search_engine.es_client.search(f"filename:{clean_filename}", limit=1, min_similarity=0)
                if search_results and len(search_results) > 0:
                    ref_metadata = search_results[0]
                    logger.info(f"Found reference image metadata: {ref_id}")
            except Exception as e:
                logger.warning(f"Error getting reference image by filename: {str(e)}")
                
        # If still not found, create minimal metadata
        if not ref_metadata:
            logger.warning(f"Could not find reference image metadata for {clean_filename}")
            ref_id = clean_filename
            ref_metadata = {
                'id': ref_id,
                'filename': clean_filename
            }
        
        # Perform the similarity search
        search_results = None
        error_msg = None
        
        try:
            if use_embedding:
                # Get the embedding for the reference image
                embedding = get_embedding_for_image_id(ref_id)
                
                if embedding is None:
                    logger.warning(f"No embedding found for reference image: {ref_id}")
                    
                    if use_text:
                        # Try text fallback using patterns from the reference
                        pattern_query = ""
                        if ref_metadata and 'patterns' in ref_metadata:
                            patterns = ref_metadata.get('patterns', {})
                            primary = patterns.get('primary_pattern', '')
                            main_theme = patterns.get('main_theme', '')
                            keywords = patterns.get('keywords', [])
                            
                            # Construct a more detailed query using all available pattern info
                            query_parts = []
                            if primary:
                                query_parts.append(primary)
                            if main_theme and main_theme != primary:
                                query_parts.append(main_theme)
                            
                            # Add up to 3 most relevant keywords
                            top_keywords = keywords[:3] if keywords else []
                            query_parts.extend(top_keywords)
                            
                            pattern_query = " ".join(query_parts)
                                
                        if pattern_query:
                            logger.info(f"Using text fallback with pattern: {pattern_query}")
                            search_results = search_engine.es_client.search(pattern_query, limit, min_similarity)
                        else:
                            error_msg = f"No embedding and no pattern found for reference image: {ref_id}"
                    else:
                        error_msg = f"No embedding found for reference image: {ref_id}"
                else:
                    # Use the embedding for hybrid search (vector + text)
                    logger.info(f"Searching for similar images using embedding from: {ref_id}")
                    
                    # Get text query from reference image metadata for hybrid search
                    text_query = ""
                    if ref_metadata and 'patterns' in ref_metadata:
                        patterns = ref_metadata.get('patterns', {})
                        primary = patterns.get('primary_pattern', '')
                        main_theme = patterns.get('main_theme', '')
                        keywords = ' '.join(patterns.get('keywords', []))
                        
                        # Construct text query from patterns
                        text_parts = []
                        if primary:
                            text_parts.append(primary)
                        if main_theme and main_theme != primary:
                            text_parts.append(main_theme)
                        if keywords:
                            text_parts.append(keywords)
                            
                        text_query = " ".join(text_parts)
                    
                    # Do hybrid search combining vector and text
                    if use_text and text_query:
                        logger.info(f"Performing hybrid search with text: '{text_query}'")
                        search_results = search_engine.es_client._hybrid_search(
                            embedding=embedding, 
                            text_query=text_query,
                            limit=limit, 
                            min_similarity=min_similarity,
                            exclude_id=ref_id,  # Exclude the reference image itself
                            text_weight=text_weight, 
                            vector_weight=image_weight
                        )
                    else:
                        # Pure vector search if text search is disabled or no text available
                        logger.info("Performing pure vector search")
                        search_results = search_engine.es_client.search_by_vector(
                            embedding=embedding,
                            limit=limit,
                            min_similarity=min_similarity
                        )
            else:
                error_msg = f"No embedding used, using text-based search"
                
        except Exception as e:
            error_msg = f"Error during similarity search: {str(e)}"
            logger.exception(f"Error during similarity search: {str(e)}")
            
        # Handle error cases
        if error_msg:
            logger.error(error_msg)
            return jsonify({
                'error': error_msg,
                'results': [],
                'result_count': 0
            }), 500
            
        # If no results, return empty list
        if not search_results:
            logger.warning(f"No similar images found for reference: {ref_id}")
            return jsonify({
                'reference_id': ref_id,
                'query': f"similar_to:{ref_id}",
                'results': [],
                'result_count': 0,
                'search_info': {
                    'min_similarity': float(min_similarity),
                    'embedding_type': 'text-based',
                    'text_weight': float(text_weight),
                    'vector_weight': float(image_weight)
                }
            })
            
        # Enhance results with additional metadata
        enhanced_results = enhance_search_results(search_results)
        
        # Return the response with results
        response = {
            'reference_id': ref_id,
            'query': f"similar_to:{ref_id}",
            'result_count': len(enhanced_results),
            'results': enhanced_results,
            'search_info': {
                'min_similarity': float(min_similarity),
                'embedding_type': 'text-based',
                'text_weight': float(text_weight),
                'vector_weight': float(image_weight)
            }
        }
        
        # Sort by similarity if requested
        if sort_by_similarity:
            response['results'].sort(key=lambda x: x.get('similarity', 0), reverse=True)
            
        logger.info(f"Returning {len(enhanced_results)} similar images for {filename}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in similar_by_id: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500 