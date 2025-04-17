from flask import Blueprint, jsonify, request
import logging
import os
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add the project root to the path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.config import DEFAULT_SEARCH_LIMIT, DEFAULT_MIN_SIMILARITY, MAX_SEARCH_RESULTS
from src.search import search_engine

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
        results: List of search results from search engine
        
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
                limit = min(int(data.get('limit', 20)), MAX_SEARCH_RESULTS)
            except (ValueError, TypeError):
                limit = 20
                
            try:
                min_similarity = max(0.0, min(float(data.get('min_similarity', 0.1)), 1.0))  # Between 0 and 1
            except (ValueError, TypeError):
                min_similarity = 0.1
        else:  # GET
            query = request.args.get('query', '').strip()
            try:
                limit = min(int(request.args.get('limit', 20)), MAX_SEARCH_RESULTS)
            except (ValueError, TypeError):
                limit = 20
                
            try:
                min_similarity = max(0.0, min(float(request.args.get('min_similarity', 0.1)), 1.0))  # Between 0 and 1
            except (ValueError, TypeError):
                min_similarity = 0.1
        
        # Validate query
        if not query:
            return jsonify({'error': 'Search query is required'}), 400
            
        # Log search request
        if DEBUG:
            logger.info(f"Search request: query='{query}', limit={limit}, min_similarity={min_similarity}")
        
        try:
            # Perform in-memory search
            results = search_engine.metadata_search(query, limit, min_similarity)
            
            # Enhance results with additional metadata
            enhanced_results = enhance_search_results(results)
            
            # Build response
            response = {
                "query": query,
                "result_count": len(enhanced_results),
                "results": enhanced_results,
                "search_method": "metadata_search"  # Indicate search method used
            }
            
            logger.info(f"Search returned {len(enhanced_results)} results for query: '{query}'")
            return jsonify(response)
        
        except Exception as e:
            logger.error(f"Search engine error: {str(e)}")
            return jsonify({'error': f"Error processing search: {str(e)}"}), 500
            
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
    Find images similar to a specific image by its ID or filename.
    This is a simplified implementation that uses pattern metadata for matching.
    
    Args:
        filename: The image filename or ID
        
    Parameters:
        - text_query: Optional text query to combine with pattern similarity (GET query param or POST field)
        - limit: Maximum number of results to return (GET query param or POST field)
        - min_similarity: Minimum similarity score (0-1) (GET query param or POST field)
        
    Returns:
        JSON response with similar images
    """
    try:
        # Get parameters from either POST body or GET query params
        if request.method == 'POST':
            data = request.json or {}
            text_query = data.get('text_query', '').strip()
            try:
                limit = int(data.get('limit', DEFAULT_SEARCH_LIMIT))
            except:
                limit = DEFAULT_SEARCH_LIMIT
                
            try:
                min_similarity = float(data.get('min_similarity', DEFAULT_MIN_SIMILARITY))
            except:
                min_similarity = DEFAULT_MIN_SIMILARITY
                
        else:  # GET
            text_query = request.args.get('text_query', '').strip()
            try:
                limit = int(request.args.get('limit', DEFAULT_SEARCH_LIMIT))
            except:
                limit = DEFAULT_SEARCH_LIMIT
                
            try:
                min_similarity = float(request.args.get('min_similarity', DEFAULT_MIN_SIMILARITY))
            except:
                min_similarity = DEFAULT_MIN_SIMILARITY
            
        # Clean up filename
        clean_filename = os.path.basename(filename)
        
        # Log parameters for debugging
        logger.info(f"Similar request: filename='{clean_filename}', text_query='{text_query}', " +
                   f"limit={limit}, min_similarity={min_similarity}")
        
        # Check if we have this image in our metadata
        ref_path = None
        
        # First try to find the image in our metadata
        for path, metadata in search_engine.metadata.items():
            if metadata.get('filename') == clean_filename or metadata.get('id') == clean_filename:
                ref_path = path
                break
        
        # If not found in metadata, check if it exists on disk
        if ref_path is None:
            # Check upload directory
            potential_paths = [
                os.path.join(os.environ.get('UPLOAD_DIR', 'uploads'), clean_filename),
                os.path.join(os.environ.get('UPLOAD_DIR', 'uploads'), 'processed', clean_filename)
            ]
            
            for path in potential_paths:
                if os.path.exists(path):
                    logger.info(f"Found image at path: {path}")
                    ref_path = path
                    break
                    
            if ref_path is None:
                return jsonify({'error': f"Image not found: {clean_filename}"}), 404
        
        # Get results using find_similar_images
        try:
            results = search_engine.find_similar_images(
                image_path=ref_path, 
                text_query=text_query,
                k=limit,
                exclude_source=True
            )
            
            # Filter by minimum similarity
            results = [r for r in results if r.get('similarity', 0) >= min_similarity]
            
            # Enhance results with additional metadata
            enhanced_results = enhance_search_results(results)
            
            # Prepare response
            response = {
                "query_image": clean_filename,
                "text_query": text_query,
                "result_count": len(enhanced_results),
                "results": enhanced_results
            }
            
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Error finding similar images: {str(e)}", exc_info=True)
            return jsonify({'error': f"Error finding similar images: {str(e)}"}), 500
        
    except Exception as e:
        logger.error(f"Error processing similar request: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500 