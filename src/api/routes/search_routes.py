from flask import request, jsonify
import logging
import sys
from pathlib import Path

# Add the project root to the path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from .. import api, search_engine, DEBUG

logger = logging.getLogger(__name__)

@api.route('/search', methods=['POST'])
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
        # Get search parameters
        data = request.json or {}
        query = data.get('query', '').strip()
        limit = min(int(data.get('limit', 20)), 100)  # Cap at 100 results
        min_similarity = max(0.0, min(float(data.get('min_similarity', 0.1)), 1.0))  # Between 0 and 1
        
        # Validate query
        if not query:
            return jsonify({'error': 'Search query is required'}), 400
            
        # Check if Elasticsearch is available
        if not search_engine.use_elasticsearch or not search_engine.es_client.is_connected():
            return jsonify({
                'error': 'Search service is temporarily unavailable. Please try again later.',
                'details': 'Elasticsearch is not available'
            }), 503
            
        # Log search request
        if DEBUG:
            logger.info(f"Search request: query='{query}', limit={limit}, min_similarity={min_similarity}")
        
        try:
            # Perform search using Elasticsearch
            results = search_engine.search(query, k=limit)
            
            # Filter by minimum similarity if specified
            if min_similarity > 0:
                results = [r for r in results if r.get('similarity', 0) >= min_similarity]
            
            # Format the results for the response
            formatted_results = []
            for result in results:
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
                    item['pattern'] = {
                        'primary': result['patterns'].get('primary_pattern', 'Unknown'),
                        'confidence': result['patterns'].get('pattern_confidence', 0.0),
                        'secondary': [p.get('name') if isinstance(p, dict) else str(p) for p in result['patterns'].get('secondary_patterns', [])],
                        'elements': [e.get('name') if isinstance(e, dict) else str(e) for e in result['patterns'].get('elements', [])]
                    }
                    
                    # Include style keywords
                    item['style_keywords'] = result['patterns'].get('style_keywords', [])
                    
                    # Include prompt if available
                    if 'prompt' in result['patterns'] and 'final_prompt' in result['patterns']['prompt']:
                        item['prompt'] = result['patterns']['prompt']['final_prompt']
                
                # Include color information
                if 'colors' in result:
                    item['colors'] = []
                    for color in result['colors'].get('dominant_colors', [])[:5]:  # Top 5 colors
                        item['colors'].append({
                            'name': color.get('name', ''),
                            'hex': color.get('hex', ''),
                            'proportion': color.get('proportion', 0.0)
                        })
                
                # Add score components if available (for debugging)
                if DEBUG and 'raw_score' in result:
                    item['score_components'] = {
                        'raw_score': result.get('raw_score', 0.0)
                    }
                    
                formatted_results.append(item)
            
            return jsonify({
                'query': query,
                'result_count': len(formatted_results),
                'results': formatted_results
            })
        
        except RuntimeError as e:
            # Handle the case where Elasticsearch is required but not available
            logger.error(f"Search engine error: {str(e)}")
            return jsonify({
                'error': 'Search service is temporarily unavailable. Please try again later.',
                'details': str(e)
            }), 503
            
    except Exception as e:
        logger.error(f"Error in search: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

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

@api.route('/similar/<path:filename>', methods=['GET'])
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
    
    Returns:
    - List of similar images with their metadata and similarity scores
    """
    try:
        # Get query parameters
        limit = min(int(request.args.get('limit', config.DEFAULT_SEARCH_LIMIT)), 100)  # Cap at 100 results
        min_similarity = max(0.0, min(float(request.args.get('min_similarity', config.DEFAULT_MIN_SIMILARITY)), 1.0))
        sort_by_similarity = request.args.get('sort', 'true').lower() == 'true'
        
        if DEBUG:
            logger.info(f"Similar search request for: {filename}, limit={limit}, min_similarity={min_similarity}, sort={sort_by_similarity}")
        
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
        clean_filename = filename.split('?')[0]  # Remove URL parameters if present
        
        # Try different ways to match the reference image
        for path, meta in search_engine.metadata.items():
            # Match by filename, id, or path ending with the filename
            if path.endswith(clean_filename) or meta.get('filename') == clean_filename or meta.get('id') == clean_filename:
                ref_metadata = meta
                ref_id = meta.get('id')
                logger.info(f"Found reference image: {ref_id} ({meta.get('filename')})")
                break
        
        if not ref_metadata:
            logger.warning(f"Reference image not found: {filename}")
            return jsonify({
                'error': f'Reference image not found: {filename}', 
                'results': [], 
                'result_count': 0
            }), 404
            
        logger.info(f"Found reference image metadata: {ref_id}")
        
        # Perform the similarity search
        try:
            # Get the text query parameters from the metadata
            text_query = None
            if ref_metadata.get('patterns'):
                patterns = ref_metadata.get('patterns', {})
                primary_pattern = patterns.get('primary_pattern')
                main_theme = patterns.get('main_theme')
                if primary_pattern and main_theme:
                    text_query = f"{primary_pattern} {main_theme}"
                elif primary_pattern:
                    text_query = primary_pattern
                elif main_theme:
                    text_query = main_theme
                    
                # Add style keywords if available
                style_keywords = patterns.get('style_keywords', [])
                if style_keywords and isinstance(style_keywords, list) and len(style_keywords) > 0:
                    # Limit to first 3 keywords to avoid overspecific queries
                    keywords_str = " ".join(style_keywords[:3])
                    if text_query:
                        text_query = f"{text_query} {keywords_str}"
                    else:
                        text_query = keywords_str
                        
            # Log the search parameters
            logger.info(f"Performing similarity search with text_query='{text_query}', exclude_id={ref_id}")
            
            # Execute the search
            results = search_engine.es_client.find_similar(
                text_query=text_query,
                exclude_id=ref_id,
                limit=limit,
                min_similarity=min_similarity
            )
            
            # Check results
            if not results:
                logger.info(f"No similar images found for {filename}")
                return jsonify({
                    'results': [],
                    'result_count': 0,
                    'reference_id': ref_id
                })
                
            # Format results for the response
            formatted_results = []
            for result in results:
                # Format each result with essential fields
                item = {
                    'id': result.get('id'),
                    'filename': result.get('filename'),
                    'path': result.get('path'),
                    'thumbnail_path': result.get('thumbnail_path'),
                    'similarity': result.get('similarity', 0.0)
                }
                
                # Include pattern information if available
                if 'patterns' in result:
                    item['pattern'] = {
                        'primary': result['patterns'].get('primary_pattern', 'Unknown'),
                        'confidence': result['patterns'].get('pattern_confidence', 0.0)
                    }
                    
                # Include color information if available
                if 'colors' in result and 'dominant_colors' in result['colors']:
                    item['colors'] = [
                        {
                            'name': color.get('name', ''),
                            'hex': color.get('hex', ''),
                            'proportion': color.get('proportion', 0.0)
                        }
                        for color in result['colors'].get('dominant_colors', [])[:3]  # Top 3 colors
                    ]
                    
                formatted_results.append(item)
                
            # Sort by similarity if requested
            if sort_by_similarity:
                formatted_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
                
            logger.info(f"Returning {len(formatted_results)} similar images for {filename}")
            
            return jsonify({
                'results': formatted_results,
                'result_count': len(formatted_results),
                'reference_id': ref_id
            })
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}", exc_info=True)
            return jsonify({
                'error': f'Error in similarity search: {str(e)}',
                'results': [],
                'result_count': 0
            }), 500
            
    except Exception as e:
        logger.error(f"Error in similar_by_id: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500 