from flask import request, jsonify
import logging
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
            
        # Log search request
        if DEBUG:
            logger.info(f"Search request: query='{query}', limit={limit}, min_similarity={min_similarity}")
        
        # Perform search using the enhanced search function
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
                    'secondary': [p.get('name') for p in result['patterns'].get('secondary_patterns', [])],
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
            if DEBUG and 'pattern_score' in result and 'color_score' in result and 'other_score' in result:
                item['score_components'] = {
                    'pattern_score': result.get('pattern_score', 0.0),
                    'color_score': result.get('color_score', 0.0),
                    'other_score': result.get('other_score', 0.0)
                }
                
            formatted_results.append(item)
        
        return jsonify({
            'query': query,
            'result_count': len(formatted_results),
            'results': formatted_results
        })
        
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