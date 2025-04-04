from flask import request, jsonify
import logging
from .. import api, search_engine

logger = logging.getLogger(__name__)

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