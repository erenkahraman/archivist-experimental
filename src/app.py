import logging
from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path
import uuid
import os
from search_engine import SearchEngine
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Suppress less important logs
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING)
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.setLevel(logging.WARNING)

# ... geri kalan app.py kodu ... 

@app.route('/api/debug/metadata', methods=['GET'])
def debug_metadata():
    """Debug endpoint to check metadata structure"""
    try:
        # Get the first few items from metadata
        sample_metadata = list(search_engine.metadata.values())[:3]
        return jsonify({
            "success": True,
            "metadata_sample": sample_metadata
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }) 