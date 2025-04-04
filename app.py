import logging
from flask import Flask, jsonify
from flask_cors import CORS

from src.api import api, create_app, DEBUG
from src.analyzers.gemini_analyzer import GeminiAnalyzer
from src.analyzers.color_analyzer import ColorAnalyzer
from src.search_engine import search_engine

# Configure logger
logger = logging.getLogger(__name__)

# Create the Flask application with proper CORS config
app = create_app()

@app.route('/')
def index():
    """Root route for the API."""
    return jsonify({
        "status": "ok",
        "api_version": "1.0.0",
        "debug_mode": DEBUG,
        "features": {
            "pattern_analysis": search_engine.gemini_analyzer is not None,
            "color_analysis": search_engine.color_analyzer is not None,
            "elasticsearch": search_engine.use_elasticsearch,
            "cache": search_engine.cache.enabled
        }
    })

if __name__ == '__main__':
    # Start the development server
    app.run(debug=DEBUG, host='0.0.0.0', port=8000) 