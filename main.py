#!/usr/bin/env python3
"""
Main entry point for the Archivist Pattern Analysis System.
"""
import logging
import os
import dotenv
from pathlib import Path

# Load environment variables from .env file
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info("Starting Archivist application")
    
    # Import and run Flask app
    from src.app import app
    
    # Default port and host
    port = int(os.environ.get('PORT', 8000))
    host = os.environ.get('HOST', '0.0.0.0')
    
    # Start server
    app.run(host=host, port=port, debug=True) 