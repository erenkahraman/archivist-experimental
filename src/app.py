import logging
from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path
import os
from search_engine import SearchEngine
import config

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Suppress verbose logs from other libraries
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING)

# Optionally disable Werkzeug logs for cleaner output
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.disabled = True

# ... rest of app.py code ... 