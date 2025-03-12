import logging
from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path
import uuid
import os
from search_engine import SearchEngine
import config

# Werkzeug loglarını tamamen kapat
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.disabled = True

# Ana uygulama loglarını ayarla
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Diğer loglayıcıları sustur
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING)

# ... geri kalan app.py kodu ... 