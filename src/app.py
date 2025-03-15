import logging
from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path
import uuid
import os
from search_engine import SearchEngine
import config
from .tasks import process_image_task
import time
from .logging_config import configure_logging

# Configure logging at application startup
configure_logging()

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

@app.route('/api/upload', methods=['POST'])
def upload_image():
    """Upload and process an image."""
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file part"})
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "error": "No selected file"})
            
        # Save the uploaded file
        filename = str(uuid.uuid4()) + '_' + str(int(time.time())) + '.png'
        image_path = config.UPLOAD_DIR / filename
        file.save(image_path)
        
        # Process asynchronously
        task = process_image_task.delay(str(image_path))
        
        return jsonify({
            "success": True,
            "message": "Image uploaded and processing started",
            "task_id": task.id,
            "image_path": str(image_path)
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/task/<task_id>', methods=['GET'])
def check_task(task_id):
    """Check the status of an async task."""
    try:
        task = process_image_task.AsyncResult(task_id)
        
        if task.state == 'PENDING':
            response = {
                "state": task.state,
                "status": "Task is pending"
            }
        elif task.state == 'FAILURE':
            response = {
                "state": task.state,
                "status": "Task failed",
                "error": str(task.info)
            }
        else:
            response = {
                "state": task.state,
                "status": "Task completed" if task.state == 'SUCCESS' else "Task in progress",
                "result": task.result if task.state == 'SUCCESS' else None
            }
            
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}) 