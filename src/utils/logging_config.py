"""Structured logging configuration."""
import logging
import json
import sys
from pathlib import Path
import config

class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_record = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception info if available
        if record.exc_info:
            log_record['exception'] = self.formatException(record.exc_info)
            
        # Add custom fields if available
        if hasattr(record, 'image_path'):
            log_record['image_path'] = record.image_path
            
        if hasattr(record, 'pattern_type'):
            log_record['pattern_type'] = record.pattern_type
            
        if hasattr(record, 'processing_time'):
            log_record['processing_time'] = record.processing_time
            
        return json.dumps(log_record)

def configure_logging():
    """Configure structured logging for the application."""
    # Create logs directory if it doesn't exist
    log_dir = Path(config.BASE_DIR) / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    # Create formatters
    json_formatter = JsonFormatter()
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler for JSON logs
    file_handler = logging.FileHandler(log_dir / 'archivist.json.log')
    file_handler.setFormatter(json_formatter)
    root_logger.addHandler(file_handler)
    
    # Set lower log levels for noisy libraries
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('werkzeug').setLevel(logging.WARNING) 