"""Structured logging configuration."""
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
import config

def configure_logging():
    """Configure logging with rotating file handlers and console output"""
    # Create logs directory if it doesn't exist
    logs_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../logs')))
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    
    # Set level to INFO by default
    root_logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    if root_logger.handlers:
        root_logger.handlers.clear()
    
    # Create formatters
    verbose_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter('%(levelname)s - %(message)s')
    
    # Create rotating file handler for all logs
    all_log_file = logs_dir / 'application.log'
    file_handler = RotatingFileHandler(
        all_log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(verbose_formatter)
    
    # Create rotating file handler for errors only
    error_log_file = logs_dir / 'errors.log'
    error_file_handler = RotatingFileHandler(
        error_log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(verbose_formatter)
    
    # Create console handler for development
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Add handlers to the root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_file_handler)
    root_logger.addHandler(console_handler)
    
    # Set specific levels for verbose libraries
    logging.getLogger('elasticsearch').setLevel(logging.WARNING)
    logging.getLogger('elastic_transport').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    # Return the configured logger
    return logging.getLogger(__name__) 