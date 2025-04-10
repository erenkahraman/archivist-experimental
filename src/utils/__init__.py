"""
Utility functions and helpers for the application
"""
from .logging_config import configure_logging
from .embedding_utils import get_embedding_for_image_id

__all__ = ['configure_logging', 'get_embedding_for_image_id']

# This directory contains utility functions for the Archivist application
