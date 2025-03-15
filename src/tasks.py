"""Celery tasks for asynchronous processing."""
from .celery_app import celery_app
from pathlib import Path
import logging
from .search_engine import SearchEngine

logger = logging.getLogger(__name__)

@celery_app.task(bind=True, max_retries=3)
def process_image_task(self, image_path_str: str) -> dict:
    """
    Process an image asynchronously.
    
    Args:
        image_path_str: String path to the image file
        
    Returns:
        Processed image metadata
    """
    try:
        image_path = Path(image_path_str)
        logger.info(f"Starting async processing of image: {image_path}")
        
        search_engine = SearchEngine()
        metadata = search_engine.process_image(image_path)
        
        logger.info(f"Completed async processing of image: {image_path}")
        return metadata
        
    except Exception as e:
        logger.error(f"Error in async image processing: {str(e)}")
        self.retry(exc=e, countdown=10)  # Retry after 10 seconds 