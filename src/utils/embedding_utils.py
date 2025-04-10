import logging
import os
import numpy as np
from typing import Optional, Dict, Any, Union
import json
from pathlib import Path

# Import config directly
from src.config.elasticsearch_config import (
    ELASTICSEARCH_HOSTS,
    ELASTICSEARCH_CLOUD_ID,
    ELASTICSEARCH_API_KEY,
    ELASTICSEARCH_USERNAME,
    ELASTICSEARCH_PASSWORD
)

# Set up logger
logger = logging.getLogger(__name__)

def get_embedding_for_image_id(image_id: str) -> Optional[np.ndarray]:
    """
    Retrieves the embedding vector for a given image ID.
    
    Args:
        image_id: The ID or path of the image to get the embedding for
        
    Returns:
        A numpy array containing the embedding vector, or None if not found
    """
    try:
        # Import here to avoid circular imports
        from src.search.elasticsearch_client import ElasticsearchClient
        
        # Create an ES client instance
        es_client = ElasticsearchClient(
            hosts=ELASTICSEARCH_HOSTS,
            cloud_id=ELASTICSEARCH_CLOUD_ID,
            api_key=ELASTICSEARCH_API_KEY,
            username=ELASTICSEARCH_USERNAME,
            password=ELASTICSEARCH_PASSWORD
        )
        
        # Check if ES is available
        if not es_client.is_connected():
            logger.error("Cannot retrieve embedding: Elasticsearch is not available")
            return None
            
        # Get the embedding from Elasticsearch
        image_data = es_client.get_document(image_id)
        
        if not image_data:
            logger.warning(f"No data found for image ID: {image_id}")
            return None
            
        # Extract the embedding from the document
        if 'embedding' in image_data:
            embedding = image_data['embedding']
            
            # If the embedding is stored as a list, convert to numpy array
            if isinstance(embedding, list):
                return np.array(embedding, dtype=np.float32)
            
            # If it's stored as a string or other format, try to parse it
            if isinstance(embedding, str):
                try:
                    # Try to parse as a comma-separated string
                    embedding_list = [float(x) for x in embedding.split(',')]
                    return np.array(embedding_list, dtype=np.float32)
                except ValueError:
                    logger.error(f"Failed to parse embedding string for image ID: {image_id}")
                    return None
        
        logger.warning(f"No embedding found in document for image ID: {image_id}")
        return None
        
    except Exception as e:
        logger.error(f"Error retrieving embedding for image ID '{image_id}': {str(e)}")
        return None 