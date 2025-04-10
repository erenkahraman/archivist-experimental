"""
Search engine module that interfaces with Elasticsearch for document retrieval and search.
"""
import logging
from typing import List, Dict, Any, Optional
import os

from .elasticsearch_client import ElasticsearchClient
from src.config.elasticsearch_config import (
    ELASTICSEARCH_HOSTS,
    ELASTICSEARCH_CLOUD_ID,
    ELASTICSEARCH_API_KEY,
    ELASTICSEARCH_USERNAME,
    ELASTICSEARCH_PASSWORD,
    INDEX_NAME
)

logger = logging.getLogger(__name__)

class SearchEngine:
    """Search engine class that manages search operations through Elasticsearch."""
    
    def __init__(self):
        """Initialize the search engine with elasticsearch client."""
        self.use_elasticsearch = True
        self.metadata = {}  # Placeholder for metadata
        
        try:
            # Initialize Elasticsearch client
            self.es_client = ElasticsearchClient(
                hosts=ELASTICSEARCH_HOSTS,
                cloud_id=ELASTICSEARCH_CLOUD_ID,
                api_key=ELASTICSEARCH_API_KEY,
                username=ELASTICSEARCH_USERNAME,
                password=ELASTICSEARCH_PASSWORD
            )
            # Set the index name separately - it's hardcoded in the client class
            logger.info("SearchEngine initialized with Elasticsearch client")
        except Exception as e:
            self.use_elasticsearch = False
            logger.error(f"Failed to initialize Elasticsearch client: {e}")

# Create a singleton instance
search_engine = SearchEngine() 