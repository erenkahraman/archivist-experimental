#!/usr/bin/env python

import json
import logging
import os
import sys
import time
from pathlib import Path

# Add the parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.search.elasticsearch_client import ElasticsearchClient

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_metadata():
    """Load metadata from file"""
    try:
        metadata_path = Path("metadata.json")
        if not metadata_path.exists():
            logger.error(f"Metadata file not found: {metadata_path}")
            return None
            
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading metadata: {e}")
        return None

def fix_timestamps(metadata):
    """Fix timestamp format in metadata"""
    fixed_count = 0
    
    # Iterate through all documents in metadata
    for key, doc in metadata.items():
        # Fix timestamps to ensure they are integers (not float or scientific notation)
        if 'timestamp' in doc:
            try:
                # Convert to integer seconds to match epoch_second format
                if isinstance(doc['timestamp'], float):
                    doc['timestamp'] = int(doc['timestamp'])
                    fixed_count += 1
                elif isinstance(doc['timestamp'], str):
                    # Try to convert string timestamp to int
                    doc['timestamp'] = int(float(doc['timestamp']))
                    fixed_count += 1
            except (ValueError, TypeError):
                # If conversion fails, set to current time
                logger.warning(f"Could not convert timestamp for {key}, using current time")
                doc['timestamp'] = int(time.time())
                fixed_count += 1
    
    logger.info(f"Fixed {fixed_count} timestamps in metadata")
    return metadata

def main():
    # Connect to Elasticsearch
    logger.info("Connecting to Elasticsearch...")
    es_client = ElasticsearchClient(hosts=["http://localhost:9200"])
    
    # Check connection
    if not es_client.is_connected():
        logger.error("Failed to connect to Elasticsearch")
        return False
    
    # Delete index if it exists
    logger.info("Deleting existing index...")
    if es_client.index_exists():
        if es_client.delete_index():
            logger.info("Index deleted successfully")
        else:
            logger.error("Failed to delete index")
            return False
    
    # Create new index with updated mappings
    logger.info("Creating index with new mappings...")
    if es_client.create_index():
        logger.info("Index created successfully")
    else:
        logger.error("Failed to create index")
        return False
    
    # Load metadata
    logger.info("Loading metadata...")
    metadata = load_metadata()
    if not metadata:
        logger.error("Failed to load metadata")
        return False
    
    # Fix timestamps in metadata
    logger.info("Fixing timestamps in metadata...")
    metadata = fix_timestamps(metadata)
    
    # Bulk index documents
    logger.info(f"Indexing {len(metadata)} documents...")
    docs_to_index = list(metadata.values())
    result = es_client.bulk_index(docs_to_index)
    
    if result:
        logger.info("Documents indexed successfully")
        return True
    else:
        logger.error("Failed to index documents")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 