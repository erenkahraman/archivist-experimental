#!/usr/bin/env python
"""
Script to reindex all images with proper CLIP embeddings.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add the src directory to the path so we can import from it
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.utils import get_search_engine
from src.config import load_config

def main():
    parser = argparse.ArgumentParser(description="Reindex all images with CLIP embeddings")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger = logging.getLogger("reindex_embeddings")
    logger.info("Starting reindexing process...")
    
    # Load config and get search engine
    load_config()
    search_engine = get_search_engine()
    
    # Run the reindexing
    result = search_engine.reindex_all_with_embeddings()
    
    if result:
        logger.info("Successfully reindexed all images with embeddings")
        return 0
    else:
        logger.error("Failed to reindex images with embeddings")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 