#!/usr/bin/env python3
"""
Elasticsearch Health Check Script

This script checks the health and performance of your Elasticsearch cluster
and provides recommendations for optimization.

Usage:
    python es_health_check.py
"""

import sys
import os
import logging
from pathlib import Path
import json

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the monitor and other needed components
from src.utils.es_monitor import ElasticsearchMonitor
from src.search.elasticsearch_client import ElasticsearchClient
import config

def main():
    """Run the Elasticsearch health check"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run the monitor
    monitor = ElasticsearchMonitor()
    monitor.print_health_report()
    
    # Now check for orphaned documents
    print("\n=== CHECKING FOR ORPHANED DOCUMENTS ===")
    check_orphaned_documents()

def check_orphaned_documents():
    """Check for documents in Elasticsearch that don't correspond to files on disk"""
    try:
        # Initialize Elasticsearch client
        es_client = ElasticsearchClient(
            hosts=config.ELASTICSEARCH_HOSTS,
            cloud_id=config.ELASTICSEARCH_CLOUD_ID,
            api_key=config.ELASTICSEARCH_API_KEY,
            username=config.ELASTICSEARCH_USERNAME,
            password=config.ELASTICSEARCH_PASSWORD
        )
        
        if not es_client.is_connected():
            print("ERROR: Cannot connect to Elasticsearch")
            return
            
        # Get total document count
        count_result = es_client.client.count(index=es_client.index_name)
        total_docs = count_result["count"]
        print(f"Total documents in index: {total_docs}")
        
        # Scroll through all documents
        query_body = {
            "query": {
                "match_all": {}
            },
            "_source": ["id", "path", "filename"]  # Only get fields we need
        }
        
        # Initialize scroll
        scroll_response = es_client.client.search(
            index=es_client.index_name, 
            body=query_body,
            scroll="2m",  # Keep the search context open for 2 minutes
            size=100  # Get 100 documents at a time
        )
        
        # Get the scroll ID
        scroll_id = scroll_response["_scroll_id"]
        
        # First set of hits
        hits = scroll_response["hits"]["hits"]
        
        orphaned_docs = []
        total_checked = 0
        upload_dir = config.UPLOAD_DIR
        
        # Process hits until there are no more
        while len(hits) > 0:
            total_checked += len(hits)
            print(f"Checking {len(hits)} documents ({total_checked}/{total_docs})...")
            
            for hit in hits:
                doc = hit["_source"]
                doc_id = hit["_id"]
                
                # Check if file exists
                file_path = None
                
                # Try to get path from document
                if "path" in doc:
                    file_path = Path(upload_dir) / doc["path"]
                elif "filename" in doc:
                    # Try to find by filename (less reliable)
                    file_path = Path(upload_dir) / doc["filename"]
                
                if file_path and not file_path.exists():
                    orphaned_docs.append({
                        "id": doc_id,
                        "path": str(file_path),
                        "source": doc
                    })
            
            # Get next batch of hits
            scroll_response = es_client.client.scroll(
                scroll_id=scroll_id,
                scroll="2m"  # Keep the search context open for another 2m
            )
            
            # Update scroll ID and hits
            scroll_id = scroll_response["_scroll_id"]
            hits = scroll_response["hits"]["hits"]
        
        # Clear the scroll to free resources
        es_client.client.clear_scroll(scroll_id=scroll_id)
        
        # Report findings
        if orphaned_docs:
            print(f"\nFound {len(orphaned_docs)} orphaned documents that don't correspond to files on disk:")
            for doc in orphaned_docs:
                print(f"  - ID: {doc['id']}, Path: {doc['path']}")
                
            # Offer to delete orphaned documents
            choice = input("\nDo you want to delete these orphaned documents? (y/n): ")
            if choice.lower() == 'y':
                deleted = 0
                for doc in orphaned_docs:
                    try:
                        es_client.delete_document(doc['id'])
                        deleted += 1
                    except Exception as e:
                        print(f"Error deleting document {doc['id']}: {str(e)}")
                
                print(f"Deleted {deleted} orphaned documents.")
                
                # Check search results for "paisley" after cleanup
                print("\nChecking search results for 'paisley' after cleanup:")
                results = es_client.search(query="paisley")
                print(f"Found {len(results)} results for 'paisley'")
                
                # Show document IDs
                for result in results:
                    print(f"  - ID: {result.get('id')}, Path: {result.get('path')}, Filename: {result.get('filename')}")
        else:
            print("No orphaned documents found. All indexed documents correspond to files on disk.")
            
            # Check search results for "paisley"
            print("\nChecking search results for 'paisley':")
            results = es_client.search(query="paisley")
            print(f"Found {len(results)} results for 'paisley'")
            
            # Show document IDs
            for result in results:
                print(f"  - ID: {result.get('id')}, Path: {result.get('path')}, Filename: {result.get('filename')}")
    
    except Exception as e:
        print(f"Error checking for orphaned documents: {str(e)}")

if __name__ == "__main__":
    main() 