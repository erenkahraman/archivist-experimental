#!/usr/bin/env python
"""
Test script for the enhanced search functionality.
"""

import os
import sys
import json
import time
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.search_engine import SearchEngine
import config

def main():
    """Test the search functionality with various queries."""
    # Initialize search engine
    print("Initializing search engine...")
    search_engine = SearchEngine()
    
    if not search_engine.metadata:
        print("No metadata found. Please add images to the system first.")
        return
    
    print(f"Found {len(search_engine.metadata)} images in the metadata.")
    
    # Define test queries
    test_queries = [
        "paisley",
        "floral",
        "blue",
        "abstract geometric",
        "vintage",
        "intricate pattern"
    ]
    
    # Test each query
    for query in test_queries:
        print(f"\n=== Testing query: '{query}' ===")
        
        # Perform search
        start_time = time.time()
        results = search_engine.search(query, k=5)
        search_time = time.time() - start_time
        
        # Print results
        print(f"Search completed in {search_time:.2f}s, found {len(results)} results")
        
        if results:
            print("\nTop results:")
            for i, result in enumerate(results[:3]):
                print(f"{i+1}. {result.get('filename', 'unknown')} - Similarity: {result.get('similarity', 0):.2f}")
                print(f"   Main theme: {result.get('patterns', {}).get('main_theme', 'unknown')}")
                print(f"   Primary pattern: {result.get('patterns', {}).get('primary_pattern', 'unknown')}")
                
                # Print content details
                content_details = result.get('patterns', {}).get('content_details', [])
                if content_details:
                    print(f"   Content details: {', '.join([d.get('name', '') for d in content_details if isinstance(d, dict)])}")
                
                # Print top colors
                colors = result.get('colors', {}).get('dominant_colors', [])
                if colors:
                    print(f"   Top colors: {', '.join([c.get('name', '') for c in colors[:2] if isinstance(c, dict)])}")
        else:
            print("No results found.")
    
    # Test similarity search
    print("\n=== Testing similarity search ===")
    
    # Get a random document to use as reference
    if search_engine.metadata:
        reference_path = next(iter(search_engine.metadata.keys()))
        reference_doc = search_engine.metadata[reference_path]
        reference_id = reference_doc.get('id', reference_path)
        
        print(f"Finding images similar to: {reference_doc.get('filename', 'unknown')}")
        
        # Get the text description to use for similarity
        description = reference_doc.get('patterns', {}).get('prompt', {}).get('final_prompt', '')
        if description:
            print(f"Using description: '{description[:100]}...'")
            
            # Find similar images
            start_time = time.time()
            similar_results = search_engine.es_client.find_similar(
                text_query=description, 
                exclude_id=reference_id, 
                limit=5
            )
            search_time = time.time() - start_time
            
            # Print results
            print(f"Similarity search completed in {search_time:.2f}s, found {len(similar_results)} results")
            
            if similar_results:
                print("\nTop similar images:")
                for i, result in enumerate(similar_results[:3]):
                    print(f"{i+1}. {result.get('filename', 'unknown')} - Similarity: {result.get('similarity', 0):.2f}")
                    print(f"   Main theme: {result.get('patterns', {}).get('main_theme', 'unknown')}")

if __name__ == "__main__":
    main() 