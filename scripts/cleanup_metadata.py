#!/usr/bin/env python
"""
Script to clean up metadata for missing files.
This helps fix gallery issues where deleted images still appear in search results.
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.search_engine import SearchEngine
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run the cleanup operation and print results."""
    print("Starting metadata cleanup...")
    
    # Initialize search engine
    search_engine = SearchEngine()
    
    # Get initial metadata count
    initial_count = len(search_engine.metadata)
    print(f"Initial metadata count: {initial_count} entries")
    
    # Run the cleanup operation
    print("Cleaning up missing files...")
    cleaned_count = search_engine.cleanup_missing_files()
    
    # Get final metadata count
    final_count = len(search_engine.metadata)
    
    # Print results
    print("\nCleanup Results:")
    print(f"- Cleaned up {cleaned_count} entries for missing files")
    print(f"- Initial metadata count: {initial_count} entries")
    print(f"- Final metadata count: {final_count} entries")
    
    if cleaned_count > 0:
        print("\n✅ Cleanup successful! The gallery should now display correctly.")
    else:
        print("\n✅ No issues found. All metadata entries have corresponding files.")

if __name__ == "__main__":
    main() 