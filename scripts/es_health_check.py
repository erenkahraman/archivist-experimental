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

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the monitor
from src.utils.es_monitor import ElasticsearchMonitor

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

if __name__ == "__main__":
    main() 