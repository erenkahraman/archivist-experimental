#!/usr/bin/env python3
"""
Archivist - Image processing and search system with AI-powered analysis

Main application entry point for running the Archivist service.
"""
import logging
import sys
from src.app import start_app
from src.logging_config import configure_logging

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)

def main():
    """Main application entry point."""
    try:
        logger.info("Starting Archivist application")
        start_app()
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
        return 0
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main()) 