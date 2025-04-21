#!/usr/bin/env python3
"""
Archivist - Flask application for image processing and search.

This file serves as an entry point for the Flask development server.
For production deployment, use a WSGI server like Gunicorn instead.
"""
from src.app import create_app
import os

# Set debug mode
DEBUG = os.environ.get('DEBUG', 'False').lower() in ('true', '1', 't', 'yes')

# Create the application instance
app = create_app()

if __name__ == '__main__':
    try:
        # Use threaded mode for better concurrency
        app.run(debug=DEBUG, host='0.0.0.0', port=8000, threaded=True)
    except OSError as e:
        if 'Address already in use' in str(e):
            print('Port 8000 in use; starting on port 8001')
            app.run(debug=DEBUG, host='0.0.0.0', port=8001, threaded=True)
        else:
            raise 