#!/usr/bin/env python3
"""
Archivist - Flask application for image processing and search.

This file serves as an entry point for the Flask development server.
For production deployment, use a WSGI server like Gunicorn instead.
"""
from src.app import create_app, DEBUG

# Create the application instance
app = create_app()

if __name__ == '__main__':
    # Start the development server
    app.run(debug=DEBUG, host='0.0.0.0', port=8000) 