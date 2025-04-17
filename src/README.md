# Archivist Backend Structure

This document describes the modular structure of the Archivist backend.

## Directory Structure

```
src/
├── __init__.py
├── app.py                # Main application entry point
├── api/                  # API routes and controllers
│   ├── __init__.py
│   └── routes.py         # API endpoints
├── analyzers/            # Analysis modules
│   ├── __init__.py
│   ├── gemini_analyzer.py# Optimized Gemini AI integration
│   └── json_parser.py    # JSON response parser
├── config/               # Configuration
│   ├── __init__.py
│   ├── config.py         # Configuration settings
│   ├── pattern_database.json # Pattern references
│   └── prompts.py        # Token-optimized prompts
├── storage/              # Storage management
│   └── image_storage.py  # Image and metadata storage
└── utils/                # Utility functions
    ├── __init__.py
    └── logging_config.py # Logging configuration
```

## Module Descriptions

### `app.py`

The main application entry point. Creates and configures the Flask application.

### `api/`

Contains all API routes and controllers for the application.

### `analyzers/`

Contains modules for analyzing images.

- `gemini_analyzer.py`: Token-optimized integration with Google's Gemini AI for pattern analysis.
- `json_parser.py`: Efficient JSON parsing utility for AI responses.

### `config/`

Contains configuration files.

- `prompts.py`: Token-efficient prompt templates for the Gemini API.
- `pattern_database.json`: Reference database for enhancing pattern recognition.

### `storage/`

Contains file storage management.

- `image_storage.py`: Handles image and metadata storage.

### `utils/`

Contains utility functions.

- `logging_config.py`: Configures logging for the application.

## Usage

To start the application:

```python
from src.app import start_app

if __name__ == "__main__":
    start_app()
``` 