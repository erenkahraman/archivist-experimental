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
├── core/                 # Core functionality
│   ├── __init__.py
│   └── search_engine.py  # Search engine implementation
├── analyzers/            # Analysis modules
│   ├── __init__.py
│   ├── color_analyzer.py # Color analysis
│   └── gemini_analyzer.py# Gemini AI integration
├── config/               # Configuration
│   ├── __init__.py
│   └── prompts.py        # Prompt templates
└── utils/                # Utility functions
    ├── __init__.py
    └── logging_config.py # Logging configuration
```

## Module Descriptions

### `app.py`

The main application entry point. Creates and configures the Flask application.

### `api/`

Contains all API routes and controllers for the application.

- `routes.py`: Defines all API endpoints, including image upload, search, and metadata retrieval.

### `core/`

Contains core functionality of the application.

- `search_engine.py`: Implements the search engine that processes and analyzes images.

### `analyzers/`

Contains modules for analyzing images.

- `color_analyzer.py`: Analyzes colors in images.
- `gemini_analyzer.py`: Integrates with Google's Gemini AI for image analysis.

### `config/`

Contains configuration files.

- `prompts.py`: Defines prompt templates for AI models.

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