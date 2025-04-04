# Archivist

A Python-based image processing and search system with AI-powered image analysis and advanced search capabilities.

## Features

- **Image Processing**: Analyze images for patterns, colors, and visual elements.
- **AI-Powered Analysis**: Integration with Google's Gemini API for advanced image understanding.
- **Powerful Search**: Find images based on patterns, colors, and visual elements.
- **Elasticsearch Integration**: Scalable search backend for efficient image retrieval.
- **API-First Design**: REST API for easy integration with frontend applications.

## Architecture

The project follows a modular architecture:

- **Core Engine**: Handles image processing, metadata extraction, and search functionality.
- **AI Integration**: Connects with Google's Gemini API for advanced image analysis.
- **Search Backend**: Uses Elasticsearch for efficient and scalable search capabilities.
- **RESTful API**: Flask-based API for interacting with the system.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/archivist.git
   cd archivist
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up Elasticsearch (see [Elasticsearch Setup](docs/elasticsearch_setup.md) for details).

4. Set up environment variables in a `.env` file:
   ```
   GEMINI_API_KEY=your_gemini_api_key
   ```

## Usage

1. Start the server:
   ```bash
   python -m src.app
   ```

2. The API will be available at `http://localhost:8000`

3. Use the REST API to:
   - Upload images
   - Process and analyze images
   - Search for images using natural language queries

## API Endpoints

- `POST /api/upload`: Upload images for processing
- `GET /api/images`: Get all processed images
- `POST /api/search`: Search for images
- `POST /api/set-gemini-key`: Set or update the Gemini API key
- `DELETE /api/delete/<filename>`: Delete an image

## Search Capabilities

With the Elasticsearch integration, the search functionality supports:

- **Full-text search**: Search by descriptions and patterns
- **Fuzzy matching**: Find results even with misspellings
- **Color search**: Find images with specific colors
- **Pattern matching**: Search for specific visual patterns
- **Compound queries**: Combine multiple search criteria

## Development

See [Development Guide](docs/development.md) for information on the project structure and how to contribute.

## Elasticsearch Integration

This project uses Elasticsearch as the search backend. The integration:

1. Creates an index with mappings for all image metadata
2. Automatically indexes metadata after image processing
3. Provides advanced search capabilities with fuzzy matching
4. Supports bulk operations for efficient indexing of large datasets

See [Elasticsearch Setup](docs/elasticsearch_setup.md) for detailed setup instructions. 