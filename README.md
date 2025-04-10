# Archivist

A Python-based image processing and search system with AI-powered image analysis and advanced search capabilities.

## Features

- **Advanced Image Processing**: Analyze images for patterns, colors, and visual elements using computer vision techniques.
- **AI-Powered Analysis**: Integration with Google's Gemini API for comprehensive image understanding and description.
- **Semantic Search**: Find images using natural language queries or by image similarity.
- **Vector Search**: Use CLIP embeddings for searching visually similar images.
- **Elasticsearch Integration**: Scalable search backend for efficient image retrieval with advanced querying.
- **Thumbnails Generation**: Automatically create and manage thumbnail images for faster browsing.
- **Caching Layer**: Redis-based caching for improved performance and reduced API calls.
- **RESTful API**: Comprehensive API for integrating with any frontend application.
- **Web Interface**: Simple UI for search and image management.

## Architecture

The project follows a modular architecture:

- **Core Engine**: Handles image processing, metadata extraction, and search functionality.
- **AI Integration**: Connects with Google's Gemini API for advanced image analysis.
- **Search Backend**: Uses both vector embeddings (CLIP) and Elasticsearch for efficient and scalable search capabilities.
- **Caching Layer**: Uses Redis for caching to speed up repeated searches and reduce API calls.
- **RESTful API**: Flask-based API for interacting with the system.
- **UI**: Minimal web interface for searching and viewing results.

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
   ```bash
   ./start_elasticsearch_bg.sh  # Starts Elasticsearch in the background
   ```

4. Set up environment variables in a `.env` file:
   ```
   GEMINI_API_KEY=your_gemini_api_key
   ELASTICSEARCH_HOSTS=http://localhost:9200
   ELASTICSEARCH_INDEX=archivist
   DEBUG=true
   ```

## Usage

1. Start the server:
   ```bash
   python main.py
   ```

2. The API will be available at `http://localhost:8000`

3. Use the REST API to:
   - Upload images for processing
   - Search for images using text or image-based queries
   - Manage image metadata
   - Get image recommendations

## API Endpoints

- `POST /api/upload`: Upload images for processing
- `GET /api/images`: Get all processed images
- `POST /api/search`: Search for images using text
- `POST /api/similarity`: Find similar images
- `POST /api/set-gemini-key`: Set or update the Gemini API key
- `DELETE /api/delete/<filename>`: Delete an image
- `PUT /api/metadata/<filename>`: Update image metadata
- `GET /api/analytics`: Get search analytics

## Search Capabilities

The search functionality supports:

- **Semantic search**: Use natural language to find images
- **Similarity search**: Find images similar to a reference image
- **Color search**: Find images with specific colors
- **Pattern matching**: Search for specific visual patterns
- **Hybrid search**: Combine text and image similarity for better results
- **Filters**: Apply filters based on metadata attributes

## Utilities and Scripts

The project includes various utility scripts:

- `scripts/reindex_embeddings.py`: Reindex all image embeddings
- `scripts/cleanup_metadata.py`: Clean up orphaned metadata entries
- `scripts/fix_index.py`: Fix Elasticsearch index issues
- `scripts/test_search.py`: Test search functionality
- `scripts/es_health_check.py`: Check Elasticsearch cluster health

## Development

See [Development Guide](docs/development.md) for information on the project structure and how to contribute.

## Elasticsearch Integration

This project uses Elasticsearch as the search backend. The integration:

1. Creates an index with mappings for all image metadata
2. Automatically indexes metadata after image processing
3. Provides advanced search capabilities with fuzzy matching and boosting
4. Supports bulk operations for efficient indexing of large datasets

See [Elasticsearch Setup](docs/elasticsearch_setup.md) for detailed setup instructions. 