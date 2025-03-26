"""Elasticsearch configuration settings"""

# Elasticsearch connection
ELASTICSEARCH_HOSTS = ["http://localhost:9200"]  # Default local Elasticsearch instance

# Set to None if using default local instance without authentication
ELASTICSEARCH_CLOUD_ID = None
ELASTICSEARCH_API_KEY = None
ELASTICSEARCH_USERNAME = None
ELASTICSEARCH_PASSWORD = None

# Index settings
INDEX_NAME = "images"
INDEX_SHARDS = 1
INDEX_REPLICAS = 0

# Search settings
DEFAULT_SEARCH_LIMIT = 20
DEFAULT_MIN_SIMILARITY = 0.1

# Caching settings (if Redis is used)
ENABLE_CACHE = False
CACHE_TTL = 300  # Time to live for cached search results (in seconds) 