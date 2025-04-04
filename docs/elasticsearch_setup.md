# Elasticsearch Integration Guide

This guide explains how to set up and configure Elasticsearch for the Archivist image search system.

## Installation Options

### Option 1: Local Installation

1. Download and install Elasticsearch from the [official website](https://www.elastic.co/downloads/elasticsearch)

2. Start the Elasticsearch service:
   ```bash
   # Navigate to the Elasticsearch directory
   cd path/to/elasticsearch

   # Start Elasticsearch
   ./bin/elasticsearch
   ```

3. Verify the installation:
   ```bash
   # Check if Elasticsearch is running
   curl -X GET "localhost:9200/"
   ```

   If successful, you should see a response like:
   ```json
   {
     "name" : "your-node-name",
     "cluster_name" : "elasticsearch",
     "cluster_uuid" : "...",
     "version" : {
       "number" : "8.x.x",
       "build_flavor" : "default",
       "build_type" : "tar",
       "build_hash" : "...",
       "build_date" : "...",
       "build_snapshot" : false,
       "lucene_version" : "...",
       "minimum_wire_compatibility_version" : "...",
       "minimum_index_compatibility_version" : "..."
     },
     "tagline" : "You Know, for Search"
   }
   ```

### Option 2: Elastic Cloud (Managed Service)

1. Sign up for an account at [Elastic Cloud](https://cloud.elastic.co/)

2. Create a new deployment and note down:
   - Cloud ID
   - Username and password or API key

## Configure Archivist

### Update Configuration

1. Update the Elasticsearch configuration in `src/config/elasticsearch_config.py`:

   ```python
   # For local installation
   ELASTICSEARCH_HOSTS = ["http://localhost:9200"]
   ELASTICSEARCH_CLOUD_ID = None
   ELASTICSEARCH_API_KEY = None
   ELASTICSEARCH_USERNAME = None
   ELASTICSEARCH_PASSWORD = None
   ```

   Or for Elastic Cloud:

   ```python
   ELASTICSEARCH_HOSTS = None
   ELASTICSEARCH_CLOUD_ID = "your-deployment-cloud-id"
   ELASTICSEARCH_API_KEY = "your-api-key"  # Either set API key or username/password
   ELASTICSEARCH_USERNAME = "elastic"     # If not using API key
   ELASTICSEARCH_PASSWORD = "your-password"  # If not using API key
   ```

2. Install required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

## Initial Setup

When you start the application for the first time with Elasticsearch configured, it will:

1. Connect to your Elasticsearch instance
2. Create the 'images' index with appropriate mappings
3. Index all existing image metadata if available

## Verifying the Installation

1. Upload a test image to the application.

2. Check if the document was indexed in Elasticsearch:
   
   ```bash
   # For local installation
   curl -X GET "localhost:9200/images/_search?pretty"
   
   # For Elastic Cloud (using authentication)
   curl -u elastic:your-password -X GET "your-elasticsearch-url/images/_search?pretty"
   ```

3. The search results should include the metadata of your uploaded image.

## Performance Optimization

For production environments, consider:

1. Adjusting shards and replicas in `elasticsearch_config.py`:
   ```python
   INDEX_SHARDS = 3  # Increase for larger datasets
   INDEX_REPLICAS = 1  # For redundancy
   ```

2. Enabling Redis caching for frequent queries:
   ```python
   ENABLE_CACHE = True  # Enable Redis caching
   CACHE_TTL = 300  # Cache TTL in seconds
   ```

3. Monitoring your Elasticsearch cluster using Kibana or other monitoring tools.

## Troubleshooting

- If you cannot connect to Elasticsearch, check if the service is running and if the host/port configuration is correct.
- If searches return no results, verify that documents are being properly indexed after image processing.
- Check the application logs for any Elasticsearch-related errors.
- For authentication issues with Elastic Cloud, verify your credentials in the configuration. 