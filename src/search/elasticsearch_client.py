from elasticsearch import Elasticsearch, helpers
from elasticsearch.exceptions import NotFoundError, ConnectionError, ConnectionTimeout
import logging
from typing import Dict, List, Any, Optional, Generator
import time
import math
import numpy as np
from functools import wraps

# Configure logger
logger = logging.getLogger(__name__)

def retry_on_exception(max_retries=3, retry_interval=1.0, 
                      allowed_exceptions=(ConnectionError, ConnectionTimeout, NotFoundError)):
    """
    Decorator for retrying operations on Elasticsearch with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        retry_interval: Initial time to wait between retries (seconds)
        allowed_exceptions: Tuple of exceptions that trigger a retry
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            retry_count = 0
            current_interval = retry_interval
            last_error = None
            
            while retry_count < max_retries:
                try:
                    return func(self, *args, **kwargs)
                except allowed_exceptions as e:
                    retry_count += 1
                    last_error = e
                    
                    if retry_count < max_retries:
                        logger.warning(f"{func.__name__} failed (attempt {retry_count}): {str(e)}. "
                                     f"Retrying in {current_interval:.2f}s...")
                        time.sleep(current_interval)
                        # Exponential backoff
                        current_interval *= 1.5
                    else:
                        logger.error(f"{func.__name__} failed after {max_retries} attempts: {str(e)}")
                        raise
            
            # This should never be reached due to the raise in the except block
            return None
        return wrapper
    return decorator

class ElasticsearchClient:
    """Client for interacting with Elasticsearch"""
    
    def __init__(self, hosts: List[str] = None, cloud_id: str = None, api_key: str = None, 
                 username: str = None, password: str = None, max_retries: int = 3, 
                 retry_interval: float = 2.0):
        """
        Initialize the Elasticsearch client with connection parameters
        
        Args:
            hosts: List of Elasticsearch host URLs
            cloud_id: Cloud ID for Elastic Cloud
            api_key: API key for authentication
            username: Username for basic authentication
            password: Password for basic authentication
            max_retries: Maximum number of connection attempts
            retry_interval: Time to wait between retries in seconds
        """
        self.index_name = "images"
        
        # Set connection parameters
        self.connection_params = {}
        if hosts:
            self.connection_params["hosts"] = hosts
        if cloud_id:
            self.connection_params["cloud_id"] = cloud_id
        if api_key:
            self.connection_params["api_key"] = api_key
        elif username and password:
            self.connection_params["basic_auth"] = (username, password)
            
        # Add sensible timeouts
        self.connection_params["request_timeout"] = 30
        self.connection_params["retry_on_timeout"] = True
            
        # Initialize client with retry mechanism
        self.client = None
        retry_count = 0
        
        while retry_count < max_retries and self.client is None:
            try:
                self.client = Elasticsearch(**self.connection_params)
                if self.client.ping():
                    logger.info(f"Connected to Elasticsearch at {hosts if hosts else 'default'}")
                else:
                    logger.error("Failed to ping Elasticsearch")
                    self.client = None
            except Exception as e:
                retry_count += 1
                logger.warning(f"Connection attempt {retry_count} failed: {str(e)}")
                if retry_count < max_retries:
                    time.sleep(retry_interval)
                    retry_interval *= 1.5
                else:
                    logger.error(f"Failed to connect to Elasticsearch after {max_retries} attempts")
    
    def is_connected(self) -> bool:
        """
        Check if connected to Elasticsearch
        
        Returns:
            bool: True if connected, False otherwise
        """
        if not self.client:
            return False
        try:
            return self.client.ping()
        except Exception as e:
            logger.error(f"Elasticsearch connection error: {str(e)}")
            return False
    
    @retry_on_exception(max_retries=3, retry_interval=1.0)
    def index_exists(self) -> bool:
        """
        Check if the index exists
        
        Returns:
            bool: True if index exists, False otherwise
        """
        if not self.is_connected():
            return False
        return self.client.indices.exists(index=self.index_name)

    @retry_on_exception(max_retries=3, retry_interval=1.0)
    def create_index(self, force_recreate=False) -> bool:
        """
        Create index with appropriate mappings
        
        Args:
            force_recreate: Whether to recreate the index if it exists
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_connected():
            logger.error("Cannot create index: not connected to Elasticsearch")
            return False
            
        try:
            # Check if index exists
            index_exists = self.client.indices.exists(index=self.index_name)
            
            if index_exists:
                if force_recreate:
                    logger.info(f"Deleting existing index '{self.index_name}'")
                    self.client.indices.delete(index=self.index_name)
                else:
                    logger.info(f"Index '{self.index_name}' already exists")
                    return True
                    
            # Define custom analyzers
            settings = {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "partial_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "partial_filter"]
                        }
                    },
                    "filter": {
                        "partial_filter": {
                            "type": "edge_ngram",
                            "min_gram": 2,
                            "max_gram": 20
                        },
                        "color_synonym_filter": {
                            "type": "synonym",
                            "synonyms": [
                                "blue, navy, azure, cobalt, indigo",
                                "red, crimson, scarlet, ruby, vermilion",
                                "green, olive, emerald, lime, sage",
                                "yellow, gold, amber, mustard, ochre",
                                "orange, tangerine, rust, coral",
                                "purple, violet, lavender, lilac, mauve",
                                "pink, rose, magenta, fuchsia",
                                "brown, tan, beige, khaki, chocolate",
                                "gray, grey, silver, slate, charcoal",
                                "black, onyx, jet, ebony",
                                "white, ivory, cream, pearl"
                            ]
                        },
                        "pattern_synonym_filter": {
                            "type": "synonym",
                            "synonyms": [
                                "floral, flower, botanical, blossom, bloom",
                                "geometric, geometry, shapes, abstract",
                                "stripe, striped, linear, line",
                                "check, checked, plaid, tartan, gingham",
                                "dot, dots, polka dot, spotted, circular",
                                "paisley, teardrop, boteh, persian",
                                "animal, fauna, wildlife, creature",
                                "tropical, exotic, jungle, rainforest",
                                "damask, jacquard, brocade, ornate",
                                "abstract, non-representational, modern",
                                "ethnic, tribal, folk, cultural, indigenous",
                                "retro, vintage, nostalgic, classic",
                                "minimalist, minimal, simple, clean"
                            ]
                        }
                    }
                }
            }
            
            # Define mappings
            mappings = {
                "properties": {
                    # Basic metadata fields
                    "id": {"type": "keyword"},
                    "filename": {"type": "keyword"},
                    "path": {"type": "keyword"},
                    "thumbnail_path": {"type": "keyword"},
                    "added_date": {"type": "date"},
                    "last_modified": {"type": "date"},
                    "file_size": {"type": "long"},
                    "width": {"type": "integer"},
                    "height": {"type": "integer"},
                    "timestamp": {"type": "date"},
                    "has_fallback_analysis": {"type": "boolean"},
                    
                    # CLIP embedding vector field - removed deprecated parameters
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 512,  # CLIP-ViT-B/32 has 512 dimensions
                        "index": True  # Enable indexing for future ANN/kNN search
                    },
                    
                    # Pattern information
                    "patterns": {
                        "properties": {
                            "main_theme": {
                                "type": "text",
                                "analyzer": "english",
                                "fields": {
                                    "raw": {"type": "keyword"},
                                    "partial": {"type": "text", "analyzer": "partial_analyzer"}
                                }
                            },
                            "main_theme_confidence": {"type": "float"},
                            "category": {
                                "type": "text",
                                "analyzer": "english",
                                "fields": {
                                    "raw": {"type": "keyword"},
                                    "partial": {"type": "text", "analyzer": "partial_analyzer"}
                                }
                            },
                            "category_confidence": {"type": "float"},
                            "primary_pattern": {
                                "type": "text",
                                "analyzer": "english",
                                "fields": {
                                    "raw": {"type": "keyword"},
                                    "partial": {"type": "text", "analyzer": "partial_analyzer"}
                                }
                            },
                            "pattern_confidence": {"type": "float"},
                            "content_details": {
                                "type": "nested",
                                "properties": {
                                    "name": {
                                        "type": "text",
                                        "analyzer": "english",
                                        "fields": {
                                            "raw": {"type": "keyword"},
                                            "partial": {"type": "text", "analyzer": "partial_analyzer"}
                                        }
                                    },
                                    "confidence": {"type": "float"}
                                }
                            },
                            "stylistic_attributes": {
                                "type": "nested",
                                "properties": {
                                    "name": {
                                        "type": "text",
                                        "analyzer": "english",
                                        "fields": {
                                            "raw": {"type": "keyword"},
                                            "partial": {"type": "text", "analyzer": "partial_analyzer"}
                                        }
                                    },
                                    "confidence": {"type": "float"}
                                }
                            },
                            "secondary_patterns": {
                                "type": "nested",
                                "properties": {
                                    "name": {
                                        "type": "text",
                                        "analyzer": "english",
                                        "fields": {
                                            "raw": {"type": "keyword"},
                                            "partial": {"type": "text", "analyzer": "partial_analyzer"}
                                        }
                                    },
                                    "confidence": {"type": "float"}
                                }
                            },
                            "style_keywords": {
                                "type": "text",
                                "analyzer": "english",
                                "fields": {
                                    "raw": {"type": "keyword"}
                                }
                            },
                            "prompt": {
                                "properties": {
                                    "final_prompt": {
                                        "type": "text",
                                        "analyzer": "english",
                                        "fields": {
                                            "raw": {"type": "keyword"},
                                            "partial": {"type": "text", "analyzer": "partial_analyzer"}
                                        }
                                    }
                                }
                            },
                            "colors": {
                                "properties": {
                                    "dominant_colors": {
                                        "type": "nested",
                                        "properties": {
                                            "name": {
                                                "type": "text",
                                                "analyzer": "standard",
                                                "fields": {
                                                    "raw": {"type": "keyword"},
                                                    "partial": {"type": "text", "analyzer": "partial_analyzer"},
                                                    "synonym": {"type": "text", "analyzer": "standard", "search_analyzer": "standard", "term_vector": "with_positions_offsets"}
                                                }
                                            },
                                            "hex": {"type": "keyword"},
                                            "proportion": {"type": "float"}
                                        }
                                    },
                                    "color_distribution": {"type": "object"}
                                }
                            }
                        }
                    },
                    "colors": {
                        "properties": {
                            "dominant_colors": {
                                "type": "nested",
                                "properties": {
                                    "name": {
                                        "type": "text",
                                        "analyzer": "standard",
                                        "fields": {
                                            "raw": {"type": "keyword"},
                                            "partial": {"type": "text", "analyzer": "partial_analyzer"},
                                            "synonym": {"type": "text", "analyzer": "standard", "search_analyzer": "standard", "term_vector": "with_positions_offsets"}
                                        }
                                    },
                                    "hex": {"type": "keyword"},
                                    "proportion": {"type": "float"}
                                }
                            },
                            "color_distribution": {"type": "object"}
                        }
                    }
                }
            }
            
            # Create the index with both settings and mappings
            self.client.indices.create(
                index=self.index_name,
                body={
                    "settings": settings,
                    "mappings": mappings
                }
            )
            logger.info(f"Created index '{self.index_name}' with enhanced mappings and analyzers")
            return True
        except Exception as e:
            logger.error(f"Failed to create index: {str(e)}")
            return False 

    @retry_on_exception(max_retries=3, retry_interval=1.0)
    def index_document(self, document: Dict[str, Any]) -> bool:
        """
        Index a single document into Elasticsearch
        
        Args:
            document: The document to index
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_connected():
            logger.error("Cannot index document: not connected to Elasticsearch")
            return False
            
        # Make sure the index exists
        if not self.index_exists():
            logger.info(f"Index '{self.index_name}' doesn't exist, creating it")
            if not self.create_index():
                logger.error(f"Failed to create index '{self.index_name}'")
                return False
        
        # Extract the document ID
        doc_id = document.get("id") or document.get("path") or document.get("filename")
        if not doc_id:
            logger.error("Document must have an 'id', 'path', or 'filename' field")
            return False
        
        # Create a copy of the document to avoid modifying the original
        document_copy = document.copy()
        
        # Validate the embedding if present
        embedding_status = "missing"
        if "embedding" in document_copy:
            embedding = document_copy["embedding"]
            if embedding is not None:
                embedding_status = "present"
                
                # Check if it's a list and has the right dimensions
                if isinstance(embedding, list):
                    embedding_length = len(embedding)
                    
                    if embedding_length != 512:
                        logger.warning(f"Document {doc_id} has embedding with incorrect dimensions: {embedding_length} (expected 512)")
                        
                        # Ensure proper encoding for ES - normalize length
                        if embedding_length > 512:
                            document_copy["embedding"] = embedding[:512]
                            logger.info(f"Truncated embedding from {embedding_length} to 512 dimensions for document {doc_id}")
                            embedding_status = "truncated"
                        elif embedding_length < 512:
                            logger.warning(f"Document {doc_id} has embedding with incorrect dimensions: {embedding_length} (expected 512). Removing embedding before indexing.")
                            document_copy.pop("embedding")
                            embedding_status = "removed_invalid_dims"
                    else:
                        logger.info(f"Document {doc_id} has valid embedding with {embedding_length} dimensions")
                        
                # If it's an ndarray, convert to list
                elif hasattr(embedding, 'tolist'):
                    try:
                        document_copy["embedding"] = embedding.tolist()
                        embedding_length = len(document_copy["embedding"])
                        logger.info(f"Converted ndarray embedding to list ({embedding_length} dimensions) for document {doc_id}")
                        embedding_status = "converted"
                    except Exception as e:
                        logger.error(f"Failed to convert ndarray embedding to list for document {doc_id}: {str(e)}")
                        document_copy.pop("embedding")
                        embedding_status = "removed_invalid"
                else:
                    # Remove invalid embeddings to avoid indexing issues
                    logger.warning(f"Document {doc_id} has invalid embedding type: {type(embedding)}, removing it before indexing")
                    document_copy.pop("embedding")
                    embedding_status = "removed_invalid"
            else:
                # Remove null embeddings to avoid indexing issues
                logger.warning(f"Document {doc_id} has null embedding, removing it before indexing")
                document_copy.pop("embedding")
                embedding_status = "removed_null"
        
        # Index the document
        try:
            self.client.index(index=self.index_name, id=doc_id, document=document_copy)
            logger.info(f"Indexed document with ID: {doc_id} (embedding: {embedding_status})")
            return True
        except Exception as e:
            logger.error(f"Failed to index document {doc_id}: {str(e)}")
            # Add more detailed error information
            if "Content-Type header" in str(e):
                logger.error("Content-Type header issue - check Elasticsearch client version compatibility")
            elif "ConnectionTimeout" in str(e):
                logger.error("Connection timeout - check Elasticsearch server availability and network")
            elif "RequestError" in str(e) and "mapper_parsing_exception" in str(e):
                logger.error("Mapping error - document structure may not match index mapping")
            elif "document_already_exists_exception" in str(e):
                logger.warning(f"Document {doc_id} already exists - consider using update operation instead")
            return False
    
    @retry_on_exception(max_retries=3, retry_interval=1.0)
    def bulk_index(self, documents: List[Dict[str, Any]], chunk_size: int = 500) -> bool:
        """
        Bulk index multiple documents into Elasticsearch
        
        Args:
            documents: List of documents to index
            chunk_size: Number of documents per bulk request
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_connected():
            logger.error("Cannot bulk index: not connected to Elasticsearch")
            return False
            
        if not documents:
            logger.warning("No documents to bulk index")
            return True
            
        # Make sure the index exists
        if not self.index_exists():
            logger.info(f"Index '{self.index_name}' doesn't exist, creating it")
            if not self.create_index():
                logger.error(f"Failed to create index '{self.index_name}'")
                return False
            
        # Process documents in chunks
        actions = []
        embedding_stats = {
            "present": 0,
            "missing": 0,
            "truncated": 0,
            "removed_invalid_dims": 0,
            "converted": 0,
            "removed_null": 0,
            "removed_invalid": 0
        }
        
        for doc in documents:
            # Extract ID
            doc_id = doc.get("id") or doc.get("path") or doc.get("filename")
            if not doc_id:
                logger.warning("Skipping document without ID field")
                continue
            
            # Create a copy to avoid modifying the original
            processed_doc = doc.copy()
            
            # Validate the embedding if present
            embedding_status = "missing"
            if "embedding" in processed_doc:
                embedding = processed_doc["embedding"]
                if embedding is not None:
                    embedding_status = "present"
                    
                    # Check if it's a list and has the right dimensions
                    if isinstance(embedding, list):
                        embedding_length = len(embedding)
                        
                        if embedding_length != 512:
                            logger.debug(f"Document {doc_id} has embedding with incorrect dimensions: {embedding_length} (expected 512)")
                            
                            # Ensure proper encoding for ES - normalize length
                            if embedding_length > 512:
                                processed_doc["embedding"] = embedding[:512]
                                logger.debug(f"Truncated embedding to 512 dimensions for document {doc_id}")
                                embedding_status = "truncated"
                            elif embedding_length < 512:
                                logger.warning(f"Document {doc_id} has embedding with incorrect dimensions: {embedding_length} (expected 512). Removing embedding before indexing.")
                                processed_doc.pop("embedding")
                                embedding_status = "removed_invalid_dims"
                        
                    # If it's an ndarray, convert to list
                    elif hasattr(embedding, 'tolist'):
                        try:
                            processed_doc["embedding"] = embedding.tolist()
                            embedding_status = "converted"
                        except Exception as e:
                            logger.error(f"Failed to convert ndarray embedding to list for document {doc_id}: {str(e)}")
                            processed_doc.pop("embedding")
                            embedding_status = "removed_invalid"
                    else:
                        # Remove invalid embeddings to avoid indexing issues
                        logger.warning(f"Document {doc_id} has invalid embedding type: {type(embedding)}, removing it before indexing")
                        processed_doc.pop("embedding")
                        embedding_status = "removed_invalid"
                else:
                    # Remove null embeddings to avoid indexing issues
                    processed_doc.pop("embedding")
                    embedding_status = "removed_null"
                    
            # Update embedding stats
            embedding_stats[embedding_status] += 1
                
            action = {
                "_index": self.index_name,
                "_id": doc_id,
                "_source": processed_doc
            }
            actions.append(action)
            
        if not actions:
            logger.warning("No valid documents to bulk index")
            return False
            
        # Execute bulk indexing
        try:
            success, failed = helpers.bulk(
                self.client, 
                actions, 
                chunk_size=chunk_size,
                raise_on_error=False
            )
            
            # Log embedding statistics
            logger.info(f"Embedding statistics for bulk indexing: {embedding_stats}")
            
            if failed:
                logger.warning(f"Bulk indexed {success} documents, {len(failed)} failed:")
                for failure in failed[:5]:  # Log first 5 failures
                    logger.warning(f"  - {failure}")
            else:
                logger.info(f"Bulk indexed {success} documents successfully")
                
            return success > 0
        except Exception as e:
            logger.error(f"Bulk indexing failed: {str(e)}")
            # Add more detailed error information
            if "Content-Type header" in str(e):
                logger.error("Content-Type header issue - check Elasticsearch client version compatibility")
            elif "ConnectionTimeout" in str(e):
                logger.error("Connection timeout - check Elasticsearch server availability and network")
            elif "RequestError" in str(e) and "mapper_parsing_exception" in str(e):
                logger.error("Mapping error - document structure may not match index mapping")
            return False
        
    @retry_on_exception(max_retries=3, retry_interval=1.0)
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from Elasticsearch by ID
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_connected():
            logger.error("Cannot delete document: not connected to Elasticsearch")
            return False
        
        try:
            self.client.delete(index=self.index_name, id=doc_id)
            logger.info(f"Document with ID {doc_id} deleted successfully")
            return True
        except Exception as e:
            # Document not found is not an error in this context
            if "404" in str(e) or "not_found" in str(e).lower():
                logger.warning(f"Document with ID {doc_id} not found for deletion")
                return False
            # Other errors
            logger.error(f"Error deleting document with ID {doc_id}: {str(e)}")
            return False 

    @retry_on_exception(max_retries=3, retry_interval=1.0)
    def search(self, query: str, limit: int = 20, min_similarity: float = 0.1) -> List[Dict[str, Any]]:
        """
        Basic text search that matches text across various metadata fields.
        Uses a simple query string approach that works with all field types.
        """
        if not self.is_connected():
            logger.error("Cannot search: not connected to Elasticsearch")
            return []

        # Clean the query
        clean_query = query.strip()
        if not clean_query:
            logger.warning("Received empty search query.")
            return []

        logger.info(f"Performing basic text search for: '{clean_query}', limit={limit}")

        # Use a simple query_string query which works with both text and keyword fields
        query_body = {
            "size": limit,
            "query": {
                "query_string": {
                    "query": clean_query,
                    "default_operator": "AND",
                    "fields": [
                        "patterns.primary_pattern^3",
                        "patterns.main_theme^2.5",
                        "patterns.style_keywords^2", 
                        "patterns.secondary_patterns.name^1.5",
                        "patterns.content_details.name^1.5",
                        "colors.dominant_colors.name^1",
                        "patterns.prompt.final_prompt^1",
                        "filename^0.5"
                    ]
                }
            },
            "sort": [
                "_score",
                {"timestamp": {"order": "desc"}}
            ]
        }

        # Execute the search and process results
        return self._execute_search(query_body, limit, min_similarity)

    @retry_on_exception(max_retries=3, retry_interval=1.0)
    def find_similar(self, embedding=None, limit=20, min_similarity=0.1, exclude_id=None, 
                   text_query=None, image_weight=0.7, text_weight=0.3):
        """
        Find similar images using vector embedding, text search, or both.
        Simplified approach that prioritizes getting results over perfect scoring.
        
        Args:
            embedding: Vector embedding to use for similarity search
            limit: Maximum results to return
            min_similarity: Minimum similarity threshold (mostly ignored in basic search)
            exclude_id: Optional ID to exclude from results
            text_query: Optional text query to combine with vector search
            image_weight: Weight for vector similarity in hybrid search (0-1)
            text_weight: Weight for text similarity in hybrid search (0-1)
            
        Returns:
            List of similar documents
        """
        # Check for valid inputs
        has_embedding = embedding is not None
        has_text = text_query is not None and text_query.strip() != ""
        
        if not has_embedding and not has_text:
            logger.error("find_similar requires either embedding or text_query")
            return []
            
        if not self.is_connected():
            logger.error("Cannot find similar: not connected to Elasticsearch")
            return []
        
        # If we have both, do a hybrid search
        if has_embedding and has_text:
            # Prepare embedding if needed
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()
                
            logger.info(f"Hybrid search: text='{text_query}', embedding dimensions={len(embedding) if isinstance(embedding, list) else 'unknown'}")
            
            # Get text search results
            text_results = self.search(text_query, limit=limit*2, min_similarity=0)
            
            # Get vector search results
            vector_results = self._vector_search(embedding, limit=limit*2, min_similarity=0, exclude_id=exclude_id)
            
            # Simple approach - merge both result sets with preference for higher scores
            all_results = {}
            
            # Process text results
            for result in text_results:
                doc_id = result.get('id')
                if doc_id:
                    all_results[doc_id] = result
            
            # Process vector results - augment existing or add new
            for result in vector_results:
                doc_id = result.get('id')
                if not doc_id or (exclude_id and doc_id == exclude_id):
                    continue
                
                if doc_id in all_results:
                    # If already exists from text search, take the higher score
                    existing = all_results[doc_id]
                    if result.get('raw_score', 0) > existing.get('raw_score', 0):
                        all_results[doc_id] = result
                else:
                    all_results[doc_id] = result
            
            # Convert to list and sort by score
            combined_results = list(all_results.values())
            combined_results.sort(key=lambda x: x.get('raw_score', 0), reverse=True)
            
            # Limit to requested number
            combined_results = combined_results[:limit]
            
            logger.info(f"Hybrid search returned {len(combined_results)} results")
            return combined_results
            
        elif has_embedding:
            # Vector similarity search only
            return self._vector_search(embedding, limit, min_similarity, exclude_id)
        elif has_text:
            # Text similarity search only
            return self.search(text_query, limit, min_similarity)
        else:
            # This shouldn't happen due to the checks above
            return []

    def _execute_search(self, query, limit, min_similarity):
        """
        Execute a search query and process results.
        Uses lenient scoring to ensure we get results.
        
        Args:
            query: Elasticsearch query object
            limit: Maximum number of results to return
            min_similarity: Minimum similarity threshold (mostly ignored for basic search)
            
        Returns:
            List of documents with normalized similarity scores
        """
        try:
            # Execute the search
            start_time = time.time()
            response = self.client.search(
                index=self.index_name,
                body=query
            )
            elapsed = time.time() - start_time
            logger.info(f"Search completed in {elapsed:.2f}s")
            
            # Extract hits
            hits = response.get('hits', {}).get('hits', [])
            
            # Basic processing - log info about results
            logger.info(f"Search returned {len(hits)} raw hits from Elasticsearch")
            
            # For basic search, we'll be more lenient with scoring
            # to ensure we get results even with partial matches
            results = []
            
            if hits:
                # Get score range for reference only
                scores = [hit.get('_score', 0) for hit in hits]
                min_score = min(scores) if scores else 0
                max_score = max(scores) if scores else 0
                logger.info(f"Score range: min={min_score:.4f}, max={max_score:.4f}")
                
                # Process all hits - we'll be lenient and include everything 
                for hit in hits:
                    doc = hit.get('_source', {})
                    raw_score = hit.get('_score', 0)
                    
                    # Scale score to percentage - even low scores get included
                    # This is basic search, we just want to show anything that matches
                    similarity = 100.0
                    if max_score > 0:
                        similarity = (raw_score / max_score) * 100.0
                    
                    # Add scores to document
                    doc["similarity"] = round(similarity, 4)
                    doc["raw_score"] = raw_score
                    results.append(doc)
                
                # Sort by score (highest first)
                results.sort(key=lambda x: x.get("raw_score", 0), reverse=True)
            
            logger.info(f"Returning {len(results)} results after processing")
            return results
            
        except Exception as e:
            logger.error(f"Search execution error: {str(e)}")
            return []

    def _vector_search(self, embedding, limit, min_similarity, exclude_id=None):
        """
        Perform basic vector similarity search using CLIP embedding.
        Simplified to ensure results are always returned.
        """
        # Debug the embedding
        embedding_size = len(embedding) if isinstance(embedding, list) else (embedding.size if hasattr(embedding, 'size') else 'unknown')
        logger.info(f"Performing vector search with embedding of size {embedding_size}")
        
        # Convert numpy array to list if needed
        if hasattr(embedding, 'tolist'):
            embedding = embedding.tolist()
        
        # Create a very simple script score query with a low threshold
        query = {
            "size": limit * 2,  # Get more results than needed to ensure we have enough
            "query": {
                "script_score": {
                    "query": {"match_all": {}},  # Match all documents with embeddings
                    "script": {
                        "source": """
                            if (!doc.containsKey('embedding') || doc['embedding'].size() == 0) { return 0.0; }
                            try {
                                // Simple cosine similarity
                                return (dotProduct(params.query_vector, 'embedding') + 1.0) / 2.0;
                            } catch (Exception e) {
                                return 0.0;
                            }
                        """,
                        "params": {
                            "query_vector": embedding
                        }
                    }
                }
            }
        }
        
        # Add exclusion if provided
        if exclude_id:
            query["query"] = {
                "bool": {
                    "must": [query["query"]],
                    "must_not": [
                        {"term": {"id": exclude_id}},
                        {"term": {"filename": exclude_id}},
                        {"term": {"path": exclude_id}}
                    ]
                }
            }
        
        try:
            # Execute the search directly
            response = self.client.search(
                index=self.index_name,
                body=query
            )
            
            # Process results
            hits = response.get('hits', {}).get('hits', [])
            logger.info(f"Vector search returned {len(hits)} raw hits")
            
            # Very lenient processing - include everything
            results = []
            for hit in hits:
                doc = hit.get('_source', {})
                raw_score = hit.get('_score', 0)
                
                # Keep all scores, even very low ones
                doc["similarity"] = round(raw_score * 100, 2)  # Simple scaling to percentage
                doc["raw_score"] = raw_score
                doc["vector_score"] = round(raw_score * 100, 2)  # Add explicit vector score for reference
                
                # Add to results regardless of score
                results.append(doc)
            
            # Sort by score (highest first)
            results.sort(key=lambda x: x.get("raw_score", 0), reverse=True)
            
            # Limit to requested number
            results = results[:limit]
            
            logger.info(f"Returning {len(results)} vector search results after processing")
            return results
            
        except Exception as e:
            logger.error(f"Vector search error: {str(e)}")
            return []

    @retry_on_exception(max_retries=3, retry_interval=1.0)
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document from the index by ID
        
        Args:
            doc_id: Document ID to retrieve
            
        Returns:
            Document data as dict, or None if not found
        """
        if not self.is_connected():
            logger.error("Cannot get document: not connected to Elasticsearch")
            return None
            
        try:
            # Try to get document by ID
            response = self.client.get(index=self.index_name, id=doc_id)
            if response and response.get('found', False):
                # Return the document source
                return response.get('_source', {})
            else:
                logger.warning(f"Document with ID '{doc_id}' not found")
                return None
        except NotFoundError:
            logger.warning(f"Document with ID '{doc_id}' not found")
            return None
        except Exception as e:
            logger.error(f"Error getting document with ID '{doc_id}': {str(e)}")
            return None 

    def search_by_vector(self, embedding, limit=20, min_similarity=0.1, text_weight=0.3, vector_weight=0.7):
        """
        Search by CLIP embedding vector - simplified version
        
        Args:
            embedding: The vector embedding to search by
            limit: Maximum number of results to return
            min_similarity: Minimum similarity threshold (not strictly enforced in basic search)
            text_weight: Not used in this implementation
            vector_weight: Not used in this implementation
            
        Returns:
            List of similar documents sorted by similarity
        """
        if embedding is None:
            logger.error("Cannot search by vector: embedding is None")
            return []
            
        if not self.is_connected():
            logger.error("Cannot search by vector: not connected to Elasticsearch")
            return []
            
        # Direct call to _vector_search with the specified parameters
        # Use a very low min_similarity to ensure we get results
        logger.info(f"Vector search with {limit} limit (min_similarity={min_similarity})")
        actual_min_similarity = 0.0001  # Use an extremely low threshold to ensure results
        results = self._vector_search(embedding, limit, actual_min_similarity)
        
        # Log search outcomes
        if results:
            logger.info(f"search_by_vector returned {len(results)} results")
            # Log first few scores for debugging
            if results and len(results) > 0:
                scores = [f"{r.get('similarity', 0):.2f}%" for r in results[:5]]
                logger.info(f"Top scores: {', '.join(scores)}")
        else:
            logger.warning(f"search_by_vector returned no results")
            
        return results

    def _hybrid_search(self, embedding, text_query, limit=20, min_similarity=0.1, exclude_id=None,
                     text_weight=0.3, vector_weight=0.7):
        """
        Hybrid search combining vector similarity with text search
        
        This performs both vector search and text search separately, then combines results
        with weighted scoring based on the text_weight and vector_weight parameters.
        
        Args:
            embedding: Vector embedding for visual similarity
            text_query: Text query for semantic search
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold
            exclude_id: Optional ID to exclude from results
            text_weight: Weight of text results in final score (0-1)
            vector_weight: Weight of vector results in final score (0-1)
            
        Returns:
            Combined and re-scored results
        """
        # Normalize weights to sum to 1.0
        total_weight = text_weight + vector_weight
        if total_weight <= 0:
            logger.warning("Invalid weights, defaulting to 50/50 split")
            text_weight = 0.5
            vector_weight = 0.5
        else:
            text_weight = text_weight / total_weight
            vector_weight = vector_weight / total_weight
            
        logger.info(f"Hybrid search with weights: text={text_weight:.2f}, vector={vector_weight:.2f}")
        
        # Track timing for performance analysis
        start_time = time.time()
        
        # Step 1: Perform text search with higher limit to ensure enough candidates
        text_limit = min(limit * 3, 100)  # Get more candidates for rescoring
        text_results = self.search(text_query, limit=text_limit, min_similarity=0)
        
        # Step 2: Perform vector search with higher limit
        vector_limit = min(limit * 3, 100)
        vector_results = self.search_by_vector(embedding, limit=vector_limit, min_similarity=0)
        
        # Log individual search results
        logger.info(f"Text search returned {len(text_results)} results")
        logger.info(f"Vector search returned {len(vector_results)} results")
        
        # Early exit if both searches returned nothing
        if not text_results and not vector_results:
            logger.warning("No results from either text or vector search")
            return []
            
        # Step 3: Create ID-indexed maps for fast lookup
        text_map = {item.get('id', ''): item for item in text_results if item.get('id')}
        vector_map = {item.get('id', ''): item for item in vector_results if item.get('id')}
        
        # Step 4: Get the union of all IDs from both result sets
        all_ids = set(text_map.keys()) | set(vector_map.keys())
        if exclude_id and exclude_id in all_ids:
            all_ids.remove(exclude_id)
            
        # Step 5: Combine and re-score results
        combined_results = []
        
        for doc_id in all_ids:
            # Get individual results and scores
            text_item = text_map.get(doc_id)
            vector_item = vector_map.get(doc_id)
            
            # Get scores (default to 0 if the item wasn't in that result set)
            text_score = text_item.get('similarity', 0) if text_item else 0
            vector_score = vector_item.get('similarity', 0) if vector_item else 0
            
            # Calculate combined score with weights
            combined_score = (text_score * text_weight) + (vector_score * vector_weight)
            
            # Use document from either source (prefer vector for completeness)
            base_doc = vector_item if vector_item else text_item
            
            # Create combined result with all scores
            result = {
                **base_doc,
                'similarity': combined_score,
                'text_similarity': text_score,
                'vector_similarity': vector_score,
                'search_info': {
                    'text_weight': text_weight,
                    'vector_weight': vector_weight
                }
            }
            
            # Only keep results above threshold
            if combined_score >= min_similarity * 100:
                combined_results.append(result)
                
        # Sort by combined score
        combined_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        
        # Limit to requested number
        combined_results = combined_results[:limit]
        
        # Report timing
        elapsed = time.time() - start_time
        logger.info(f"Hybrid search completed in {elapsed:.2f}s, returned {len(combined_results)} results")
        
        return combined_results 