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
                        "dims": 512  # CLIP-ViT-B/32 has 512 dimensions
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
                            # Pad with zeros if too short (not ideal but better than failing)
                            padding = [0.0] * (512 - embedding_length)
                            document_copy["embedding"] = embedding + padding
                            logger.warning(f"Padded embedding from {embedding_length} to 512 dimensions for document {doc_id} - this may affect search quality")
                            embedding_status = "padded"
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
            "padded": 0,
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
                                # Pad with zeros if too short (not ideal but better than failing)
                                padding = [0.0] * (512 - embedding_length)
                                processed_doc["embedding"] = embedding + padding
                                logger.debug(f"Padded embedding to 512 dimensions for document {doc_id}")
                                embedding_status = "padded"
                        
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
        Enhanced search function using composite query structure
        
        Args:
            query: The search query string
            limit: Maximum number of results to return
            min_similarity: Minimum similarity score threshold
            
        Returns:
            List of matching documents
        """
        if not self.is_connected():
            logger.error("Cannot search: not connected to Elasticsearch")
            return []
            
        # For empty or wildcard queries
        if query == "*" or not query.strip():
            query_body = {"match_all": {}}
            return self._execute_search(query_body, limit, min_similarity)
        
        # Log search parameters for debugging
        logger.info(f"Performing enhanced search for: '{query}' with limit {limit}")
        
        # Split query into terms for more flexibility
        query_terms = [term.strip() for term in query.split() if term.strip()]
        min_should_match = min(len(query_terms), 2) if len(query_terms) > 1 else 1
        
        # Build the main compound query
        should_clauses = []
        
        # 1. Exact matches on main_theme with highest boost
        should_clauses.append({
            "match": {
                "patterns.main_theme.raw": {
                    "query": query,
                    "boost": 6.0
                }
            }
        })
        
        # 2. Exact matches on primary_pattern with high boost
        should_clauses.append({
            "match": {
                "patterns.primary_pattern.raw": {
                    "query": query,
                    "boost": 5.0
                }
            }
        })
        
        # 3. Nested query for content details with enhanced boosting
        should_clauses.append({
            "nested": {
                "path": "patterns.content_details",
                "query": {
                    "bool": {
                        "should": [
                            {
                                "match": {
                                    "patterns.content_details.name.raw": {
                                        "query": query,
                                        "boost": 4.5
                                    }
                                }
                            },
                            {
                                "match": {
                                    "patterns.content_details.name": {
                                        "query": query,
                                        "boost": 4.0
                                    }
                                }
                            },
                            {
                                "match": {
                                    "patterns.content_details.name.partial": {
                                        "query": query,
                                        "boost": 3.5
                                    }
                                }
                            }
                        ]
                    }
                },
                "score_mode": "max",
                "boost": 4.0
            }
        })
        
        # 4. Multi-match across all relevant text fields
        should_clauses.append({
            "multi_match": {
                "query": query,
                "fields": [
                    "patterns.main_theme^4.0",
                    "patterns.main_theme.partial^3.5",
                    "patterns.primary_pattern^3.0",
                    "patterns.primary_pattern.partial^2.5",
                    "patterns.prompt.final_prompt^2.0",
                    "patterns.prompt.final_prompt.partial^1.5",
                    "patterns.style_keywords^1.5"
                ],
                "type": "best_fields",
                "fuzziness": "AUTO",
                "prefix_length": 2,
                "boost": 3.0
            }
        })
        
        # 5. Nested query for stylistic attributes
        should_clauses.append({
            "nested": {
                "path": "patterns.stylistic_attributes",
                "query": {
                    "bool": {
                        "should": [
                            {
                                "match": {
                                    "patterns.stylistic_attributes.name.raw": {
                                        "query": query,
                                        "boost": 2.0
                                    }
                                }
                            },
                            {
                                "match": {
                                    "patterns.stylistic_attributes.name": {
                                        "query": query,
                                        "boost": 1.8
                                    }
                                }
                            },
                            {
                                "match": {
                                    "patterns.stylistic_attributes.name.partial": {
                                        "query": query,
                                        "boost": 1.5
                                    }
                                }
                            }
                        ]
                    }
                },
                "score_mode": "max",
                "boost": 2.0
            }
        })
        
        # 6. Nested query for secondary patterns
        should_clauses.append({
            "nested": {
                "path": "patterns.secondary_patterns",
                "query": {
                    "bool": {
                        "should": [
                            {
                                "match": {
                                    "patterns.secondary_patterns.name.raw": {
                                        "query": query,
                                        "boost": 2.0
                                    }
                                }
                            },
                            {
                                "match": {
                                    "patterns.secondary_patterns.name": {
                                        "query": query,
                                        "boost": 1.8
                                    }
                                }
                            }
                        ]
                    }
                },
                "score_mode": "max",
                "boost": 1.5
            }
        })
        
        # 7. Nested query for colors
        should_clauses.append({
            "nested": {
                "path": "colors.dominant_colors",
                "query": {
                    "bool": {
                        "should": [
                            {
                                "match": {
                                    "colors.dominant_colors.name.raw": {
                                        "query": query,
                                        "boost": 1.5
                                    }
                                }
                            },
                            {
                                "match": {
                                    "colors.dominant_colors.name": {
                                        "query": query,
                                        "boost": 1.2
                                    }
                                }
                            }
                        ]
                    }
                },
                "score_mode": "avg",
                "boost": 1.0
            }
        })
        
        # Build the main bool query
        bool_query = {
            "bool": {
                "should": should_clauses,
                "minimum_should_match": 1
            }
        }
        
        # Wrap in a function_score query to factor in confidence values
        function_score_query = {
            "function_score": {
                "query": bool_query,
                "functions": [
                    {
                        "field_value_factor": {
                            "field": "patterns.main_theme_confidence",
                            "factor": 2.0,
                            "missing": 0.8,
                            "modifier": "log1p"
                        }
                    },
                    {
                        "field_value_factor": {
                            "field": "patterns.pattern_confidence",
                            "factor": 1.5,
                            "missing": 0.7,
                            "modifier": "log1p"
                        }
                    },
                    # Reduce score for fallback analysis results
                    {
                        "script_score": {
                            "script": {
                                "source": "doc.containsKey('has_fallback_analysis') && doc['has_fallback_analysis'].value ? 0.75 : 1.0"
                            }
                        }
                    },
                    # Add recency boost
                    {
                        "gauss": {
                            "timestamp": {
                                "scale": "30d",
                                "decay": 0.5
                            }
                        },
                        "weight": 0.5
                    }
                ],
                "score_mode": "multiply",
                "boost_mode": "multiply"
            }
        }
        
        # Execute search with the function_score query
        return self._execute_search(function_score_query, limit, min_similarity)
    
    def _execute_search(self, query_body, limit, min_similarity):
        """
        Helper method to execute a search with the given query body and process results
        
        Args:
            query_body: The Elasticsearch query to execute
            limit: Maximum number of results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of documents sorted by similarity
        """
        start_time = time.time()
        try:
            # Log full query for debugging
            query_debug = str(query_body)
            if len(query_debug) > 1000:
                query_debug = query_debug[:500] + "..." + query_debug[-500:]
            logger.debug(f"Executing search with query: {query_debug}")
            
            # Execute the search
            response = self.client.search(
                index=self.index_name,
                body={
                    "size": limit,
                    "query": query_body,
                    "_source": True,
                    "track_scores": True
                }
            )
            
            search_time = time.time() - start_time
            logger.info(f"Search completed in {search_time:.2f}s")
            
            # Log hit count
            hit_count = len(response["hits"]["hits"])
            logger.info(f"Search returned {hit_count} raw hits from Elasticsearch")
            
            # If no results, return empty list
            if hit_count == 0:
                logger.info("No search results found")
                return []
            
            # Get max score for normalization
            max_score = response["hits"]["max_score"] if response["hits"]["hits"] else 1.0
            min_score = min([hit["_score"] for hit in response["hits"]["hits"]]) if response["hits"]["hits"] else 0.0
            score_range = max(max_score - min_score, 0.001)  # Avoid division by zero
            
            logger.info(f"Search score range: min={min_score:.4f}, max={max_score:.4f}, range={score_range:.4f}")
            
            # Process and normalize results
            results = []
            for hit in response["hits"]["hits"]:
                doc = hit["_source"]
                raw_score = hit["_score"]
                
                # Determine if this is a vector search (script_score with embedding)
                is_vector_search = "script_score" in str(query_body) and "embedding" in str(query_body)
                
                # Different normalization strategies based on search type
                if is_vector_search:
                    # For vector similarity (CLIP embeddings)
                    # The script already returns a score in [0,2] range where:
                    # - 0 means completely different
                    # - 1 means neutral/random similarity
                    # - 2 means identical
                    
                    # Normalize to [0,1] range where:
                    # - 0 means random similarity or worse
                    # - 1 means identical
                    vector_similarity = max(0, (raw_score - 1.0)) if raw_score > 1.0 else 0.0
                    
                    # Apply exponential scaling to emphasize high similarities
                    # This makes scores more useful visually (fewer items with near-identical scores)
                    similarity = min(1.0, vector_similarity ** 0.75 * 1.2)  
                    
                elif "function_score" in str(query_body) and "script_score" in str(query_body):
                    # For hybrid similarity (text + vector)
                    # The scores will typically be higher, so normalize differently
                    
                    # If score range is significant, use dynamic scaling
                    if score_range > 0.1:  
                        # Enhanced normalization with curve
                        normalized_score = (raw_score - min_score) / score_range
                        # Apply curve to increase contrast between results
                        similarity = min(1.0, normalized_score ** 0.8)
                    else:
                        # Simple min-max normalization if range is small
                        similarity = (raw_score - min_score) / score_range if score_range > 0 else 0.0
                        # Add exponential curve to spread out values
                        similarity = min(1.0, similarity ** 0.85)
                else:
                    # For text-based or general searches
                    if score_range > 0.01:  # If there's a meaningful difference between scores
                        # Enhanced score normalization with dynamic scaling
                        normalized_score = (raw_score - min_score) / score_range
                        # Apply curve to increase separation between results
                        similarity = min(1.0, normalized_score ** 0.75)  
                    else:
                        # Simple normalization if scores are very close
                        similarity = min(1.0, raw_score / max_score) if max_score > 0 else 0.0
                
                # Add scores to the document
                doc["similarity"] = round(max(min_similarity, similarity), 4)  # Round to 4 decimal places
                doc["raw_score"] = raw_score
                
                # Only include results above threshold
                if doc["similarity"] >= min_similarity:
                    results.append(doc)
            
            # Log results count
            logger.info(f"After filtering and scoring, {len(results)} results remain")
            
            # Sort by similarity
            results.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Improve differentiation in final results by spreading scores if needed
            if results and all(abs(r["similarity"] - results[0]["similarity"]) < 0.001 for r in results):
                logger.info("All results have nearly identical similarity scores, applying spread transformation")
                
                # Apply progressive spread from top score to create a more useful distribution
                count = len(results)
                if count > 1:
                    top_score = max(results[0]["similarity"], min_similarity + 0.3)  # Set a reasonable top score
                    bottom_score = max(min_similarity, min_similarity + 0.01)  # Ensure some differentiation
                    
                    for i, result in enumerate(results):
                        # Calculate a progressive score that decays faster at the top
                        # This creates a more natural curve for visual presentation
                        position_ratio = i / (count - 1)  # 0.0 to 1.0
                        # Use curve that emphasizes differences between top results
                        curve_factor = position_ratio ** 1.5  # Steeper at the top
                        spread_score = top_score - (top_score - bottom_score) * curve_factor
                        result["similarity"] = round(spread_score, 4)
                        
                        # Add a flag to indicate scores were artificially spread
                        result["_adjusted_score"] = True
            
            return results
            
        except Exception as e:
            search_time = time.time() - start_time
            logger.error(f"Search failed after {search_time:.2f}s: {str(e)}", exc_info=True)
            return []

    @retry_on_exception(max_retries=3, retry_interval=1.0)
    def find_similar(self, embedding=None, limit=20, min_similarity=0.1, exclude_id=None, 
                   text_query=None, image_weight=0.7, text_weight=0.3):
        """
        Enhanced function to find similar documents based on text query or vector similarity or both
        
        Args:
            embedding: Vector embedding for similarity search (from CLIP)
            limit: Maximum number of results to return
            min_similarity: Minimum similarity score threshold
            exclude_id: ID of image to exclude from results
            text_query: Text query for searching
            image_weight: Weight for image vector similarity component (when both text and embedding are provided)
            text_weight: Weight for text similarity component (when both text and embedding are provided)
            
        Returns:
            List of documents sorted by similarity
        """
        if not self.is_connected():
            logger.error("Cannot perform similarity search: not connected to Elasticsearch")
            return []
            
        # Determine search mode based on inputs
        has_text = text_query and text_query.strip()
        has_embedding = embedding is not None
        
        # Log search approach
        if has_text and has_embedding:
            logger.info(f"Performing hybrid similarity search with text: '{text_query}' and embedding (weights: image={image_weight}, text={text_weight})")
        elif has_embedding:
            logger.info("Performing vector-based similarity search with CLIP embedding")
        elif has_text:
            logger.info(f"Performing enhanced text-based similarity search for: '{text_query}'")
        else:
            logger.info("Performing general similarity search (no specific criteria)")
            
        # Handle the different search modes
        if has_text and has_embedding:
            # Hybrid search (both text and vector)
            return self._hybrid_search(text_query, embedding, limit, min_similarity, exclude_id, 
                                       image_weight, text_weight)
        elif has_embedding:
            # Vector similarity search only
            return self._vector_search(embedding, limit, min_similarity, exclude_id)
        elif has_text:
            # Text similarity search only
            return self._text_search(text_query, limit, min_similarity, exclude_id)
        else:
            # Fallback to general search
            query_body = {"match_all": {}}
            
            # Add exclusion if provided
            if exclude_id:
                query_body = {
                    "bool": {
                        "must": [query_body],
                        "must_not": [
                            {"term": {"id": exclude_id}},
                            {"term": {"filename": exclude_id}},
                            {"term": {"path": exclude_id}}
                        ]
                    }
                }
                
            # Execute the search
            return self._execute_search(query_body, limit, min_similarity)
            
    def _vector_search(self, embedding, limit, min_similarity, exclude_id=None):
        """
        Perform vector similarity search using CLIP embedding
        """
        # Debug the embedding
        embedding_size = len(embedding) if isinstance(embedding, list) else (embedding.size if hasattr(embedding, 'size') else 'unknown')
        logger.info(f"Performing vector search with embedding of size {embedding_size}")
        
        # Create script score query for vector similarity
        script_score_query = {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    # Handle missing embeddings and apply proper scaling
                    "source": """
                        // Check if document has embedding field
                        if (!doc.containsKey('embedding')) {
                            // No embedding, assign a very low score
                            return 0.01;
                        }
                        
                        // Get embedding vector and compute similarity
                        try {
                            double cosine = cosineSimilarity(params.query_vector, 'embedding');
                            // Convert from [-1,1] to [0,1] range
                            double normalized = (cosine + 1.0) / 2.0;
                            // Apply exponential scaling to emphasize high similarity
                            return Math.pow(normalized, 0.5) * 2.0;
                        } catch (Exception e) {
                            // Error computing similarity, assign a very low score
                            return 0.01;
                        }
                    """,
                    "params": {"query_vector": embedding}
                }
            }
        }
        
        # Add exclusion if provided
        if exclude_id:
            script_score_query["script_score"]["query"] = {
                "bool": {
                    "must": [{"match_all": {}}],
                    "must_not": [
                        {"term": {"id": exclude_id}},
                        {"term": {"filename": exclude_id}},
                        {"term": {"path": exclude_id}}
                    ]
                }
            }
            
        # Add function score to consider confidence factors
        function_score_query = {
            "function_score": {
                "query": script_score_query,
                "functions": [
                    # Down-weight results with fallback analysis
                    {
                        "script_score": {
                            "script": {
                                "source": "doc.containsKey('has_fallback_analysis') && doc['has_fallback_analysis'].value ? 0.75 : 1.0"
                            }
                        }
                    },
                    # Add recency boost
                    {
                        "gauss": {
                            "timestamp": {
                                "scale": "30d",
                                "decay": 0.5
                            }
                        },
                        "weight": 0.2
                    }
                ],
                "score_mode": "multiply",
                "boost_mode": "multiply"
            }
        }
        
        return self._execute_search(function_score_query, limit, min_similarity)
        
    def _text_search(self, text_query, limit, min_similarity, exclude_id=None):
        """
        Perform enhanced text-based similarity search
        """
        # Split query into terms for better matching control
        query_terms = [term.strip() for term in text_query.split() if term.strip()]
        min_should_match = min(len(query_terms), 2) if len(query_terms) > 1 else 1
        
        # Build enhanced similar query with multiple fields and clauses
        should_clauses = []
        
        # Match on all important fields with appropriate boosts
        should_clauses.append({
            "match": {
                "patterns.main_theme.raw": {
                    "query": text_query,
                    "boost": 5.0
                }
            }
        })
        
        should_clauses.append({
            "match": {
                "patterns.primary_pattern.raw": {
                    "query": text_query,
                    "boost": 4.5
                }
            }
        })
        
        # Multi match across standard fields
        should_clauses.append({
            "multi_match": {
                "query": text_query,
                "fields": [
                    "patterns.main_theme^3.5",
                    "patterns.main_theme.partial^3.0",
                    "patterns.primary_pattern^3.0",
                    "patterns.primary_pattern.partial^2.5",
                    "patterns.prompt.final_prompt^2.0",
                    "patterns.prompt.final_prompt.partial^1.5",
                    "patterns.style_keywords^2.0"
                ],
                "type": "best_fields",
                "fuzziness": "AUTO",
                "prefix_length": 2,
                "boost": 3.0
            }
        })
        
        # Nested queries for arrays of data
        for nested_path, boost in [
            ("patterns.content_details", 3.0),
            ("patterns.stylistic_attributes", 2.5),
            ("patterns.secondary_patterns", 2.0)
        ]:
            should_clauses.append({
                "nested": {
                    "path": nested_path,
                    "query": {
                        "bool": {
                            "should": [
                                {"match": {f"{nested_path}.name.raw": {"query": text_query, "boost": boost}}},
                                {"match": {f"{nested_path}.name": {"query": text_query, "boost": boost * 0.8}}},
                                {"match": {f"{nested_path}.name.partial": {"query": text_query, "boost": boost * 0.6}}}
                            ]
                        }
                    },
                    "score_mode": "max",
                    "boost": boost
                }
            })
        
        # Add color search
        should_clauses.append({
            "nested": {
                "path": "colors.dominant_colors",
                "query": {
                    "bool": {
                        "should": [
                            {"match": {"colors.dominant_colors.name.raw": {"query": text_query, "boost": 2.5}}},
                            {"match": {"colors.dominant_colors.name": {"query": text_query, "boost": 2.0}}},
                            {"match": {"colors.dominant_colors.name.partial": {"query": text_query, "boost": 1.5}}},
                            {"match": {"colors.dominant_colors.name.synonym": {"query": text_query, "boost": 2.0}}}
                        ]
                    }
                },
                "score_mode": "max",
                "boost": 2.0
            }
        })
        
        # Build bool query
        bool_query = {
            "bool": {
                "should": should_clauses,
                "minimum_should_match": 1
            }
        }
        
        # Add exclusion if provided
        if exclude_id:
            bool_query["bool"]["must_not"] = [
                {"term": {"id": exclude_id}},
                {"term": {"filename": exclude_id}},
                {"term": {"path": exclude_id}}
            ]
        
        # Wrap in function_score query to factor in confidence scores
        function_score_query = {
            "function_score": {
                "query": bool_query,
                "functions": [
                    {
                        "field_value_factor": {
                            "field": "patterns.main_theme_confidence",
                            "factor": 1.5,
                            "missing": 0.8,
                            "modifier": "log1p"
                        }
                    },
                    {
                        "field_value_factor": {
                            "field": "patterns.pattern_confidence",
                            "factor": 1.2,
                            "missing": 0.7,
                            "modifier": "log1p"
                        }
                    },
                    # Reduce score for fallback analysis results
                    {
                        "script_score": {
                            "script": {
                                "source": "doc.containsKey('has_fallback_analysis') && doc['has_fallback_analysis'].value ? 0.75 : 1.0"
                            }
                        }
                    },
                    # Add recency boost
                    {
                        "gauss": {
                            "timestamp": {
                                "scale": "30d",
                                "decay": 0.5
                            }
                        },
                        "weight": 0.3
                    }
                ],
                "score_mode": "multiply",
                "boost_mode": "multiply"
            }
        }
        
        return self._execute_search(function_score_query, limit, min_similarity)
        
    def _hybrid_search(self, text_query, embedding, limit, min_similarity, exclude_id=None,
                     image_weight=0.7, text_weight=0.3):
        """
        Perform hybrid search combining both text and vector similarity
        
        Args:
            text_query: Text query for text-based search
            embedding: CLIP embedding vector for image-based search
            limit: Maximum number of results to return
            min_similarity: Minimum similarity threshold
            exclude_id: ID of image to exclude from results
            image_weight: Weight for image vector similarity (0.0-1.0)
            text_weight: Weight for text similarity (0.0-1.0)
            
        Returns:
            List of documents sorted by similarity
        """
        # Validate and normalize weights
        sum_weights = image_weight + text_weight
        if sum_weights <= 0:
            image_weight, text_weight = 0.5, 0.5  # Default to equal weights
        else:
            # Normalize weights to sum to 1.0
            image_weight = image_weight / sum_weights
            text_weight = text_weight / sum_weights
        
        # Debug the embedding
        embedding_size = len(embedding) if isinstance(embedding, list) else (embedding.size if hasattr(embedding, 'size') else 'unknown')
        logger.info(f"Performing hybrid search with embedding of size {embedding_size} and text query '{text_query}'")
        logger.info(f"Using weights: image={image_weight:.2f}, text={text_weight:.2f}")
        
        # Create separate queries for text and vector components
        
        # Text component
        text_should_clauses = []
        
        # Key field matches with high boost
        text_should_clauses.append({
            "match": {
                "patterns.main_theme.raw": {
                    "query": text_query,
                    "boost": 5.0
                }
            }
        })
        
        text_should_clauses.append({
            "match": {
                "patterns.primary_pattern.raw": {
                    "query": text_query,
                    "boost": 4.5
                }
            }
        })
        
        # Multi-match query across all relevant fields with appropriate boosting
        text_should_clauses.append({
            "multi_match": {
                "query": text_query,
                "fields": [
                    "patterns.main_theme^3.0",
                    "patterns.main_theme.partial^2.5",
                    "patterns.primary_pattern^2.5",
                    "patterns.primary_pattern.partial^2.0",
                    "patterns.prompt.final_prompt^1.5",
                    "patterns.style_keywords^2.0"
                ],
                "type": "best_fields",
                "fuzziness": "AUTO",
                "prefix_length": 2,
                "boost": 3.0
            }
        })
        
        # Nested field searches for arrays
        for nested_path, boost in [
            ("patterns.content_details", 2.5),
            ("patterns.stylistic_attributes", 2.0),
            ("patterns.secondary_patterns", 1.5),
            ("colors.dominant_colors", 2.0)
        ]:
            text_should_clauses.append({
                "nested": {
                    "path": nested_path,
                    "query": {
                        "bool": {
                            "should": [
                                {"match": {f"{nested_path}.name.raw": {"query": text_query, "boost": boost}}},
                                {"match": {f"{nested_path}.name": {"query": text_query, "boost": boost * 0.8}}}
                            ]
                        }
                    },
                    "score_mode": "max",
                    "boost": boost
                }
            })
            
        # Text component bool query with minimum_should_match for better precision
        text_bool_query = {
            "bool": {
                "should": text_should_clauses,
                "minimum_should_match": 1
            }
        }
        
        # Enhanced script for blending text and vector similarity
        blend_script = f"""
            // Get original query score (text similarity)
            double textScore = _score;
            
            // Get vector similarity contribution
            double vectorScore = 0.0;  // Default low score
            
            // Check if document has embedding field
            if (doc.containsKey('embedding')) {{
                try {{
                    // Calculate cosine similarity between query vector and document vector
                    double cosine = cosineSimilarity(params.query_vector, 'embedding');
                    
                    // Convert from [-1,1] to [0,1] range
                    double normalized = (cosine + 1.0) / 2.0;
                    
                    // Emphasize high similarities with exponential scaling
                    // This makes very similar images stand out more
                    vectorScore = Math.pow(normalized, 0.8) * 2.0;
                }} catch (Exception e) {{
                    // Keep default low score on error
                    vectorScore = 0.01;
                }}
            }}
            
            // Normalize text score
            double normalizedTextScore = Math.min(1.0, textScore / 10.0);
            
            // Blend scores using configurable weights
            double finalScore = (normalizedTextScore * {text_weight}) + (vectorScore * {image_weight});
            
            // Ensure score is at least 0.01 to prevent potential normalization issues
            return Math.max(finalScore, 0.01);
        """
        
        # Build the hybrid query with the improved blending script
        combined_query = {
            "function_score": {
                "query": text_bool_query,
                "script_score": {
                    "script": {
                        "source": blend_script,
                        "params": {"query_vector": embedding}
                    }
                },
                "boost_mode": "replace"  # Replace the original score with our hybrid score
            }
        }
        
        # Add exclusion if provided
        if exclude_id:
            combined_query["function_score"]["query"]["bool"]["must_not"] = [
                {"term": {"id": exclude_id}},
                {"term": {"filename": exclude_id}},
                {"term": {"path": exclude_id}}
            ]
            
        # Add final function score with confidence boosting and recency
        function_score_query = {
            "function_score": {
                "query": combined_query,
                "functions": [
                    {
                        "field_value_factor": {
                            "field": "patterns.main_theme_confidence",
                            "factor": 1.2,
                            "missing": 0.8,
                            "modifier": "log1p"
                        }
                    },
                    {
                        "field_value_factor": {
                            "field": "patterns.pattern_confidence",
                            "factor": 1.1,
                            "missing": 0.7,
                            "modifier": "log1p"
                        }
                    },
                    # Down-weight fallback results
                    {
                        "script_score": {
                            "script": {
                                "source": "doc.containsKey('has_fallback_analysis') && doc['has_fallback_analysis'].value ? 0.75 : 1.0"
                            }
                        }
                    },
                    # Add recency boost (prefer newer items slightly)
                    {
                        "gauss": {
                            "timestamp": {
                                "scale": "30d",
                                "decay": 0.5
                            }
                        },
                        "weight": 0.2
                    }
                ],
                "score_mode": "multiply",
                "boost_mode": "multiply"
            }
        }
        
        return self._execute_search(function_score_query, limit, min_similarity) 