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
            return self.es.ping()
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
                                "floral, flower, botanical, bloom, garden",
                                "geometric, geometry, shape, pattern",
                                "stripe, striped, linear, line, banded",
                                "dot, dots, polka dot, spotted, circular pattern",
                                "plaid, tartan, check, checkered",
                                "paisley, paisleys",
                                "abstract, non-representational",
                                "tropical, exotic, jungle, palm",
                                "damask, scroll, acanthus",
                                "chevron, zigzag, herringbone",
                                "animal, wildlife, fauna, creature"
                            ]
                        }
                    }
                }
            }

            # Enhanced mappings
            mappings = {
                "properties": {
                    "id": {"type": "keyword"},
                    "filename": {"type": "keyword"},
                    "path": {"type": "keyword"},
                    "thumbnail_path": {"type": "keyword"},
                    "timestamp": {"type": "date", "format": "epoch_second||epoch_millis||strict_date_optional_time"},
                    "embedding": {"type": "dense_vector", "dims": 512},
                    "patterns": {
                        "properties": {
                            "main_theme": {
                                "type": "text",
                                "analyzer": "standard",
                                "boost": 3.0,
                                "fields": {
                                    "raw": {"type": "keyword"},
                                    "partial": {"type": "text", "analyzer": "partial_analyzer"}
                                }
                            },
                            "main_theme_confidence": {"type": "float"},
                            "primary_pattern": {
                                "type": "text",
                                "analyzer": "standard",
                                "boost": 2.5,
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
                                        "analyzer": "standard",
                                        "boost": 2.0,
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
                                        "analyzer": "standard",
                                        "boost": 1.5,
                                        "fields": {
                                            "raw": {"type": "keyword"},
                                            "partial": {"type": "text", "analyzer": "partial_analyzer"}
                                        }
                                    },
                                    "confidence": {"type": "float"}
                                }
                            },
                            "category": {
                                "type": "text",
                                "analyzer": "standard",
                                "fields": {
                                    "raw": {"type": "keyword"},
                                    "partial": {"type": "text", "analyzer": "partial_analyzer"}
                                }
                            },
                            "category_confidence": {"type": "float"},
                            "style_keywords": {"type": "text"},
                            "prompt": {
                                "properties": {
                                    "final_prompt": {
                                        "type": "text",
                                        "analyzer": "standard",
                                        "boost": 1.0,
                                        "fields": {
                                            "partial": {"type": "text", "analyzer": "partial_analyzer"}
                                        }
                                    }
                                }
                            },
                            "secondary_patterns": {
                                "type": "nested",
                                "properties": {
                                    "name": {
                                        "type": "text",
                                        "analyzer": "standard",
                                        "fields": {
                                            "raw": {"type": "keyword"},
                                            "partial": {"type": "text", "analyzer": "partial_analyzer"}
                                        }
                                    },
                                    "confidence": {"type": "float"}
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
                return False
        
        # Extract the document ID
        doc_id = document.get("id") or document.get("path") or document.get("filename")
        if not doc_id:
            logger.error("Document must have an 'id', 'path', or 'filename' field")
            return False
            
        self.client.index(index=self.index_name, id=doc_id, document=document)
        logger.info(f"Indexed document with ID: {doc_id}")
        return True
    
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
                return False
            
        # Process documents in chunks
        actions = []
        for doc in documents:
            # Extract ID
            doc_id = doc.get("id") or doc.get("path") or doc.get("filename")
            if not doc_id:
                logger.warning("Skipping document without ID field")
                continue
                
            action = {
                "_index": self.index_name,
                "_id": doc_id,
                "_source": doc
            }
            actions.append(action)
            
        if not actions:
            logger.warning("No valid documents to bulk index")
            return False

            
        # Execute bulk indexing
        success, failed = helpers.bulk(
            self.client, 
            actions, 
            chunk_size=chunk_size,
            raise_on_error=False
        )
        
        logger.info(f"Bulk indexed {success} documents, {len(failed) if failed else 0} failed")
        return success > 0
    
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
        
        # 1. Exact matches on main_theme with high boost
        should_clauses.append({
            "match": {
                "patterns.main_theme.raw": {
                    "query": query,
                    "boost": 5.0
                }
            }
        })
        
        # 2. Exact matches on primary_pattern with high boost
        should_clauses.append({
            "match": {
                "patterns.primary_pattern.raw": {
                    "query": query,
                    "boost": 4.5
                }
            }
        })
        
        # 3. Multi-match across all relevant fields
        should_clauses.append({
            "multi_match": {
                "query": query,
                "fields": [
                    "patterns.main_theme^3",
                    "patterns.main_theme.partial^2.5",
                    "patterns.primary_pattern^2.5",
                    "patterns.primary_pattern.partial^2",
                    "patterns.prompt.final_prompt^1.2",
                    "patterns.prompt.final_prompt.partial^1"
                ],
                "type": "best_fields",
                "fuzziness": "AUTO",
                "prefix_length": 2,
                "boost": 3.0
            }
        })
        
        # 4. Nested query for content details
        should_clauses.append({
            "nested": {
                "path": "patterns.content_details",
                "query": {
                    "bool": {
                        "should": [
                            {
                                "match": {
                                    "patterns.content_details.name": {
                                        "query": query,
                                        "boost": 2.0
                                    }
                                }
                            },
                            {
                                "match": {
                                    "patterns.content_details.name.partial": {
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
        
        # 5. Nested query for stylistic attributes
        should_clauses.append({
            "nested": {
                "path": "patterns.stylistic_attributes",
                "query": {
                    "bool": {
                        "should": [
                            {
                                "match": {
                                    "patterns.stylistic_attributes.name": {
                                        "query": query,
                                        "boost": 1.5
                                    }
                                }
                            },
                            {
                                "match": {
                                    "patterns.stylistic_attributes.name.partial": {
                                        "query": query,
                                        "boost": 1.0
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
        
        # 6. Nested query for colors
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
                                        "boost": 2.0
                                    }
                                }
                            },
                            {
                                "match": {
                                    "colors.dominant_colors.name": {
                                        "query": query,
                                        "boost": 1.5
                                    }
                                }
                            },
                            {
                                "match": {
                                    "colors.dominant_colors.name.partial": {
                                        "query": query,
                                        "boost": 1.0
                                    }
                                }
                            }
                        ]
                    }
                },
                "score_mode": "avg",
                "boost": 1.5
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
        Helper method to execute a search with the given query body
        """
        start_time = time.time()
        try:
            response = self.client.search(
                index=self.index_name,
                body={
                    "size": limit,
                    "query": query_body,
                    "_source": True
                }
            )
            
            search_time = time.time() - start_time
            logger.info(f"Search completed in {search_time:.2f}s")
            
            # Get max score for normalization
            max_score = response["hits"]["max_score"] if response["hits"]["hits"] else 1.0
            
            # Process and normalize results
            results = []
            for hit in response["hits"]["hits"]:
                doc = hit["_source"]
                raw_score = hit["_score"]
                
                # Normalize score to 0-1 range
                similarity = min(1.0, raw_score / max_score) if max_score > 0 else 0.0
                
                # Add scores to the document
                doc["similarity"] = max(min_similarity, similarity)
                doc["raw_score"] = raw_score
                
                # Only include results above threshold
                if doc["similarity"] >= min_similarity:
                    results.append(doc)
            
            # Sort by similarity
            results.sort(key=lambda x: x["similarity"], reverse=True)
            
            logger.info(f"Search returned {len(results)} results")
            return results
            
        except Exception as e:
            search_time = time.time() - start_time
            logger.error(f"Search failed after {search_time:.2f}s: {str(e)}")
            return []
    
    @retry_on_exception(max_retries=3, retry_interval=1.0)
    def find_similar(self, embedding=None, limit=20, min_similarity=0.1, exclude_id=None, 
                   text_query=None):
        """
        Find similar documents based on text query or vector similarity
        
        Args:
            embedding: Vector embedding for similarity search
            limit: Maximum number of results to return
            min_similarity: Minimum similarity score threshold
            exclude_id: ID of image to exclude from results
            text_query: Text query for searching

        Returns:
            List of documents sorted by similarity
        """
        if not self.is_connected():
            logger.error("Cannot perform similarity search: not connected to Elasticsearch")
            return []
            

        # Build the main query
        if text_query:
            logger.info(f"Performing enhanced text-based similarity search for: '{text_query}'")
            
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
                        "patterns.main_theme^3",
                        "patterns.main_theme.partial^2.5",
                        "patterns.primary_pattern^2.5",
                        "patterns.primary_pattern.partial^2",
                        "patterns.prompt.final_prompt^1.2",
                        "patterns.prompt.final_prompt.partial^1"
                    ],
                    "type": "best_fields",
                    "fuzziness": "AUTO",
                    "prefix_length": 2,
                    "boost": 3.0
                }
            })
            
            # Nested queries for content details and stylistic attributes
            should_clauses.append({
                "nested": {
                    "path": "patterns.content_details",
                    "query": {
                        "bool": {
                            "should": [
                                {"match": {"patterns.content_details.name": {"query": text_query, "boost": 2.0}}},
                                {"match": {"patterns.content_details.name.partial": {"query": text_query, "boost": 1.5}}}
                            ]
                        }
                    },
                    "score_mode": "max",
                    "boost": 2.0
                }
            })
            
            should_clauses.append({
                "nested": {
                    "path": "patterns.stylistic_attributes",
                    "query": {
                        "bool": {
                            "should": [
                                {"match": {"patterns.stylistic_attributes.name": {"query": text_query, "boost": 1.5}}},
                                {"match": {"patterns.stylistic_attributes.name.partial": {"query": text_query, "boost": 1.0}}}
                            ]
                        }
                    },
                    "score_mode": "max",
                    "boost": 1.5
                }
            })
            

            # Add dominant color search
            should_clauses.append({
                "nested": {
                    "path": "colors.dominant_colors",
                    "query": {
                        "bool": {
                            "should": [
                                {"match": {"colors.dominant_colors.name.raw": {"query": text_query, "boost": 2.0}}},
                                {"match": {"colors.dominant_colors.name": {"query": text_query, "boost": 1.5}}},
                                {"match": {"colors.dominant_colors.name.partial": {"query": text_query, "boost": 1.0}}}
                            ]
                        }
                    },
                    "score_mode": "avg",
                    "boost": 1.5
                }
            })
            
            # Build bool query
            bool_query = {
                "bool": {
                    "should": should_clauses,
                    "minimum_should_match": 1
                }
            }
            
            # Wrap in function_score query to factor in confidence scores
            query_body = {
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
        elif embedding is not None:
            # Vector similarity search
            logger.info("Performing vector-based similarity search")
            query_body = {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": embedding}
                    }
                }
            }
        else:
            # Fallback to general search
            logger.info("Performing general similarity search")
            query_body = {"match_all": {}}
            
        # Add exclusion if provided
        if exclude_id:
            # Create a must_not clause to exclude the reference document
            if "function_score" in query_body:
                if "bool" not in query_body["function_score"]["query"]:
                    query_body["function_score"]["query"] = {
                        "bool": {
                            "must": [query_body["function_score"]["query"]],
                            "must_not": []
                        }
                    }
                
                # Add exclusion by ID and filename
                query_body["function_score"]["query"]["bool"]["must_not"] = [
                    {"term": {"id": exclude_id}},
                    {"term": {"filename": exclude_id}},
                    {"term": {"path": exclude_id}}

                ]
            elif "script_score" in query_body:
                # For vector search
                if "bool" not in query_body["script_score"]["query"]:
                    query_body["script_score"]["query"] = {
                        "bool": {
                            "must": [query_body["script_score"]["query"]],
                            "must_not": []
                        }
                    }
                
                # Add exclusion
                query_body["script_score"]["query"]["bool"]["must_not"] = [
                    {"term": {"id": exclude_id}},
                    {"term": {"filename": exclude_id}},
                    {"term": {"path": exclude_id}}
                ]
            else:
                # For simple queries
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
                
            logger.info(f"Excluding ID: {exclude_id}")
        
        # Execute the search and process results
        start_time = time.time()
        try:
            response = self.client.search(
                index=self.index_name,
                body={
                    "size": limit,
                    "query": query_body,
                    "_source": True
                }
            )
            
            search_time = time.time() - start_time
            logger.info(f"Similarity search completed in {search_time:.2f}s")
            
            # Process results
            results = []
            max_score = response["hits"]["max_score"] if response["hits"]["hits"] else 1.0
            
            for hit in response["hits"]["hits"]:
                doc = hit["_source"]
                raw_score = hit["_score"]
                
                # Normalize score to 0-1 range
                similarity = min(1.0, raw_score / max_score) if max_score > 0 else 0.0
                
                # Add similarity score
                doc["raw_score"] = raw_score
                doc["similarity"] = max(min_similarity, similarity)
                
                # Only include results above threshold
                if doc["similarity"] >= min_similarity:
                    results.append(doc)
            
            # Sort by similarity (descending)
            results.sort(key=lambda x: x["similarity"], reverse=True)
            
            logger.info(f"Similarity search returned {len(results)} results (of {len(response['hits']['hits'])} total)")
            return results
            
        except Exception as e:
            search_time = time.time() - start_time
            logger.error(f"Similarity search failed after {search_time:.2f}s: {str(e)}")
            return [] 
