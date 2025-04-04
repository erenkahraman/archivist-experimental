from elasticsearch import Elasticsearch, helpers
from elasticsearch.exceptions import NotFoundError
import logging
from typing import Dict, List, Any, Optional, Generator
import config
import time

# Configure logger
logger = logging.getLogger(__name__)

class ElasticsearchClient:
    """Client for interacting with Elasticsearch"""
    
    def __init__(self, hosts: List[str] = None, cloud_id: str = None, api_key: str = None, username: str = None, password: str = None):
        """
        Initialize the Elasticsearch client.
        
        Args:
            hosts: List of Elasticsearch host URLs
            cloud_id: Cloud ID for Elastic Cloud
            api_key: API key for authentication
            username: Username for basic authentication
            password: Password for basic authentication
        """
        self.index_name = "images"
        
        # Connection parameters
        self.connection_params = {}
        
        # Set hosts if provided
        if hosts:
            self.connection_params["hosts"] = hosts
            
        # Set cloud_id if provided
        if cloud_id:
            self.connection_params["cloud_id"] = cloud_id
            
        # Set authentication
        if api_key:
            self.connection_params["api_key"] = api_key
        elif username and password:
            self.connection_params["basic_auth"] = (username, password)
            
        # Initialize client
        try:
            self.es = Elasticsearch(**self.connection_params)
            logger.info(f"Connected to Elasticsearch: {self.es.info()['version']['number']}")
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {str(e)}")
            self.es = None
    
    def is_connected(self) -> bool:
        """Check if connected to Elasticsearch"""
        if not self.es:
            return False
        try:
            return self.es.ping()
        except Exception as e:
            logger.error(f"Elasticsearch connection error: {str(e)}")
            return False
    
    def create_index(self) -> bool:
        """
        Create the images index with appropriate mappings.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_connected():
            logger.error("Cannot create index: not connected to Elasticsearch")
            return False
            
        # Define the mapping for the images index
        mappings = {
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "filename": {"type": "keyword"},
                    "path": {"type": "keyword"},
                    "thumbnail_path": {"type": "keyword"},
                    "timestamp": {"type": "date"},
                    
                    # Pattern information
                    "patterns": {
                        "properties": {
                            # New fields with multi-field mappings
                            "main_theme": {
                                "type": "text", 
                                "analyzer": "synonym_analyzer",
                                "fields": {
                                    "raw": {"type": "keyword"},
                                    "partial": {
                                        "type": "text",
                                        "analyzer": "partial_analyzer",
                                        "search_analyzer": "standard"
                                    }
                                }
                            },
                            "main_theme_confidence": {"type": "float"},
                            "content_details": {
                                "type": "nested",
                                "properties": {
                                    "name": {
                                        "type": "text", 
                                        "analyzer": "synonym_analyzer",
                                        "fields": {
                                            "raw": {"type": "keyword"},
                                            "partial": {
                                                "type": "text",
                                                "analyzer": "partial_analyzer",
                                                "search_analyzer": "standard"
                                            }
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
                                        "analyzer": "synonym_analyzer",
                                        "fields": {
                                            "raw": {"type": "keyword"},
                                            "partial": {
                                                "type": "text",
                                                "analyzer": "partial_analyzer",
                                                "search_analyzer": "standard"
                                            }
                                        }
                                    },
                                    "confidence": {"type": "float"}
                                }
                            },
                            "primary_pattern": {
                                "type": "text", 
                                "analyzer": "synonym_analyzer",
                                "fields": {
                                    "raw": {"type": "keyword"},
                                    "partial": {
                                        "type": "text",
                                        "analyzer": "partial_analyzer",
                                        "search_analyzer": "standard"
                                    }
                                }
                            },
                            "pattern_confidence": {"type": "float"},
                            "secondary_patterns": {
                                "type": "nested",
                                "properties": {
                                    "name": {
                                        "type": "text", 
                                        "analyzer": "synonym_analyzer",
                                        "fields": {
                                            "raw": {"type": "keyword"},
                                            "partial": {
                                                "type": "text",
                                                "analyzer": "partial_analyzer",
                                                "search_analyzer": "standard"
                                            }
                                        }
                                    },
                                    "confidence": {"type": "float"}
                                }
                            },
                            "elements": {
                                "type": "nested",
                                "properties": {
                                    "name": {
                                        "type": "text", 
                                        "analyzer": "synonym_analyzer",
                                        "fields": {
                                            "raw": {"type": "keyword"},
                                            "partial": {
                                                "type": "text",
                                                "analyzer": "partial_analyzer",
                                                "search_analyzer": "standard"
                                            }
                                        }
                                    },
                                    "confidence": {"type": "float"}
                                }
                            },
                            "prompt": {
                                "properties": {
                                    "original_prompt": {"type": "text", "analyzer": "standard"},
                                    "final_prompt": {
                                        "type": "text", 
                                        "analyzer": "synonym_analyzer",
                                        "fields": {
                                            "partial": {
                                                "type": "text",
                                                "analyzer": "partial_analyzer",
                                                "search_analyzer": "standard"
                                            }
                                        }
                                    }
                                }
                            },
                            "style_keywords": {
                                "type": "text", 
                                "analyzer": "synonym_analyzer",
                                "fields": {
                                    "raw": {"type": "keyword"},
                                    "partial": {
                                        "type": "text",
                                        "analyzer": "partial_analyzer",
                                        "search_analyzer": "standard"
                                    }
                                }
                            }
                        }
                    },
                    
                    # Color information
                    "colors": {
                        "properties": {
                            "dominant_colors": {
                                "type": "nested",
                                "properties": {
                                    "name": {
                                        "type": "text", 
                                        "analyzer": "synonym_analyzer",
                                        "fields": {
                                            "raw": {"type": "keyword"},
                                            "partial": {
                                                "type": "text",
                                                "analyzer": "partial_analyzer",
                                                "search_analyzer": "standard"
                                            }
                                        }
                                    },
                                    "hex": {"type": "keyword"},
                                    "proportion": {"type": "float"}
                                }
                            },
                            "color_palette": {
                                "type": "nested",
                                "properties": {
                                    "name": {"type": "keyword"},
                                    "hex": {"type": "keyword"}
                                }
                            }
                        }
                    }
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "custom_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "stemmer"]
                        },
                        "partial_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "edge_ngram_filter"]
                        },
                        "synonym_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": [
                                "lowercase",
                                "synonym_filter",
                                "stemmer"
                            ]
                        }
                    },
                    "filter": {
                        "edge_ngram_filter": {
                            "type": "edge_ngram",
                            "min_gram": 2,
                            "max_gram": 10
                        },
                        "synonym_filter": {
                            "type": "synonym_graph",
                            "expand": true,
                            "synonyms": [
                                # Color synonyms
                                "crimson, ruby, cherry, scarlet => red",
                                "burgundy, maroon => dark red",
                                "salmon, coral pink => pink",
                                "navy, cobalt, azure => blue",
                                "teal, turquoise => blue green",
                                "khaki, tan, beige => light brown",
                                "gold, goldenrod => yellow",
                                "lime, chartreuse => light green",
                                "forest => dark green",
                                "olive => yellow green",
                                "mint => pale green",
                                "magenta, fuchsia => bright pink",
                                "lavender, periwinkle => light purple",
                                "indigo => deep blue",
                                "amber => orange yellow",
                                "slate => blue gray",
                                "charcoal => dark gray",
                                "ivory, cream => off white",
                                "silver => light gray",
                                
                                # Pattern synonyms
                                "flower, bloom, floral => floral",
                                "geometric, shape => geometric",
                                "dots, polka dot => dotted",
                                "stripe, striped, stripes => striped",
                                "diamonds, diamond pattern => diamond",
                                "grid, checks, checked => checkered",
                                "abstract, non representational => abstract",
                                "zig zag, chevron => chevron",
                                "animal print, animal skin => animal print",
                                "paisley, teardrop => paisley",
                                "squares, blocks, grid => geometric",
                                "curvy, wavy, waves => wave",
                                "botanical, natural, nature inspired => natural",
                                "gradient, ombre => gradient",
                                "herringbone, fishbone => herringbone",
                                "damask, ornate pattern => damask",
                                "ikat, ethnic => ethnic",
                                "abstract geometric => abstract geometric",
                                "medallion => circular pattern",
                                "toile, scenic => toile",
                                
                                # Textile terms
                                "fabric, cloth, textile, material => textile",
                                "cotton, natural fiber => cotton",
                                "silk, silky => silk",
                                "wool, woolen => wool",
                                "polyester, synthetic => synthetic",
                                "linen, flax => linen",
                                "satin, glossy => satin",
                                "velvet, plush => velvet",
                                "denim, jeans => denim",
                                "leather, hide => leather",
                                "suede => soft leather",
                                "knit, knitted => knit",
                                "woven, weave => woven",
                                "embroidered, embroidery => embroidery",
                                "brocade, jacquard => brocade",
                                "tapestry, wall hanging => tapestry",
                                "canvas, heavy fabric => canvas",
                                "upholstery => furniture fabric"
                            ]
                        }
                    }
                }
            }
        }
        
        try:
            # Check if index exists
            if self.es.indices.exists(index=self.index_name):
                logger.info(f"Index '{self.index_name}' already exists")
                return True
                
            # Create the index
            self.es.indices.create(index=self.index_name, body=mappings)
            logger.info(f"Created index '{self.index_name}' with mappings")
            return True
        except Exception as e:
            logger.error(f"Failed to create index: {str(e)}")
            return False
    
    def index_document(self, document: Dict[str, Any], invalidate_cache: bool = True) -> bool:
        """
        Index a single document into Elasticsearch.
        
        Args:
            document: The document to index
            invalidate_cache: Whether to invalidate the search cache after indexing
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_connected():
            logger.error("Cannot index document: not connected to Elasticsearch")
            return False
            
        try:
            # Make sure the index exists
            if not self.es.indices.exists(index=self.index_name):
                self.create_index()
            
            # Extract the document ID
            doc_id = document.get("id", document.get("path", None))
            if not doc_id:
                logger.error("Document must have an 'id' or 'path' field")
                return False
                
            # Index the document
            self.es.index(index=self.index_name, id=doc_id, document=document)
            logger.info(f"Indexed document with ID: {doc_id}")
            
            # Invalidate cache if requested
            if invalidate_cache:
                from src.search_engine import search_engine  # Import here to avoid circular imports
                if hasattr(search_engine, 'cache'):
                    search_engine.cache.invalidate_all()
                    logger.info("Invalidated search cache after indexing document")
            
            return True
        except Exception as e:
            logger.error(f"Failed to index document: {str(e)}")
            return False
    
    def bulk_index(self, documents: List[Dict[str, Any]], invalidate_cache: bool = True) -> bool:
        """
        Bulk index multiple documents into Elasticsearch.
        
        Args:
            documents: List of documents to index
            invalidate_cache: Whether to invalidate the search cache after indexing
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_connected():
            logger.error("Cannot bulk index: not connected to Elasticsearch")
            return False
            
        if not documents:
            logger.warning("No documents to bulk index")
            return False
            
        try:
            # Make sure the index exists
            if not self.es.indices.exists(index=self.index_name):
                self.create_index()
                
            # Prepare actions for bulk indexing
            actions = []
            for doc in documents:
                # Extract the document ID
                doc_id = doc.get("id", doc.get("path", None))
                if not doc_id:
                    logger.warning("Skipping document without 'id' or 'path' field")
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
            result = helpers.bulk(self.es, actions)
            logger.info(f"Bulk indexed {result[0]} documents")
            
            # Invalidate cache if requested
            if invalidate_cache:
                from src.search_engine import search_engine  # Import here to avoid circular imports
                if hasattr(search_engine, 'cache'):
                    search_engine.cache.invalidate_all()
                    logger.info("Invalidated search cache after bulk indexing")
            
            return True
        except Exception as e:
            logger.error(f"Failed to bulk index documents: {str(e)}")
            return False
    
    def update_document(self, doc_id: str, document: Dict[str, Any], invalidate_cache: bool = True) -> bool:
        """
        Update an existing document in Elasticsearch.
        
        Args:
            doc_id: The document ID to update
            document: The document data to update
            invalidate_cache: Whether to invalidate the search cache after updating
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_connected():
            logger.error("Cannot update document: not connected to Elasticsearch")
            return False
            
        try:
            self.es.update(index=self.index_name, id=doc_id, doc=document)
            logger.info(f"Updated document with ID: {doc_id}")
            
            # Invalidate cache if requested
            if invalidate_cache:
                from src.search_engine import search_engine  # Import here to avoid circular imports
                if hasattr(search_engine, 'cache'):
                    search_engine.cache.invalidate_all()
                    logger.info("Invalidated search cache after document update")
            
            return True
        except NotFoundError:
            # Document doesn't exist, index it instead
            logger.info(f"Document with ID {doc_id} doesn't exist, indexing instead")
            return self.index_document(document, invalidate_cache)
        except Exception as e:
            logger.error(f"Failed to update document: {str(e)}")
            return False
    
    def delete_document(self, doc_id: str, invalidate_cache: bool = True) -> bool:
        """
        Delete a document from Elasticsearch.
        
        Args:
            doc_id: The document ID to delete
            invalidate_cache: Whether to invalidate the search cache after deletion
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_connected():
            logger.error("Cannot delete document: not connected to Elasticsearch")
            return False
            
        try:
            self.es.delete(index=self.index_name, id=doc_id)
            logger.info(f"Deleted document with ID: {doc_id}")
            
            # Invalidate cache if requested
            if invalidate_cache:
                from src.search_engine import search_engine  # Import here to avoid circular imports
                if hasattr(search_engine, 'cache'):
                    search_engine.cache.invalidate_all()
                    logger.info("Invalidated search cache after document deletion")
            
            return True
        except NotFoundError:
            logger.warning(f"Document with ID {doc_id} not found")
            return False
        except Exception as e:
            logger.error(f"Failed to delete document: {str(e)}")
            return False
    
    def _parse_query(self, query_string: str) -> Dict[str, Any]:
        """
        Parse a search query into components for advanced search.
        Not used in the simple search implementation.
        
        Args:
            query_string: The raw search query string
            
        Returns:
            Dictionary with query components
        """
        # This method is no longer used but kept for backward compatibility
        return {
            "colors": [],
            "quoted_phrases": [],
            "potential_phrases": [],
            "keywords": [query_string],
            "concepts": [query_string],
            "original": query_string
        }
    
    def search(self, query: str, limit: int = 20, min_similarity: float = 0.1) -> List[Dict[str, Any]]:
        """
        Simple search for images using Elasticsearch.
        
        Args:
            query: The search query string
            limit: Maximum number of results to return
            min_similarity: Minimum similarity score threshold
            
        Returns:
            List of matching documents with similarity scores
        """
        if not self.is_connected():
            logger.error("Cannot search: not connected to Elasticsearch")
            # Use in-memory search instead
            try:
                from src.search_engine import search_engine
                return search_engine._in_memory_search(query, limit)
            except Exception as e:
                logger.error(f"Fallback to in-memory search failed: {str(e)}")
                return []
        
        try:
            # Build a simple query that matches the query term across multiple fields
            dsl = {
                "size": limit,
                "query": {
                    "bool": {
                        "should": [
                            {"match": {"patterns.primary_pattern": query}},
                            {"match": {"patterns.style_keywords": query}},
                            {"match": {"patterns.prompt.final_prompt": query}},
                            {"match": {"colors.dominant_colors.name": query}}
                        ],
                        "minimum_should_match": 1
                    }
                }
            }
            
            # Execute the search
            response = self.es.search(index=self.index_name, body=dsl)
            
            # Process results
            results = []
            for hit in response["hits"]["hits"]:
                doc = hit["_source"]
                score = hit["_score"]
                
                # Normalize the score to 0-1 range for consistency
                normalized_score = min(score / 10.0, 1.0)
                
                # Skip results with score below threshold
                if normalized_score < min_similarity:
                    continue
                    
                # Add score to the document
                doc["similarity"] = normalized_score
                
                results.append(doc)
                
            logger.info(f"Elasticsearch search for '{query}' found {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Elasticsearch search error: {str(e)}")
            return []
    
    def _log_query_details(self, query_body: Dict[str, Any]) -> None:
        """
        Log query details for debugging.
        Not used in the simple search implementation.
        """
        pass
    
    def rebuild_index(self) -> bool:
        """
        Rebuild the index with new mappings and reindex all documents.
        This is required after changing analyzers or mappings.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_connected():
            logger.error("Cannot rebuild index: not connected to Elasticsearch")
            return False
            
        try:
            # Check if index exists
            if not self.es.indices.exists(index=self.index_name):
                logger.info(f"Index '{self.index_name}' doesn't exist, creating it")
                return self.create_index()
                
            # Create a temporary index name
            temp_index = f"{self.index_name}_temp"
            
            logger.info(f"Rebuilding index '{self.index_name}' with new mappings")
            
            # 1. Create a new temporary index with the updated mappings
            old_index = self.index_name
            self.index_name = temp_index
            temp_creation_success = self.create_index()
            self.index_name = old_index
            
            if not temp_creation_success:
                logger.error("Failed to create temporary index for reindexing")
                return False
            
            # 2. Count documents in the original index
            count_request = self.es.count(index=self.index_name)
            total_docs = count_request["count"]
            
            if total_docs == 0:
                logger.info("No documents to reindex. Deleting old index and creating new one.")
                self.es.indices.delete(index=self.index_name)
                self.index_name = temp_index
                return True
                
            logger.info(f"Reindexing {total_docs} documents from '{self.index_name}' to '{temp_index}'")
            
            # 3. Reindex from old to new index
            reindex_body = {
                "source": {
                    "index": self.index_name
                },
                "dest": {
                    "index": temp_index
                }
            }
            
            # Execute reindex
            self.es.reindex(body=reindex_body, wait_for_completion=True)
            
            # 4. Verify new index has all documents
            new_count_request = self.es.count(index=temp_index)
            new_total_docs = new_count_request["count"]
            
            if new_total_docs != total_docs:
                logger.error(f"Document count mismatch after reindexing: {total_docs} vs {new_total_docs}")
                # Clean up temp index
                self.es.indices.delete(index=temp_index)
                return False
                
            # 5. Delete the old index
            logger.info(f"Reindexing complete. Deleting old index '{self.index_name}'")
            self.es.indices.delete(index=self.index_name)
            
            # 6. Create an alias from the old name to the new index
            logger.info(f"Creating alias from '{self.index_name}' to '{temp_index}'")
            self.es.indices.update_aliases(body={
                "actions": [
                    {"add": {"index": temp_index, "alias": self.index_name}}
                ]
            })
            
            # Update our internal index name to point to the new one
            self.index_name = temp_index
            
            logger.info(f"Index rebuild complete with new mappings and {new_total_docs} documents")
            
            # Invalidate cache since we've reindexed everything
            from src.search_engine import search_engine  # Import here to avoid circular imports
            if hasattr(search_engine, 'cache'):
                search_engine.cache.invalidate_all()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to rebuild index: {str(e)}")
            return False
    
    def explain_search_result(self, doc_id: str, query: str) -> Dict[str, Any]:
        """
        Explain why a document matched a query. Kept for backward compatibility.
        Not used in the simple search implementation.
        
        Args:
            doc_id: The document ID to explain
            query: The query to explain against
            
        Returns:
            Dictionary with explanation details
        """
        return {
            "doc_id": doc_id,
            "query": query,
            "explanation": "Simple search doesn't support detailed explanations"
        } 