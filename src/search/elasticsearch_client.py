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
            self.client = Elasticsearch(**self.connection_params)
            logger.info(f"Connected to Elasticsearch: {self.client.info()['version']['number']}")
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {str(e)}")
            self.client = None
    
    def is_connected(self) -> bool:
        """Check if connected to Elasticsearch"""
        if not self.client:
            return False
        try:
            return self.client.ping()
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
                            "primary_pattern": {"type": "text", "analyzer": "standard"},
                            "pattern_confidence": {"type": "float"},
                            "secondary_patterns": {
                                "type": "nested",
                                "properties": {
                                    "name": {"type": "text", "analyzer": "standard"},
                                    "confidence": {"type": "float"}
                                }
                            },
                            "prompt": {
                                "properties": {
                                    "final_prompt": {"type": "text", "analyzer": "standard"}
                                }
                            },
                            "elements": {
                                "type": "nested",
                                "properties": {
                                    "name": {"type": "text", "analyzer": "standard"},
                                    "confidence": {"type": "float"}
                                }
                            },
                            "style_keywords": {"type": "text", "analyzer": "standard"}
                        }
                    },
                    
                    # Color information
                    "colors": {
                        "properties": {
                            "dominant_colors": {
                                "type": "nested",
                                "properties": {
                                    "name": {"type": "text", "analyzer": "standard"},
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
                        }
                    }
                }
            }
        }
        
        try:
            # Check if index exists
            if self.client.indices.exists(index=self.index_name):
                logger.info(f"Index '{self.index_name}' already exists")
                return True
                
            # Create the index
            self.client.indices.create(index=self.index_name, body=mappings)
            logger.info(f"Created index '{self.index_name}' with mappings")
            return True
        except Exception as e:
            logger.error(f"Failed to create index: {str(e)}")
            return False
    
    def index_document(self, document: Dict[str, Any]) -> bool:
        """
        Index a single document into Elasticsearch.
        
        Args:
            document: The document to index
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_connected():
            logger.error("Cannot index document: not connected to Elasticsearch")
            return False
            
        try:
            # Make sure the index exists
            if not self.client.indices.exists(index=self.index_name):
                self.create_index()
            
            # Extract the document ID
            doc_id = document.get("id", document.get("path", None))
            if not doc_id:
                logger.error("Document must have an 'id' or 'path' field")
                return False
                
            # Index the document
            self.client.index(index=self.index_name, id=doc_id, document=document)
            logger.info(f"Indexed document with ID: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to index document: {str(e)}")
            return False
    
    def bulk_index(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Bulk index multiple documents into Elasticsearch.
        
        Args:
            documents: List of documents to index
            
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
            if not self.client.indices.exists(index=self.index_name):
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
            result = helpers.bulk(self.client, actions)
            logger.info(f"Bulk indexed {result[0]} documents")
            return True
        except Exception as e:
            logger.error(f"Failed to bulk index documents: {str(e)}")
            return False
    
    def update_document(self, doc_id: str, document: Dict[str, Any]) -> bool:
        """
        Update an existing document in Elasticsearch.
        
        Args:
            doc_id: The document ID to update
            document: The document data to update
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_connected():
            logger.error("Cannot update document: not connected to Elasticsearch")
            return False
            
        try:
            self.client.update(index=self.index_name, id=doc_id, doc=document)
            logger.info(f"Updated document with ID: {doc_id}")
            return True
        except NotFoundError:
            # Document doesn't exist, index it instead
            logger.info(f"Document with ID {doc_id} doesn't exist, indexing instead")
            return self.index_document(document)
        except Exception as e:
            logger.error(f"Failed to update document: {str(e)}")
            return False
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from Elasticsearch.
        
        Args:
            doc_id: The document ID to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_connected():
            logger.error("Cannot delete document: not connected to Elasticsearch")
            return False
            
        try:
            self.client.delete(index=self.index_name, id=doc_id)
            logger.info(f"Deleted document with ID: {doc_id}")
            return True
        except NotFoundError:
            logger.warning(f"Document with ID {doc_id} not found")
            return False
        except Exception as e:
            logger.error(f"Failed to delete document: {str(e)}")
            return False
    
    def search(self, query: str, limit: int = 20, min_similarity: float = 0.1) -> List[Dict[str, Any]]:
        """
        Search for images using Elasticsearch's full-text and fuzzy matching capabilities.
        
        Args:
            query: The search query string
            limit: Maximum number of results to return
            min_similarity: Minimum similarity score threshold
            
        Returns:
            List of matching documents with similarity scores
        """
        if not self.is_connected():
            logger.error("Cannot search: not connected to Elasticsearch")
            return []
            
        # Parse the query into components
        query = query.lower().strip()
        
        # Split on commas first to get distinct search "phrases"
        query_phrases = [phrase.strip() for phrase in query.split(',')]
        
        # Common color names to help with color matching
        color_names = [
            "red", "blue", "green", "yellow", "orange", "purple", "pink", 
            "brown", "black", "white", "gray", "grey", "teal", "turquoise", 
            "gold", "silver", "bronze", "maroon", "navy", "olive", "mint",
            "cyan", "magenta", "lavender", "violet", "indigo", "coral", "peach"
        ]
        
        try:
            # Build the search query
            should_clauses = []
            
            # Add query for each phrase
            for phrase in query_phrases:
                # Add match clauses for pattern fields
                should_clauses.extend([
                    {
                        "match": {
                            "patterns.primary_pattern": {
                                "query": phrase,
                                "fuzziness": "AUTO",
                                "boost": 2.0
                            }
                        }
                    },
                    {
                        "match": {
                            "patterns.prompt.final_prompt": {
                                "query": phrase,
                                "fuzziness": "AUTO"
                            }
                        }
                    },
                    {
                        "match": {
                            "patterns.style_keywords": {
                                "query": phrase,
                                "fuzziness": "AUTO"
                            }
                        }
                    }
                ])
                
                # Add nested query for secondary patterns
                should_clauses.append({
                    "nested": {
                        "path": "patterns.secondary_patterns",
                        "query": {
                            "match": {
                                "patterns.secondary_patterns.name": {
                                    "query": phrase,
                                    "fuzziness": "AUTO"
                                }
                            }
                        }
                    }
                })
                
                # Add nested query for elements
                should_clauses.append({
                    "nested": {
                        "path": "patterns.elements",
                        "query": {
                            "match": {
                                "patterns.elements.name": {
                                    "query": phrase,
                                    "fuzziness": "AUTO"
                                }
                            }
                        }
                    }
                })
                
                # If the phrase contains a color name, add a nested query for dominant colors
                if any(color in phrase for color in color_names):
                    should_clauses.append({
                        "nested": {
                            "path": "colors.dominant_colors",
                            "query": {
                                "match": {
                                    "colors.dominant_colors.name": {
                                        "query": phrase,
                                        "fuzziness": "AUTO",
                                        "boost": 1.5
                                    }
                                }
                            }
                        }
                    })
                
                # Match filename
                should_clauses.append({
                    "match": {
                        "filename": {
                            "query": phrase,
                            "boost": 0.5
                        }
                    }
                })
            
            # Combine all clauses in a bool query
            query_body = {
                "query": {
                    "bool": {
                        "should": should_clauses,
                        "minimum_should_match": 1
                    }
                },
                "size": limit
            }
            
            # Execute the search
            start_time = time.time()
            response = self.client.search(index=self.index_name, body=query_body)
            search_time = time.time() - start_time
            
            # Extract and format results
            results = []
            for hit in response["hits"]["hits"]:
                doc = hit["_source"]
                doc["similarity"] = hit["_score"] / response["hits"]["max_score"]  # Normalize scores
                
                # Only include results above min_similarity threshold
                if doc["similarity"] >= min_similarity:
                    results.append(doc)
            
            logger.info(f"Search for '{query}' found {len(results)} results in {search_time:.2f}s")
            return results
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return [] 