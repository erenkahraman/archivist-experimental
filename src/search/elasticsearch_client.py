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
            self.client.update(index=self.index_name, id=doc_id, doc=document)
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
            self.client.delete(index=self.index_name, id=doc_id)
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
        Parse a query string into components for advanced search.
        
        Args:
            query_string: The raw search query string
            
        Returns:
            Dict containing parsed components: 
            {'colors': [...], 'quoted_phrases': [...], 'potential_phrases': [...], 'keywords': [...], 'original': query_string}
        """
        # Normalize the query
        query_string = query_string.lower().strip()
        
        # Extract phrases in quotes
        import re
        quoted_phrases = re.findall(r'"([^"]*)"', query_string)
        
        # If multi-word query and not quoted, add the entire query as a quoted phrase
        normalized_query = query_string
        for phrase in quoted_phrases:
            normalized_query = normalized_query.replace(f'"{phrase}"', '')
        
        words = [w.strip() for w in re.split(r'[,\s]+', normalized_query) if w.strip()]
        if len(words) > 1 and not quoted_phrases:
            full_query_phrase = normalized_query.strip()
            if full_query_phrase:
                quoted_phrases.append(full_query_phrase)
        
        # Remove phrases from query for further processing
        query_for_processing = query_string
        for phrase in quoted_phrases:
            query_for_processing = query_for_processing.replace(f'"{phrase}"', '').replace(phrase, '')
        
        # Split remaining terms by spaces and commas
        remainder = [term.strip() for term in re.split(r'[,\s]+', query_for_processing) if term.strip()]
        
        # Color detection
        colors = []
        keywords = []
        
        # Common color names from our color reference database
        basic_colors = [
            "red", "blue", "green", "yellow", "orange", "purple", "pink", 
            "brown", "black", "white", "gray", "grey"
        ]
        
        specific_colors = [
            "teal", "turquoise", "maroon", "navy", "olive", "mint", "cyan", 
            "magenta", "lavender", "violet", "indigo", "coral", "peach",
            "crimson", "azure", "beige", "tan", "gold", "silver", "bronze",
            "burgundy", "scarlet", "vermilion", "ruby", "cherry", "cardinal",
            "fuchsia", "salmon", "amber", "khaki", "mustard", "lemon",
            "lime", "forest", "emerald", "sage", "chartreuse", "avocado",
            "aqua", "sky", "cobalt", "cerulean", "sapphire", "plum", "mauve",
            "amethyst", "periwinkle", "chocolate", "sienna", "camel", "taupe",
            "charcoal", "slate", "ivory", "cream"
        ]
        
        all_colors = basic_colors + specific_colors
        
        # Check each term
        for term in remainder:
            # Check if it's a color term
            if term in all_colors:
                colors.append(term)
            # Check for compound color terms
            elif any(color in term for color in all_colors):
                for color in all_colors:
                    if color in term:
                        colors.append(term)
                        break
            else:
                keywords.append(term)
        
        # Check quoted phrases for any color mentions
        for phrase in quoted_phrases:
            has_color = False
            for color in all_colors:
                if color in phrase.split():
                    has_color = True
                    colors.append(phrase)
                    break
            if not has_color:
                keywords.append(phrase)
        
        # Identify potential multi-word phrases from any remaining query
        potential_phrases = []
        # Only find potential phrases if they're not already covered by quoted phrases
        if query_for_processing.strip():
            # Find sequences of 2-3 consecutive words that aren't colors
            words = [w.strip() for w in re.split(r'[,\s]+', query_for_processing) if w.strip()]
            for i in range(len(words) - 1):
                # Check for two consecutive words that aren't colors
                if words[i] not in all_colors and words[i+1] not in all_colors:
                    two_word_phrase = f"{words[i]} {words[i+1]}"
                    potential_phrases.append(two_word_phrase)
                    
                    # Check for three consecutive words
                    if i < len(words) - 2 and words[i+2] not in all_colors:
                        three_word_phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
                        potential_phrases.append(three_word_phrase)
        
        # Clean up any duplicates
        colors = list(set(colors))
        keywords = list(set(keywords))
        potential_phrases = list(set(potential_phrases))
        
        # Log the parsing results
        logger.info(f"Parsed query '{query_string}' into components:")
        logger.info(f" - Colors: {colors}")
        logger.info(f" - Quoted phrases: {quoted_phrases}")
        logger.info(f" - Potential phrases: {potential_phrases}")
        logger.info(f" - Keywords: {keywords}")
        
        return {
            'colors': colors,
            'quoted_phrases': quoted_phrases,
            'potential_phrases': potential_phrases,
            'keywords': keywords,
            'original': query_string
        }
    
    def search(self, query: str, limit: int = 20, min_similarity: float = 0.1) -> List[Dict[str, Any]]:
        """
        Advanced search for images using Elasticsearch with a single, structured bool query.
        
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
            
        # Log search initialization with parameters
        logger.info("===== SEARCH PROCESS START =====")
        logger.info(f"Search parameters: query='{query}', limit={limit}, min_similarity={min_similarity}")
            
        # Parse the query into components using our helper function
        parsed_query = self._parse_query(query)
        
        try:
            # Start building the search query
            search_start_time = time.time()
            
            # Initialize query components
            bool_query = {
                "bool": {
                    "should": [],
                    "must": [],
                    "filter": []
                }
            }
            
            # Check if this is a multi-word query (use stricter matching)
            original_query = parsed_query['original'].lower().strip()
            words = original_query.split()
            is_multi_word_query = len(words) > 1 and not all(word in all_colors for word in words)
            
            # If this is a multi-word query, enforce exact phrase matching using must
            if is_multi_word_query:
                logger.info(f"Multi-word query detected: '{original_query}'. Enforcing strict exact phrase matching.")
                
                # Create a strict matching clause that requires the exact phrase in must
                exact_phrase_must = {
                    "bool": {
                        "should": [
                            {"match_phrase": {"patterns.main_theme": original_query}},
                            {"match_phrase": {"patterns.content_details.name": original_query}},
                            {"match_phrase": {"patterns.elements.name": original_query}},
                            {"match_phrase": {"patterns.primary_pattern": original_query}},
                            {"match_phrase": {"patterns.prompt.final_prompt": original_query}},
                            {"nested": {
                                "path": "patterns.secondary_patterns",
                                "query": {
                                    "match_phrase": {
                                        "patterns.secondary_patterns.name": original_query
                                    }
                                }
                            }},
                            {"nested": {
                                "path": "patterns.stylistic_attributes",
                                "query": {
                                    "match_phrase": {
                                        "patterns.stylistic_attributes.name": original_query
                                    }
                                }
                            }}
                        ],
                        "minimum_should_match": 1  # At least one field must have the exact phrase
                    }
                }
                
                # Add to must clause to enforce this constraint - this is stricter than filter
                bool_query["bool"]["must"].append(exact_phrase_must)
                
                # Remove all other clauses for multi-word queries to ensure only exact matches are found
                # Only keep exact phrase matching in must clause and color queries if any
                bool_query["bool"]["should"] = []
                
                # Replace previous filter-based approach with must-based approach
                # Clear any existing filter clauses for multi-word queries
                bool_query["bool"]["filter"] = [filter_clause for filter_clause in bool_query["bool"]["filter"] 
                                               if not (isinstance(filter_clause, dict) and 
                                                     "bool" in filter_clause and 
                                                     "should" in filter_clause["bool"] and
                                                     any("match_phrase" in clause for clause in filter_clause["bool"]["should"]))]
            
            # 1. Add highly boosted match_phrase queries for both quoted and potential phrases - but SKIP if multi-word query
            # Since multi-word queries now handle phrase matching in the must clause
            if all_phrases and not is_multi_word_query:
                for phrase in all_phrases:
                    is_quoted = phrase in parsed_query['quoted_phrases']
                    is_full_query = phrase == parsed_query['original'].lower().strip()
                    
                    # Highest boost for the full query phrase, high for quoted, medium for potential
                    if is_full_query:
                        boost_factor = 25  # Extremely high boost for exact full query match
                    elif is_quoted:
                        boost_factor = 15
                    else:
                        boost_factor = 10
                        
                    phrase_type = "full_query" if is_full_query else ("quoted" if is_quoted else "potential")
                    logger.info(f"Adding {phrase_type} phrase matching for: '{phrase}'")
                    
                    # Add match_phrase queries for the phrase
                    bool_query["bool"]["should"].extend([
                        {
                            "match_phrase": {
                                "patterns.main_theme": {
                                    "query": phrase,
                                    "boost": boost_factor
                                }
                            }
                        },
                        {
                            "match_phrase": {
                                "patterns.content_details.name": {
                                    "query": phrase,
                                    "boost": boost_factor - 1
                                }
                            }
                        },
                        {
                            "match_phrase": {
                                "patterns.elements.name": {
                                    "query": phrase,
                                    "boost": boost_factor - 2
                                }
                            }
                        },
                        {
                            "match_phrase": {
                                "patterns.prompt.final_prompt": {
                                    "query": phrase,
                                    "boost": boost_factor - 3
                                }
                            }
                        },
                        {
                            "match_phrase": {
                                "patterns.primary_pattern": {
                                    "query": phrase,
                                    "boost": boost_factor - 2
                                }
                            }
                        },
                        {
                            "nested": {
                                "path": "patterns.secondary_patterns",
                                "query": {
                                    "match_phrase": {
                                        "patterns.secondary_patterns.name": {
                                            "query": phrase,
                                            "boost": boost_factor - 4
                                        }
                                    }
                                },
                                "score_mode": "max"
                            }
                        },
                        {
                            "nested": {
                                "path": "patterns.stylistic_attributes",
                                "query": {
                                    "match_phrase": {
                                        "patterns.stylistic_attributes.name": {
                                            "query": phrase,
                                            "boost": boost_factor - 4
                                        }
                                    }
                                },
                                "score_mode": "max"
                            }
                        }
                    ])
            
            # Skip adding individual term/keyword matches if this is a multi-word query
            if not is_multi_word_query:
                # 2. Add exact keyword matches targeting .raw fields
                all_terms = parsed_query['keywords'] + parsed_query['potential_phrases']
                if all_terms:
                    for term in all_terms:
                        logger.info(f"Adding exact term matching for: '{term}'")
                        
                        # Add exact matches using .raw fields with high boost
                        bool_query["bool"]["should"].extend([
                            {
                                "term": {
                                    "patterns.main_theme.raw": {
                                        "value": term,
                                        "boost": 8
                                    }
                                }
                            },
                            {
                                "term": {
                                    "patterns.content_details.name.raw": {
                                        "value": term,
                                        "boost": 7
                                    }
                                }
                            },
                            {
                                "term": {
                                    "patterns.elements.name.raw": {
                                        "value": term,
                                        "boost": 7
                                    }
                                }
                            },
                            {
                                "term": {
                                    "patterns.primary_pattern.raw": {
                                        "value": term,
                                        "boost": 7
                                    }
                                }
                            },
                            {
                                "nested": {
                                    "path": "patterns.secondary_patterns",
                                    "query": {
                                        "term": {
                                            "patterns.secondary_patterns.name.raw": {
                                                "value": term,
                                                "boost": 6
                                            }
                                        }
                                    },
                                    "score_mode": "max"
                                }
                            },
                            {
                                "nested": {
                                    "path": "patterns.stylistic_attributes",
                                    "query": {
                                        "term": {
                                            "patterns.stylistic_attributes.name.raw": {
                                                "value": term,
                                                "boost": 6
                                            }
                                        }
                                    },
                                    "score_mode": "max"
                                }
                            },
                            {
                                "term": {
                                    "patterns.style_keywords.raw": {
                                        "value": term,
                                        "boost": 5
                                    }
                                }
                            }
                        ])
                
                # 3. Add multi_match for single keywords (for recall via synonyms)
                if parsed_query['keywords']:
                    # Join single keywords for the multi_match query
                    keywords_text = " ".join(parsed_query['keywords'])
                    
                    if keywords_text:
                        logger.info(f"Adding multi_match for keywords: '{keywords_text}'")
                        bool_query["bool"]["should"].append({
                            "multi_match": {
                                "query": keywords_text,
                                "fields": [
                                    "patterns.main_theme^4",
                                    "patterns.primary_pattern^3.5",
                                    "patterns.content_details.name^3",
                                    "patterns.stylistic_attributes.name^3",
                                    "patterns.elements.name^2",
                                    "patterns.prompt.final_prompt^1.5",
                                    "patterns.style_keywords^1",
                                    "patterns.secondary_patterns.name^0.8"
                                ],
                                "type": "cross_fields",
                                "operator": "or",
                                "fuzziness": "AUTO",
                                "prefix_length": 2,
                                "boost": 4
                            }
                        })
            
            # 4. Add color matching with refined boosting
            if parsed_query['colors']:
                for color_term in parsed_query['colors']:
                    logger.info(f"Adding color matching for: '{color_term}'")
                    
                    # Add color clauses to should with .raw field having higher boost
                    bool_query["bool"]["should"].extend([
                        {
                            "nested": {
                                "path": "colors.dominant_colors",
                                "query": {
                                    "match": {
                                        "colors.dominant_colors.name.raw": {
                                            "query": color_term,
                                            "boost": 6
                                        }
                                    }
                                },
                                "score_mode": "sum"
                            }
                        },
                        {
                            "nested": {
                                "path": "colors.dominant_colors",
                                "query": {
                                    "match": {
                                        "colors.dominant_colors.name": {
                                            "query": color_term,
                                            "boost": 5
                                        }
                                    }
                                },
                                "score_mode": "sum"
                            }
                        }
                    ])
                
                # If query has ONLY color terms, add a filter to ensure at least one color matches
                if not parsed_query['keywords'] and not parsed_query['quoted_phrases'] and not parsed_query['potential_phrases'] and len(parsed_query['colors']) == 1:
                    # Only one color specified - add a filter to require this color
                    color_term = parsed_query['colors'][0]
                    logger.info(f"Adding color filter for '{color_term}' since it's the only search term")
                    bool_query["bool"]["filter"].append({
                        "nested": {
                            "path": "colors.dominant_colors",
                            "query": {
                                "bool": {
                                    "should": [
                                        {"match": {"colors.dominant_colors.name": color_term}},
                                        {"match": {"colors.dominant_colors.name.raw": color_term}}
                                    ]
                                }
                            }
                        }
                    })
            
            # 5. Calculate appropriate minimum_should_match
            # Count distinct concepts (treat each potential phrase as one concept)
            concept_count = len(parsed_query['keywords']) + len(parsed_query['quoted_phrases']) + len(parsed_query['potential_phrases']) + len(parsed_query['colors'])
            
            # Calculate minimum_should_match based on concept count
            if concept_count > 3:
                # For complex queries, require matching at least 60% of concepts
                min_should = max(1, int(concept_count * 0.6))
            elif concept_count > 1:
                # For simple multi-term queries, require at least 1 term
                min_should = 1
            else:
                # For single term queries, require that term
                min_should = 1
            
            logger.info(f"Setting minimum_should_match to {min_should}")
            bool_query["bool"]["minimum_should_match"] = min_should
            
            # Wrap the bool query in a function_score query for advanced scoring
            query_body = {
                "query": {
                    "function_score": {
                        "query": bool_query,
                        "functions": [
                            # Boost by confidence scores
                            {
                                "field_value_factor": {
                                    "field": "patterns.main_theme_confidence",
                                    "factor": 1.2,
                                    "modifier": "ln1p",
                                    "missing": 0.5
                                },
                                "weight": 1.5
                            },
                            {
                                "field_value_factor": {
                                    "field": "patterns.pattern_confidence",
                                    "factor": 1.0,
                                    "modifier": "ln1p",
                                    "missing": 0.5
                                },
                                "weight": 1.0
                            },
                            # Recency boost - favor newer content
                            {
                                "gauss": {
                                    "timestamp": {
                                        "scale": "30d",
                                        "offset": "7d",
                                        "decay": 0.5
                                    }
                                },
                                "weight": 0.5
                            }
                        ],
                        "score_mode": "sum",
                        "boost_mode": "multiply"
                    }
                },
                "size": limit,
                "min_score": 0.1  # Base threshold to avoid very low relevance results
            }
            
            # If the query has color terms, add a color proportion boost
            if parsed_query['colors']:
                # Add script score to boost by color proportion when color matches
                color_script = {
                    "script_score": {
                        "script": {
                            "source": """
                                double score = 1.0;
                                if (doc.containsKey('colors.dominant_colors') && 
                                    !doc['colors.dominant_colors.empty'].value) {
                                    for (int i = 0; i < doc['colors.dominant_colors.name'].length; ++i) {
                                        String color = doc['colors.dominant_colors.name'][i].toLowerCase();
                                        if (params.colors.contains(color)) {
                                            score += doc['colors.dominant_colors.proportion'][i] * 2.0;
                                        }
                                    }
                                }
                                return score;
                            """,
                            "params": {
                                "colors": parsed_query['colors']
                            }
                        }
                    },
                    "weight": 1.5
                }
                query_body["query"]["function_score"]["functions"].append(color_script)
            
            # Log the complete query
            self._log_query_details(query_body)
            
            # Execute the search
            logger.info("Executing Elasticsearch search...")
            start_time = time.time()
            response = self.client.search(index=self.index_name, body=query_body)
            search_time = time.time() - start_time
            logger.info(f"Elasticsearch query executed in {search_time:.2f}s")
            
            # Extract and format results
            results = []
            total_hits = response["hits"]["total"]["value"] if "total" in response["hits"] else len(response["hits"]["hits"])
            logger.info(f"Search returned {total_hits} total hits")
            
            for i, hit in enumerate(response["hits"]["hits"]):
                doc = hit["_source"]
                # Instead of normalizing by max_score, we use the raw score
                doc["similarity"] = hit["_score"]
                
                # Log each result with its score and whether it meets the threshold
                meets_threshold = doc["similarity"] >= min_similarity
                logger.info(f"Result {i+1}: id={doc.get('id', 'unknown')}, filename={doc.get('filename', 'unknown')}, score={hit['_score']:.4f}, meets_threshold={meets_threshold}")
                
                # Only include results above min_similarity threshold
                if meets_threshold:
                    results.append(doc)
                else:
                    logger.info(f"  Skipping result {i+1} as similarity {doc['similarity']:.4f} is below threshold {min_similarity}")
            
            # Log search performance
            query_time = time.time() - search_start_time
            logger.info(f"Total search process for '{query}' completed in {query_time:.2f}s")
            logger.info(f"Found {total_hits} hits, returning {len(results)} results after filtering by min_similarity={min_similarity}")
            logger.info("===== SEARCH PROCESS END =====")
            
            return results
        except Exception as e:
            logger.error(f"Search error: {str(e)}", exc_info=True)
            logger.info("===== SEARCH PROCESS FAILED =====")
            return []
    
    def _log_query_details(self, query_body: Dict[str, Any]) -> None:
        """
        Log the details of an Elasticsearch query, handling large query bodies appropriately.
        
        Args:
            query_body: The Elasticsearch query body to log
        """
        import json
        
        try:
            # Convert query to a formatted JSON string
            query_json = json.dumps(query_body, indent=2)
            
            # Check if query is too large for a single log entry
            if len(query_json) > 5000:
                logger.info(f"Query body is large ({len(query_json)} chars). Logging summary:")
                
                # Log query structure overview
                query_type = query_body.get("query", {})
                if "bool" in query_type:
                    should_clauses = query_type["bool"].get("should", [])
                    logger.info(f"Bool query with {len(should_clauses)} should clauses")
                    
                    # Count clause types
                    clause_types = {}
                    for clause in should_clauses:
                        for key in clause:
                            if key not in clause_types:
                                clause_types[key] = 0
                            clause_types[key] += 1
                    
                    # Log clause type counts
                    for clause_type, count in clause_types.items():
                        logger.info(f"  - {clause_type}: {count} clauses")
                
                # Log size and other top-level parameters
                for key, value in query_body.items():
                    if key != "query":
                        logger.info(f"  - {key}: {value}")
            else:
                # Log the entire query if it's reasonably sized
                logger.info(f"Full query body:\n{query_json}")
        except Exception as e:
            logger.error(f"Error while logging query details: {str(e)}")
            logger.info("Unable to log complete query details")
    
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
            if not self.client.indices.exists(index=self.index_name):
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
            count_request = self.client.count(index=self.index_name)
            total_docs = count_request["count"]
            
            if total_docs == 0:
                logger.info("No documents to reindex. Deleting old index and creating new one.")
                self.client.indices.delete(index=self.index_name)
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
            self.client.reindex(body=reindex_body, wait_for_completion=True)
            
            # 4. Verify new index has all documents
            new_count_request = self.client.count(index=temp_index)
            new_total_docs = new_count_request["count"]
            
            if new_total_docs != total_docs:
                logger.error(f"Document count mismatch after reindexing: {total_docs} vs {new_total_docs}")
                # Clean up temp index
                self.client.indices.delete(index=temp_index)
                return False
                
            # 5. Delete the old index
            logger.info(f"Reindexing complete. Deleting old index '{self.index_name}'")
            self.client.indices.delete(index=self.index_name)
            
            # 6. Create an alias from the old name to the new index
            logger.info(f"Creating alias from '{self.index_name}' to '{temp_index}'")
            self.client.indices.update_aliases(body={
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