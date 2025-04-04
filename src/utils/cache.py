import redis
import json
import hashlib
import logging
from typing import Any, Dict, List, Optional
import config

# Configure logger
logger = logging.getLogger(__name__)

class SearchCache:
    """Cache for search queries using Redis"""
    
    def __init__(self):
        """Initialize the Redis cache connection"""
        self.enabled = config.ENABLE_CACHE
        self.ttl = config.CACHE_TTL
        self.redis_client = None
        
        if self.enabled:
            try:
                self.redis_client = redis.Redis(
                    host='localhost',
                    port=6379,
                    db=0,
                    decode_responses=True
                )
                self.redis_client.ping()  # Test connection
                logger.info("Connected to Redis cache")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {str(e)}")
                self.enabled = False
    
    def _generate_key(self, query: str, limit: int, min_similarity: float) -> str:
        """
        Generate a unique key for the search query.
        
        Args:
            query: The search query string
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold
            
        Returns:
            str: Unique key for the query
        """
        # Create a string representing the query parameters
        query_str = f"{query}|{limit}|{min_similarity}"
        
        # Generate a hash of the query string
        key = hashlib.md5(query_str.encode()).hexdigest()
        return f"search:{key}"
    
    def get(self, query: str, limit: int, min_similarity: float) -> Optional[List[Dict[str, Any]]]:
        """
        Get search results from cache.
        
        Args:
            query: The search query string
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of search results or None if not in cache
        """
        if not self.enabled or not self.redis_client:
            return None
            
        try:
            key = self._generate_key(query, limit, min_similarity)
            cached_data = self.redis_client.get(key)
            
            if cached_data:
                logger.info(f"Cache hit for query: '{query}'")
                return json.loads(cached_data)
            else:
                logger.info(f"Cache miss for query: '{query}'")
                return None
        except Exception as e:
            logger.error(f"Error retrieving from cache: {str(e)}")
            return None
    
    def set(self, query: str, limit: int, min_similarity: float, results: List[Dict[str, Any]]) -> bool:
        """
        Store search results in cache.
        
        Args:
            query: The search query string
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold
            results: Search results to cache
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False
            
        try:
            key = self._generate_key(query, limit, min_similarity)
            
            # Convert results to JSON string
            results_json = json.dumps(results)
            
            # Store in Redis with TTL
            self.redis_client.setex(key, self.ttl, results_json)
            logger.info(f"Cached results for query: '{query}'")
            return True
        except Exception as e:
            logger.error(f"Error storing in cache: {str(e)}")
            return False
    
    def invalidate_all(self) -> bool:
        """
        Invalidate all cached search results.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False
            
        try:
            # Delete all keys matching the search pattern
            keys = self.redis_client.keys("search:*")
            if keys:
                self.redis_client.delete(*keys)
                logger.info(f"Invalidated {len(keys)} cached search results")
            return True
        except Exception as e:
            logger.error(f"Error invalidating cache: {str(e)}")
            return False 