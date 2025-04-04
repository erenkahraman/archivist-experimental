import logging
import pickle
import time
from functools import wraps
import config

logger = logging.getLogger(__name__)

class SearchCache:
    """Simple caching system for search results."""
    
    def __init__(self):
        """Initialize the cache system."""
        self.enabled = config.ENABLE_CACHE
        self.redis = None
        
        if self.enabled:
            try:
                import redis
                self.redis = redis.from_url(config.REDIS_URL)
                logger.info(f"Connected to Redis cache at {config.REDIS_URL}")
            except ImportError:
                logger.warning("Redis package not installed. In-memory cache will be used instead.")
                self.enabled = False
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}. In-memory cache will be used instead.")
                self.enabled = False
                
        # Fallback to in-memory cache if Redis is not available
        if not self.enabled:
            self.memory_cache = {}
            logger.info("Using in-memory cache")

    def get(self, key):
        """Get a value from the cache."""
        if not self.enabled:
            return self.memory_cache.get(key)
            
        try:
            if self.redis:
                data = self.redis.get(key)
                if data:
                    return pickle.loads(data)
        except Exception as e:
            logger.warning(f"Error retrieving from cache: {e}")
            
        return None

    def set(self, key, value, ttl=None):
        """Set a value in the cache."""
        if not self.enabled:
            self.memory_cache[key] = value
            return True
            
        try:
            if self.redis:
                ttl = ttl or config.CACHE_TTL
                return self.redis.setex(key, ttl, pickle.dumps(value))
        except Exception as e:
            logger.warning(f"Error setting cache: {e}")
            
        return False

    def delete(self, key):
        """Delete a value from the cache."""
        if not self.enabled:
            if key in self.memory_cache:
                del self.memory_cache[key]
            return True
            
        try:
            if self.redis:
                return self.redis.delete(key) > 0
        except Exception as e:
            logger.warning(f"Error deleting from cache: {e}")
            
        return False

    def flush(self):
        """Flush the entire cache."""
        if not self.enabled:
            self.memory_cache = {}
            return True
            
        try:
            if self.redis:
                return self.redis.flushdb()
        except Exception as e:
            logger.warning(f"Error flushing cache: {e}")
            
        return False

    def invalidate_for_key_prefix(self, prefix: str) -> None:
        """
        Invalidate all cache entries that start with the given prefix
        
        Args:
            prefix: The prefix to match against cache keys
        """
        if not self.enabled:
            return
        
        try:
            # For Redis cache
            if self.redis:
                # Get all keys with the prefix
                keys = self.redis.keys(f"{prefix}*")
                if keys:
                    # Delete all matching keys
                    self.redis.delete(*keys)
                    logger.info(f"Invalidated {len(keys)} cache entries with prefix '{prefix}'")
            # For in-memory cache
            elif self.memory_cache:
                # Identify keys to delete
                keys_to_delete = [k for k in self.memory_cache.keys() if k.startswith(prefix)]
                # Delete matching keys
                for key in keys_to_delete:
                    self.memory_cache.pop(key, None)
                logger.info(f"Invalidated {len(keys_to_delete)} in-memory cache entries with prefix '{prefix}'")
        except Exception as e:
            logger.error(f"Error invalidating cache with prefix '{prefix}': {e}")


def cached(ttl=None):
    """Decorator to cache function results."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Skip caching if cache is disabled or it's a write operation
            if not config.ENABLE_CACHE or kwargs.get('skip_cache', False):
                return func(*args, **kwargs)
                
            # Generate a cache key
            key_parts = [func.__name__]
            key_parts.extend([str(arg) for arg in args])
            
            # Add sorted kwargs to ensure consistent keys
            for k, v in sorted(kwargs.items()):
                if k != 'skip_cache':  # Don't include skip_cache in the key
                    key_parts.append(f"{k}={v}")
                    
            cache_key = ":".join(key_parts)
            
            # Try to get from cache
            cache = SearchCache()
            cached_result = cache.get(cache_key)
            
            if cached_result is not None:
                return cached_result
                
            # Cache miss, compute the result
            result = func(*args, **kwargs)
            
            # Cache the result
            cache.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator 