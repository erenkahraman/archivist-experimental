import logging
import time
import functools
from typing import Dict, Any, Optional, Callable
from collections import OrderedDict

logger = logging.getLogger(__name__)

class CacheItem:
    """Represents an item in the cache with TTL support."""
    
    def __init__(self, key: str, value: Any, ttl: Optional[int] = None):
        self.key = key
        self.value = value
        self.ttl = ttl  # Time to live in seconds
        self.created_at = time.time()
        self.last_accessed = time.time()
    
    def is_expired(self) -> bool:
        """Check if the item has expired based on its TTL."""
        if self.ttl is None:
            return False
        return time.time() > (self.created_at + self.ttl)
    
    def access(self):
        """Update the last accessed time when item is accessed."""
        self.last_accessed = time.time()


class MemoryCache:
    """In-memory LRU cache with TTL support."""
    
    def __init__(self, max_size: int = 1000, default_ttl: Optional[int] = None):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheItem] = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get an item from the cache, return default if not found or expired."""
        if key not in self._cache:
            self.misses += 1
            return default
        
        item = self._cache[key]
        
        if item.is_expired():
            self.delete(key)
            self.misses += 1
            return default
        
        # Move to end (most recently used)
        self._cache.move_to_end(key)
        item.access()
        self.hits += 1
        return item.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in the cache with optional TTL."""
        if ttl is None:
            ttl = self.default_ttl
        
        # Check if key exists and update
        if key in self._cache:
            self.delete(key)
        
        # Check if cache is full
        if len(self._cache) >= self.max_size:
            # Remove least recently used item
            self._cache.popitem(last=False)
        
        # Add new item
        self._cache[key] = CacheItem(key, value, ttl)
        # Move to end (most recently used)
        self._cache.move_to_end(key)
    
    def delete(self, key: str) -> bool:
        """Delete an item from the cache. Returns True if item was deleted."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all items from the cache."""
        self._cache.clear()
        logger.info("Cache cleared")
    
    def invalidate(self, key_prefix: str) -> int:
        """Invalidate all keys starting with key_prefix."""
        keys_to_delete = [k for k in self._cache.keys() if k.startswith(key_prefix)]
        count = 0
        for key in keys_to_delete:
            if self.delete(key):
                count += 1
        
        if count > 0:
            logger.info(f"Invalidated {count} cache entries with prefix '{key_prefix}'")
        
        return count
    
    def prune_expired(self) -> int:
        """Remove expired items from the cache."""
        keys_to_delete = [k for k, v in self._cache.items() if v.is_expired()]
        count = 0
        for key in keys_to_delete:
            if self.delete(key):
                count += 1
        
        if count > 0:
            logger.info(f"Pruned {count} expired items from cache")
        
        return count
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        }

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


def cached(function=None, ttl=None, key_prefix=None):
    """
    Decorator to cache function results.
    
    Args:
        function: The function to decorate
        ttl: Time to live in seconds
        key_prefix: Prefix for the cache key
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_prefix:
                prefix = key_prefix
            else:
                prefix = f"{func.__module__}.{func.__name__}"
            
            # Create key from arguments
            args_str = ','.join([str(arg) for arg in args])
            kwargs_str = ','.join([f"{k}={v}" for k, v in sorted(kwargs.items())])
            key = f"{prefix}({args_str},{kwargs_str})"
            
            # Check cache
            cache = SearchCache()  # Get singleton cache instance
            result = cache.get(key)
            
            if result is not None:
                logger.debug(f"Cache hit for {key}")
                return result
            
            # Cache miss, call function
            logger.debug(f"Cache miss for {key}")
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.set(key, result, ttl)
            return result
        
        return wrapper
    
    if function:
        return decorator(function)
    return decorator


class SearchCache(MemoryCache):
    """Cache specifically for search results."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SearchCache, cls).__new__(cls)
            # Initialize the instance
            cls._instance.__initialized = False
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '__initialized'):
            super().__init__(max_size=500, default_ttl=1800)  # 30 minutes default TTL
            self.__initialized = True
    
    def invalidate_all(self):
        """Invalidate all entries in the cache."""
        self.clear()