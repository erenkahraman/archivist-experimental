from pathlib import Path
from typing import List, Dict, Any
import torch
from transformers import CLIPProcessor, CLIPModel
import json
from PIL import Image
import numpy as np
import logging
import time as import_time
import os

# Relative imports from the same package
from .analyzers.color_analyzer import ColorAnalyzer
from .analyzers.gemini_analyzer import GeminiAnalyzer
from .search.elasticsearch_client import ElasticsearchClient
from .utils.cache import SearchCache
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress less important logs
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.setLevel(logging.ERROR)

class SearchEngine:
    def __init__(self, gemini_api_key=None):
        logger.info("Initializing SearchEngine...")
        try:
            # Initialize analyzers with API key
            if gemini_api_key:
                # Mask the key for logging
                masked_key = self._mask_api_key(gemini_api_key)
                logger.info(f"Using Gemini API key: {masked_key}")
            
            # Initialize analyzers without storing the key in this class
            self.gemini_analyzer = GeminiAnalyzer(api_key=gemini_api_key)
            self.color_analyzer = ColorAnalyzer(max_clusters=config.N_CLUSTERS, api_key=gemini_api_key)
            
            # Initialize Elasticsearch client
            self.use_elasticsearch = self._init_elasticsearch()
            
            # Initialize Redis cache
            self.cache = SearchCache()
            
            # Load existing metadata if available
            self.metadata = self.load_metadata()
            
            # Initialize search analytics
            self.search_logs = self._load_search_logs()
            
            # If using Elasticsearch, ensure all metadata is indexed
            if self.use_elasticsearch and self.metadata:
                self._index_all_metadata()
            
            logger.info("SearchEngine initialization completed")
        except Exception as e:
            logger.error(f"Error initializing SearchEngine: {e}")
            raise

    def _init_elasticsearch(self) -> bool:
        """Initialize Elasticsearch client and check connection"""
        try:
            # Initialize Elasticsearch client
            self.es_client = ElasticsearchClient(
                hosts=config.ELASTICSEARCH_HOSTS,
                cloud_id=config.ELASTICSEARCH_CLOUD_ID,
                api_key=config.ELASTICSEARCH_API_KEY,
                username=config.ELASTICSEARCH_USERNAME,
                password=config.ELASTICSEARCH_PASSWORD
            )
            
            # Check if connected
            if self.es_client.is_connected():
                logger.info("Successfully connected to Elasticsearch")
                
                # Create index if it doesn't exist
                self.es_client.create_index()
                return True
            else:
                logger.warning("Failed to connect to Elasticsearch, using in-memory search instead")
                return False
        except Exception as e:
            logger.error(f"Error initializing Elasticsearch: {e}")
            return False

    def _index_all_metadata(self):
        """Index all metadata into Elasticsearch"""
        try:
            if not self.metadata:
                logger.info("No metadata to index")
                return
                
            # Collect all metadata as documents
            documents = list(self.metadata.values())
            
            # Bulk index documents
            result = self.es_client.bulk_index(documents)
            if result:
                logger.info(f"Successfully indexed {len(documents)} documents in Elasticsearch")
            else:
                logger.error("Failed to index documents in Elasticsearch")
        except Exception as e:
            logger.error(f"Error indexing metadata: {e}")

    def _mask_api_key(self, key):
        if not key or len(key) < 8:
            return "INVALID_KEY"
        # Show only first 4 and last 4 characters
        return f"{key[:4]}...{key[-4:]}"

    def load_metadata(self) -> Dict:
        """Load metadata from file."""
        try:
            metadata_path = config.BASE_DIR / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return {}

    def save_metadata(self):
        """Save metadata to file."""
        try:
            metadata_path = config.BASE_DIR / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")

    def set_gemini_api_key(self, api_key: str):
        """Set or update the Gemini API key"""
        if api_key:
            # Mask the key for logging
            masked_key = self._mask_api_key(api_key)
            logger.info(f"Updating Gemini API key: {masked_key}")
            
            # Update in analyzers without storing the key in this class
            self.gemini_analyzer.set_api_key(api_key)
            self.color_analyzer.set_api_key(api_key)
            logger.info("Gemini API key updated in SearchEngine")
        else:
            logger.warning("Attempted to set empty API key")

    def process_image(self, image_path: Path) -> Dict[str, Any]:
        """Process an image and extract metadata including patterns and colors."""
        try:
            logger.info(f"Processing image: {image_path}")
            
            # Check if file exists
            if not image_path.exists():
                logger.error(f"Image file not found: {image_path}")
                return None
                
            # Create thumbnail
            thumbnail_path = self.create_thumbnail(image_path)
            if not thumbnail_path:
                logger.error(f"Failed to create thumbnail for: {image_path}")
                return None
                
            # Get relative paths for storage
            rel_image_path = image_path.relative_to(config.UPLOAD_DIR)
            rel_thumbnail_path = thumbnail_path.relative_to(config.THUMBNAIL_DIR)
            
            # Open the image for analysis - use a lower resolution for analysis
            image = Image.open(image_path).convert('RGB')
            
            # Resize for faster processing if image is large
            width, height = image.size
            target_pixels = 100_000  # Target pixel count for analysis
            if width * height > target_pixels:
                ratio = (target_pixels / (width * height)) ** 0.5
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                image = image.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert to numpy array for color analysis
            image_np = np.array(image)
            
            # First perform color analysis since it can be done locally
            color_info = self.analyze_colors(image_np)
            
            # Then use Gemini for pattern analysis
            pattern_info = self.gemini_analyzer.analyze_image(str(image_path))
            
            # Ensure pattern_info has the required fields
            if pattern_info.get('primary_pattern') is None:
                pattern_info['primary_pattern'] = pattern_info.get('category', 'Unknown')
            
            if pattern_info.get('pattern_confidence') is None:
                pattern_info['pattern_confidence'] = pattern_info.get('category_confidence', 0.8)
                
            # Add new structured fields if they're missing
            if pattern_info.get('main_theme') is None:
                pattern_info['main_theme'] = pattern_info.get('primary_pattern', pattern_info.get('category', 'Unknown'))
                
            if pattern_info.get('main_theme_confidence') is None:
                pattern_info['main_theme_confidence'] = pattern_info.get('pattern_confidence', pattern_info.get('category_confidence', 0.8))
                
            # Ensure content_details exists
            if not pattern_info.get('content_details'):
                # Create from elements if available
                pattern_info['content_details'] = []
                for element in pattern_info.get('elements', []):
                    if isinstance(element, dict) and element.get('name'):
                        pattern_info['content_details'].append({
                            'name': element.get('name', ''),
                            'confidence': element.get('confidence', 0.8)
                        })
                        
            # Ensure stylistic_attributes exists
            if not pattern_info.get('stylistic_attributes'):
                # Create from style_keywords if available
                pattern_info['stylistic_attributes'] = []
                for keyword in pattern_info.get('style_keywords', []):
                    pattern_info['stylistic_attributes'].append({
                        'name': keyword,
                        'confidence': 0.8
                    })
            
            # Generate metadata
            metadata = {
                'id': str(image_path.stem),
                'filename': image_path.name,
                'path': str(rel_image_path),
                'thumbnail_path': str(rel_thumbnail_path),
                'patterns': pattern_info,
                'colors': color_info,
                'timestamp': import_time.time()
            }
            
            # Store metadata in memory
            self.metadata[str(rel_image_path)] = metadata
            self.save_metadata()
            
            # Index in Elasticsearch if enabled
            if self.use_elasticsearch:
                indexing_successful = self.es_client.index_document(metadata)
                if indexing_successful:
                    logger.info(f"Successfully indexed {image_path.name} in Elasticsearch")
                else:
                    logger.warning(f"Failed to index {image_path.name} in Elasticsearch")
            
            logger.info(f"Image processed successfully: {image_path}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error processing image: {e}", exc_info=True)
            return None

    def create_thumbnail(self, image_path: Path) -> Path:
        """Create a thumbnail for an image."""
        try:
            # Open the image
            image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to thumbnail size
            image.thumbnail((config.IMAGE_SIZE, config.IMAGE_SIZE))
            
            # Create thumbnail path
            thumbnail_path = config.THUMBNAIL_DIR / image_path.name
            
            # Save thumbnail
            image.save(thumbnail_path)
            
            return thumbnail_path
        except Exception as e:
            logger.error(f"Error creating thumbnail: {e}")
            return None

    def analyze_colors(self, image_array: np.ndarray) -> Dict:
        """Analyze colors in an image."""
        try:
            return self.color_analyzer.analyze_colors(image_array)
        except Exception as e:
            logger.error(f"Error analyzing colors: {e}")
            return {
                "dominant_colors": [],
                "color_palette": [],
                "color_distribution": {}
            }

    def _load_search_logs(self) -> Dict:
        """Load search logs from file."""
        try:
            logs_path = config.BASE_DIR / "search_logs.json"
            if logs_path.exists():
                with open(logs_path, 'r') as f:
                    return json.load(f)
            return {"queries": [], "clicks": []}
        except Exception as e:
            logger.error(f"Error loading search logs: {e}")
            return {"queries": [], "clicks": []}

    def _save_search_logs(self):
        """Save search logs to file."""
        try:
            logs_path = config.BASE_DIR / "search_logs.json"
            with open(logs_path, 'w') as f:
                json.dump(self.search_logs, f)
        except Exception as e:
            logger.error(f"Error saving search logs: {e}")

    def log_search_query(self, query: str, result_count: int, search_time: float, 
                         session_id: str = None, user_id: str = None):
        """
        Log search query for analytics to improve search quality.
        
        Args:
            query: The search query string
            result_count: Number of results returned
            search_time: Time taken to perform the search
            session_id: Optional session identifier
            user_id: Optional user identifier
        """
        try:
            timestamp = import_time.time()
            log_entry = {
                "timestamp": timestamp,
                "query": query,
                "result_count": result_count,
                "search_time": search_time,
                "session_id": session_id,
                "user_id": user_id,
                "date": import_time.strftime("%Y-%m-%d %H:%M:%S", import_time.localtime(timestamp))
            }
            
            self.search_logs["queries"].append(log_entry)
            
            # Trim log if it gets too large (keep last 1000 entries)
            if len(self.search_logs["queries"]) > 1000:
                self.search_logs["queries"] = self.search_logs["queries"][-1000:]
                
            # Periodically save logs (every 10 searches)
            if len(self.search_logs["queries"]) % 10 == 0:
                self._save_search_logs()
                
            return True
        except Exception as e:
            logger.error(f"Error logging search query: {e}")
            return False
    
    def log_result_click(self, query: str, result_id: str, rank: int, 
                         session_id: str = None, user_id: str = None):
        """
        Log when a user clicks on a search result to help improve relevance.
        
        Args:
            query: The search query that produced the result
            result_id: ID of the clicked result
            rank: Position of the result in the results list (0-based)
            session_id: Optional session identifier
            user_id: Optional user identifier
        """
        try:
            timestamp = import_time.time()
            log_entry = {
                "timestamp": timestamp,
                "query": query,
                "result_id": result_id,
                "rank": rank,
                "session_id": session_id,
                "user_id": user_id,
                "date": import_time.strftime("%Y-%m-%d %H:%M:%S", import_time.localtime(timestamp))
            }
            
            self.search_logs["clicks"].append(log_entry)
            
            # Trim log if it gets too large (keep last 1000 entries)
            if len(self.search_logs["clicks"]) > 1000:
                self.search_logs["clicks"] = self.search_logs["clicks"][-1000:]
                
            # Always save logs when a click is recorded (important feedback)
            self._save_search_logs()
                
            return True
        except Exception as e:
            logger.error(f"Error logging result click: {e}")
            return False
            
    def get_search_analytics(self, days: int = 7) -> Dict:
        """
        Get search analytics for the specified time range.
        
        Args:
            days: Number of days to include in analytics
            
        Returns:
            Dictionary with analytics data
        """
        try:
            # Calculate cutoff timestamp
            cutoff = import_time.time() - (days * 24 * 60 * 60)
            
            # Filter queries and clicks by time range
            recent_queries = [q for q in self.search_logs["queries"] 
                              if q["timestamp"] >= cutoff]
            recent_clicks = [c for c in self.search_logs["clicks"] 
                             if c["timestamp"] >= cutoff]
                             
            # Calculate top queries
            query_counts = {}
            for query in recent_queries:
                q = query["query"].lower()
                if q in query_counts:
                    query_counts[q] += 1
                else:
                    query_counts[q] = 1
                    
            # Sort by count
            top_queries = [(q, c) for q, c in query_counts.items()]
            top_queries.sort(key=lambda x: x[1], reverse=True)
            
            # Calculate zero-result queries
            zero_results = [q for q in recent_queries if q["result_count"] == 0]
            
            # Calculate click-through rate
            query_results = {}
            for query in recent_queries:
                q = query["query"].lower()
                if q not in query_results:
                    query_results[q] = 0
                query_results[q] += query["result_count"]
                
            click_counts = {}
            for click in recent_clicks:
                q = click["query"].lower()
                if q in click_counts:
                    click_counts[q] += 1
                else:
                    click_counts[q] = 1
                    
            ctr_data = []
            for q, results in query_results.items():
                clicks = click_counts.get(q, 0)
                if results > 0:
                    ctr = clicks / results
                else:
                    ctr = 0
                ctr_data.append({"query": q, "results": results, "clicks": clicks, "ctr": ctr})
                
            # Sort by CTR (descending)
            ctr_data.sort(key=lambda x: x["ctr"], reverse=True)
            
            return {
                "period_days": days,
                "total_queries": len(recent_queries),
                "total_clicks": len(recent_clicks),
                "avg_results_per_query": sum(q["result_count"] for q in recent_queries) / max(len(recent_queries), 1),
                "avg_search_time": sum(q["search_time"] for q in recent_queries) / max(len(recent_queries), 1),
                "top_queries": top_queries[:20],  # Top 20 queries
                "zero_result_queries": [q["query"] for q in zero_results][:20],  # Top 20 zero-result queries
                "ctr_by_query": ctr_data[:20]  # Top 20 by CTR
            }
        except Exception as e:
            logger.error(f"Error generating search analytics: {e}")
            return {"error": str(e)}

    def search(self, query: str, k: int = 10, session_id: str = None, user_id: str = None) -> List[Dict]:
        """
        Advanced search for images based on pattern, color, and other metadata.
        If Elasticsearch is enabled, use it for search, otherwise fall back to in-memory search.
        Results are cached if Redis is available.
        
        Args:
            query: Search query string (can include commas to separate distinct terms)
            k: Maximum number of results to return
            session_id: Optional session identifier for analytics
            user_id: Optional user identifier for analytics
            
        Returns:
            List of image metadata dictionaries with similarity scores
        """
        min_similarity = config.DEFAULT_MIN_SIMILARITY
        search_start_time = import_time.time()
        
        # Force cache invalidation to ensure fresh results
        self.cache.invalidate_for_key_prefix("search:")
        
        # Create cache key
        cache_key = f"search:{query}:{k}:{min_similarity}"
        
        # Check cache first if enabled
        cached_results = self.cache.get(cache_key)
        if cached_results is not None:
            logger.info(f"Returning cached results for query: '{query}'")
            search_time = import_time.time() - search_start_time
            self.log_search_query(query, len(cached_results), search_time, session_id, user_id)
            return cached_results
        
        # Cache miss, perform search
        if self.use_elasticsearch:
            logger.info(f"Using Elasticsearch for search: '{query}'")
            try:
                results = self.es_client.search(
                    query=query, 
                    limit=k,
                    min_similarity=min_similarity
                )
                
                # Cache the results
                self.cache.set(cache_key, results)
                
                # Log the search for analytics
                search_time = import_time.time() - search_start_time
                self.log_search_query(query, len(results), search_time, session_id, user_id)
                
                return results
            except Exception as e:
                logger.error(f"Elasticsearch search failed, falling back to in-memory search: {e}")
        
        # Fall back to in-memory search
        logger.info(f"Using in-memory search: '{query}'")
        results = self._in_memory_search(query, k)
        
        # Cache the results
        self.cache.set(cache_key, results)
        
        # Log the search for analytics
        search_time = import_time.time() - search_start_time
        self.log_search_query(query, len(results), search_time, session_id, user_id)
        
        return results
    
    def _in_memory_search(self, query: str, k: int = 10) -> List[Dict]:
        """
        Simple in-memory search implementation as fallback.
        Treats the entire query as a single term for matching.
        Results are sorted by relevance to the query.
        
        Args:
            query: Search query string
            k: Maximum number of results to return
            
        Returns:
            List of image metadata dictionaries with similarity scores
        """
        if not self.metadata:
            logger.info("No metadata available for search")
            return []

        try:
            # Parse the query
            query = query.lower().strip()
            
            # Log the search query for analysis
            logger.info(f"In-memory search for: '{query}'")
            search_start_time = import_time.time()
            
            # Organize search results
            scored_results = []
            
            # Check each image
            for image_path, metadata in self.metadata.items():
                # Initialize score
                score = 0.0
                
                # Check patterns
                if 'patterns' in metadata and metadata['patterns']:
                    patterns = metadata['patterns']
                    
                    # Check primary pattern
                    primary_pattern = patterns.get('primary_pattern', '').lower()
                    if primary_pattern and query in primary_pattern:
                        score += 5.0
                    
                    # Check style keywords
                    for keyword in patterns.get('style_keywords', []):
                        if query in keyword.lower():
                            score += 3.0
                    
                    # Check prompt
                    prompt = patterns.get('prompt', {}).get('final_prompt', '').lower()
                    if query in prompt:
                        score += 2.0
                
                # Check if this is a color search
                if 'colors' in metadata and metadata['colors']:
                    for color_info in metadata['colors'].get('dominant_colors', []):
                        color_name = color_info.get('name', '').lower()
                        if query in color_name:
                            score += 4.0 * color_info.get('proportion', 0.5)
                
                # Only include results with non-zero scores
                if score > 0:
                    scored_results.append({
                        **metadata, 
                        'similarity': min(score / 10.0, 1.0)  # Normalize to 0-1
                    })
            
            # Sort results by similarity score (descending)
            scored_results.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Log search performance
            search_time = import_time.time() - search_start_time
            logger.info(f"In-memory search for '{query}' found {len(scored_results)} results in {search_time:.2f}s")
            
            return scored_results[:k]
        except Exception as e:
            logger.error(f"Error in in-memory search: {e}", exc_info=True)
            return []

    def delete_image(self, image_path: str) -> bool:
        """
        Delete image metadata and remove from Elasticsearch if enabled.
        Invalidates the cache to ensure consistency.
        
        Args:
            image_path: Path to the image to delete (relative to upload dir)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Remove from in-memory metadata
            if image_path in self.metadata:
                metadata = self.metadata.pop(image_path)
                self.save_metadata()
                
                # Remove from Elasticsearch if enabled
                if self.use_elasticsearch:
                    doc_id = metadata.get("id", image_path)
                    self.es_client.delete_document(doc_id)
                    
                # Invalidate cache
                self.cache.invalidate_all()
                    
                logger.info(f"Deleted image metadata: {image_path}")
                return True
            else:
                logger.warning(f"Image not found in metadata: {image_path}")
                return False
        except Exception as e:
            logger.error(f"Error deleting image: {e}")
            return False
            
    def update_image_metadata(self, image_path: str, new_metadata: Dict[str, Any]) -> bool:
        """
        Update image metadata and in Elasticsearch if enabled.
        Invalidates the cache to ensure consistency.
        
        Args:
            image_path: Path to the image to update (relative to upload dir)
            new_metadata: New metadata to apply
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Update in-memory metadata
            if image_path in self.metadata:
                # Merge new metadata with existing metadata
                self.metadata[image_path].update(new_metadata)
                self.save_metadata()
                
                # Update in Elasticsearch if enabled
                if self.use_elasticsearch:
                    doc_id = self.metadata[image_path].get("id", image_path)
                    self.es_client.update_document(doc_id, self.metadata[image_path])
                    
                # Invalidate cache
                self.cache.invalidate_all()
                    
                logger.info(f"Updated image metadata: {image_path}")
                return True
            else:
                logger.warning(f"Image not found in metadata: {image_path}")
                return False
        except Exception as e:
            logger.error(f"Error updating image metadata: {e}")
            return False
            
    def get_image_metadata(self, image_path: str) -> Dict[str, Any]:
        """Get metadata for a specific image."""
        return self.metadata.get(image_path, None)

# Create an instance of SearchEngine to be imported elsewhere
search_engine = SearchEngine() 