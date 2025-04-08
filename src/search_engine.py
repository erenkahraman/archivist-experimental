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
                
                # Create index if it doesn't exist, force recreation
                self.es_client.create_index(force_recreate=True)
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
            
            # Validate and enhance pattern_info if needed
            pattern_info = self._enhance_pattern_info(pattern_info)
            
            # Generate metadata
            metadata = {
                'id': str(image_path.stem),
                'filename': image_path.name,
                'path': str(rel_image_path),
                'thumbnail_path': str(rel_thumbnail_path),
                'patterns': pattern_info,
                'colors': color_info,
                'timestamp': int(import_time.time())
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
            timestamp = int(import_time.time())
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
            timestamp = int(import_time.time())
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
            cutoff = int(import_time.time()) - (days * 24 * 60 * 60)
            
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
        Search for images matching the query. Uses Elasticsearch if available, or falls back to in-memory search.
        
        Args:
            query: The search query string
            k: Maximum number of results to return
            session_id: Optional session ID for logging
            user_id: Optional user ID for logging
            
        Returns:
            List of matching documents sorted by similarity
        """
        # Log analytics info and timing
        logger.info(f"Searching for: '{query}'")
        search_start_time = int(import_time.time())
        
        # Check cache first
        cache_key = f"search:{query}:{k}"
        cached_results = self.cache.get(cache_key)
        
        if cached_results:
            logger.info(f"Cache hit for '{query}'")
            # Log the search for analytics
            search_time = int(import_time.time()) - search_start_time
            self.log_search_query(query, len(cached_results), search_time, session_id, user_id)
            return cached_results
        
        # Cache miss, perform search
        if self.use_elasticsearch and self.es_client.is_connected():
            logger.info(f"Using Elasticsearch to search for: '{query}'")
            # Use elasticsearch search
            results = self.es_client.search(query, limit=k)
            
            # Cache the results
            self.cache.set(cache_key, results)
            
            # Log the search for analytics
            search_time = int(import_time.time()) - search_start_time
            self.log_search_query(query, len(results), search_time, session_id, user_id)
            
            return results
        else:
            logger.warning("Elasticsearch is not available. Falling back to in-memory search.")
            # Fall back to in-memory search
            results = self._in_memory_search(query, k)
            
            # Cache the results
            self.cache.set(cache_key, results)
            
            # Log the search for analytics
            search_time = int(import_time.time()) - search_start_time
            self.log_search_query(query, len(results), search_time, session_id, user_id)
            
            return results

    def _in_memory_search(self, query: str, k: int = 10) -> List[Dict]:
        """
        Perform in-memory search when Elasticsearch is not available.
        Enhanced to match the Elasticsearch search capabilities.
        
        Args:
            query: Search query
            k: Maximum number of results
            
        Returns:
            List of results sorted by relevance
        """
        logger.info(f"Performing enhanced in-memory search for: '{query}'")
        
        if not self.metadata:
            logger.warning("No metadata available for in-memory search")
            return []
        
        # Convert query to lowercase for case-insensitive matching
        query_lower = query.lower()
        
        # Split query into terms for more flexible matching
        query_terms = [term.strip() for term in query_lower.split() if term.strip()]
        
        # Calculate similarity scores for each document
        scored_docs = []
        for doc_id, doc in self.metadata.items():
            # Initialize score components
            score_components = {
                "main_theme_exact": 0.0,
                "main_theme_partial": 0.0,
                "primary_pattern_exact": 0.0,
                "primary_pattern_partial": 0.0,
                "content_details": 0.0,
                "stylistic_attributes": 0.0,
                "colors": 0.0,
                "prompt": 0.0
            }
            
            # Get patterns data
            patterns = doc.get("patterns", {})
            
            # Exact match on main_theme (highest weight)
            main_theme = patterns.get("main_theme", "").lower()
            if main_theme:
                if main_theme == query_lower:
                    score_components["main_theme_exact"] = 5.0
                elif any(term in main_theme for term in query_terms):
                    score_components["main_theme_partial"] = 3.0
            
            # Exact match on primary_pattern
            primary_pattern = patterns.get("primary_pattern", "").lower()
            if primary_pattern:
                if primary_pattern == query_lower:
                    score_components["primary_pattern_exact"] = 4.5
                elif any(term in primary_pattern for term in query_terms):
                    score_components["primary_pattern_partial"] = 2.5
            
            # Check content details with weights
            content_details = patterns.get("content_details", [])
            content_score = 0.0
            for detail in content_details:
                if not isinstance(detail, dict):
                    continue
                    
                name = detail.get("name", "").lower()
                confidence = detail.get("confidence", 0.7)
                
                if name and name == query_lower:
                    content_score += 2.0 * confidence
                elif name and any(term in name for term in query_terms):
                    content_score += 1.5 * confidence
            
            score_components["content_details"] = min(content_score, 2.0)  # Cap content score
            
            # Check stylistic attributes
            stylistic_attrs = patterns.get("stylistic_attributes", [])
            style_score = 0.0
            for attr in stylistic_attrs:
                if not isinstance(attr, dict):
                    continue
                    
                name = attr.get("name", "").lower()
                confidence = attr.get("confidence", 0.6)
                
                if name and name == query_lower:
                    style_score += 1.5 * confidence
                elif name and any(term in name for term in query_terms):
                    style_score += 1.0 * confidence
            
            score_components["stylistic_attributes"] = min(style_score, 1.5)  # Cap style score
            
            # Check prompt
            prompt = patterns.get("prompt", {}).get("final_prompt", "").lower()
            if prompt:
                if query_lower in prompt:
                    score_components["prompt"] = 1.2
                elif any(term in prompt for term in query_terms):
                    score_components["prompt"] = 0.8
            
            # Check colors
            colors = doc.get("colors", {})
            dominant_colors = colors.get("dominant_colors", [])
            color_score = 0.0
            for color in dominant_colors:
                if not isinstance(color, dict):
                    continue
                    
                name = color.get("name", "").lower()
                proportion = color.get("proportion", 0.5)
                
                if name and name == query_lower:
                    color_score += 2.0 * proportion
                elif name and any(term in name for term in query_terms):
                    color_score += 1.5 * proportion
            
            score_components["colors"] = min(color_score, 1.5)  # Cap color score
            
            # Calculate base score as sum of components
            base_score = sum(score_components.values())
            
            # Apply confidence boosting similar to function_score query
            main_theme_confidence = patterns.get("main_theme_confidence", 0.8)
            pattern_confidence = patterns.get("pattern_confidence", 0.7)
            
            # Apply confidence multipliers
            confidence_boost = 1.0
            if base_score > 0:
                confidence_boost = 1.0 + (0.5 * main_theme_confidence) + (0.3 * pattern_confidence)
            
            # Apply recency boost
            recency_boost = 1.0
            timestamp = doc.get("timestamp", 0)
            current_time = int(import_time.time())
            days_old = (current_time - timestamp) / (24 * 60 * 60)
            if days_old < 30:  # Less than 30 days old
                recency_boost = 1.0 + (0.5 * (1 - (days_old / 30)))
            
            # Calculate final score
            final_score = base_score * confidence_boost * recency_boost
            
            # If any match found, add to results
            if final_score > 0:
                # Add document with its score
                doc_copy = doc.copy()
                doc_copy["similarity"] = min(final_score / 15.0, 1.0)  # Normalize to [0,1] range and cap at 1.0
                doc_copy["score_components"] = score_components  # For debugging
                scored_docs.append(doc_copy)
        
        # Sort by similarity (descending)
        scored_docs.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Remove score_components from final results
        for doc in scored_docs:
            if "score_components" in doc:
                del doc["score_components"]
        
        # Return top k results
        return scored_docs[:k]

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
            # Handle both full paths and just filenames
            if os.path.sep in image_path:
                # It's a path, check if it exists as is
                key = image_path
                # Also check if it's stored with a different path format
                filename = os.path.basename(image_path)
            else:
                # It's just a filename, find it in metadata by filename
                filename = image_path
                key = None
                for k, v in self.metadata.items():
                    if v.get('filename') == filename:
                        key = k
                        break
            
            # Remove from in-memory metadata using the correct key
            if key and key in self.metadata:
                metadata = self.metadata.pop(key)
                self.save_metadata()
                
                # Remove from Elasticsearch if enabled
                if self.use_elasticsearch:
                    doc_id = metadata.get("id", os.path.basename(key))
                    self.es_client.delete_document(doc_id)
                    
                # Invalidate cache
                self.cache.invalidate_all()
                    
                logger.info(f"Deleted image metadata for: {filename}")
                return True
            elif filename:
                # Try to find by ID or filename in Elasticsearch
                if self.use_elasticsearch:
                    # Try to delete by filename as ID
                    self.es_client.delete_document(filename)
                    # Also try by filename without extension
                    name_without_ext = os.path.splitext(filename)[0]
                    self.es_client.delete_document(name_without_ext)
                    # Invalidate cache
                    self.cache.invalidate_all()
                    logger.info(f"Deleted image from Elasticsearch by filename: {filename}")
                    return True
                
                logger.warning(f"Image not found in metadata by filename: {filename}")
                return False
            else:
                logger.warning(f"Image not found in metadata: {image_path}")
                return False
        except Exception as e:
            logger.error(f"Error deleting image: {e}")
            return False

    def cleanup_missing_files(self) -> int:
        """
        Clean up metadata for files that no longer exist.
        This helps prevent "missing thumbnail" errors in the UI.
        
        Returns:
            int: Number of entries cleaned up
        """
        try:
            if not self.metadata:
                logger.info("No metadata to clean up")
                return 0
                
            entries_to_remove = []
            
            # Check each metadata entry for missing files
            for rel_path, metadata in self.metadata.items():
                # Get the full paths to check
                image_path = config.UPLOAD_DIR / rel_path if isinstance(rel_path, str) else None
                thumbnail_path = config.THUMBNAIL_DIR / metadata.get('filename') if metadata.get('filename') else None
                
                # Check if the image file exists
                if not image_path or not image_path.exists():
                    entries_to_remove.append(rel_path)
                    logger.info(f"Adding missing image to cleanup: {rel_path}")
                    continue
                    
                # If image exists but thumbnail doesn't, recreate the thumbnail
                if thumbnail_path and not thumbnail_path.exists():
                    try:
                        logger.info(f"Recreating missing thumbnail for: {rel_path}")
                        self.create_thumbnail(image_path)
                    except Exception as thumb_err:
                        logger.error(f"Failed to recreate thumbnail for {rel_path}: {thumb_err}")
                        # If we can't recreate the thumbnail, the entry should be removed
                        entries_to_remove.append(rel_path)
            
            # Remove the entries for missing files
            for rel_path in entries_to_remove:
                if rel_path in self.metadata:
                    metadata = self.metadata.pop(rel_path)
                    
                    # Remove from Elasticsearch if enabled
                    if self.use_elasticsearch:
                        doc_id = metadata.get("id", rel_path) 
                        filename = metadata.get("filename", "")
                        # Try multiple ways to delete from Elasticsearch
                        self.es_client.delete_document(doc_id)
                        if filename:
                            self.es_client.delete_document(filename)
                            name_without_ext = os.path.splitext(filename)[0]
                            self.es_client.delete_document(name_without_ext)
            
            # Save the updated metadata if any entries were removed
            if entries_to_remove:
                self.save_metadata()
                # Invalidate cache to ensure consistency
                self.cache.invalidate_all()
                
            return len(entries_to_remove)
            
        except Exception as e:
            logger.error(f"Error cleaning up missing files: {e}")
            return 0

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

    def _enhance_pattern_info(self, pattern_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance and validate pattern info to ensure all required fields are present
        with proper structure for optimal search.
        
        Args:
            pattern_info: Original pattern information from the analyzer
            
        Returns:
            Enhanced pattern information with all required fields
        """
        if not pattern_info:
            pattern_info = {}
            
        # Ensure primary pattern field exists
        if not pattern_info.get('primary_pattern'):
            pattern_info['primary_pattern'] = pattern_info.get('category', 'Unknown')
            
        # Ensure pattern confidence exists
        if not pattern_info.get('pattern_confidence'):
            pattern_info['pattern_confidence'] = pattern_info.get('category_confidence', 0.8)
            
        # Ensure main_theme field exists
        if not pattern_info.get('main_theme'):
            pattern_info['main_theme'] = pattern_info.get('primary_pattern', 
                                          pattern_info.get('category', 'Unknown'))
            
        # Ensure main_theme_confidence exists
        if not pattern_info.get('main_theme_confidence'):
            pattern_info['main_theme_confidence'] = pattern_info.get('pattern_confidence', 
                                                    pattern_info.get('category_confidence', 0.8))
        
        # Ensure content_details exists and has proper structure
        if not isinstance(pattern_info.get('content_details'), list):
            pattern_info['content_details'] = []
            
            # Try to generate from elements if available
            if isinstance(pattern_info.get('elements'), list):
                for element in pattern_info.get('elements', []):
                    if isinstance(element, dict) and element.get('name'):
                        pattern_info['content_details'].append({
                            'name': element.get('name', ''),
                            'confidence': element.get('confidence', 0.8)
                        })
            
            # If still empty, extract from style_keywords or prompt
            if not pattern_info['content_details'] and isinstance(pattern_info.get('style_keywords'), list):
                # Use first two style keywords as content elements
                for i, keyword in enumerate(pattern_info.get('style_keywords', [])[:2]):
                    pattern_info['content_details'].append({
                        'name': keyword,
                        'confidence': 0.7
                    })
            
            # If still empty, extract from prompt
            if not pattern_info['content_details'] and pattern_info.get('prompt', {}).get('final_prompt'):
                prompt_text = pattern_info['prompt']['final_prompt']
                # Extract key terms from prompt
                terms = [term.strip() for term in prompt_text.split(',') if term.strip()]
                for i, term in enumerate(terms[:2]):
                    pattern_info['content_details'].append({
                        'name': term,
                        'confidence': 0.7
                    })
                    
            # If still empty, add a placeholder
            if not pattern_info['content_details']:
                pattern_info['content_details'].append({
                    'name': pattern_info.get('main_theme', 'Unknown'),
                    'confidence': 0.6
                })
        
        # Ensure stylistic_attributes exists and has proper structure
        if not isinstance(pattern_info.get('stylistic_attributes'), list):
            pattern_info['stylistic_attributes'] = []
            
            # Try to generate from style_keywords if available
            if isinstance(pattern_info.get('style_keywords'), list):
                for keyword in pattern_info.get('style_keywords', []):
                    pattern_info['stylistic_attributes'].append({
                        'name': keyword,
                        'confidence': 0.7
                    })
            
            # If still empty, extract from prompt
            if not pattern_info['stylistic_attributes'] and pattern_info.get('prompt', {}).get('final_prompt'):
                prompt_text = pattern_info['prompt']['final_prompt']
                # Extract adjectives from prompt
                terms = [term.strip() for term in prompt_text.split() if term.strip()]
                for i, term in enumerate(terms[:3]):
                    if len(term) > 3:  # Simple filter for meaningful terms
                        pattern_info['stylistic_attributes'].append({
                            'name': term,
                            'confidence': 0.6
                        })
                        
            # If still empty, add a placeholder
            if not pattern_info['stylistic_attributes']:
                pattern_info['stylistic_attributes'].append({
                    'name': 'basic',
                    'confidence': 0.5
                })
        
        # Ensure secondary_patterns exists
        if not isinstance(pattern_info.get('secondary_patterns'), list):
            pattern_info['secondary_patterns'] = []
            
        # Ensure style_keywords exists
        if not isinstance(pattern_info.get('style_keywords'), list):
            pattern_info['style_keywords'] = []
            
        # Ensure prompt exists
        if not isinstance(pattern_info.get('prompt'), dict):
            pattern_info['prompt'] = {
                'final_prompt': pattern_info.get('main_theme', 'Unknown pattern')
            }
        elif not pattern_info['prompt'].get('final_prompt'):
            pattern_info['prompt']['final_prompt'] = pattern_info.get('main_theme', 'Unknown pattern')
            
        return pattern_info 