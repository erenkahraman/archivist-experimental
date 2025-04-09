from pathlib import Path
from typing import List, Dict, Any
import torch
from transformers import CLIPProcessor, CLIPModel
import json
from PIL import Image
import numpy as np
import logging
import time
import os
from datetime import datetime

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
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except (OSError, json.JSONDecodeError) as e:
            logger.error("Error loading metadata: %s", e)
            return {}

    def save_metadata(self):
        """Save metadata to file."""
        try:
            metadata_path = config.BASE_DIR / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f)
        except OSError as e:
            logger.error("Error saving metadata: %s", e)

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
            logger.info("Processing image: %s", image_path)
            
            # Check if file exists
            if not image_path.exists():
                logger.error("Image file not found: %s", image_path)
                return None
                
            # Create thumbnail
            thumbnail_path = self.create_thumbnail(image_path)
            if not thumbnail_path:
                logger.error("Failed to create thumbnail for: %s", image_path)
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
            
            # Check if this is a fallback result and log accordingly
            is_fallback = pattern_info.get("is_fallback", False)
            if is_fallback:
                logger.warning(f"Using fallback analysis for {image_path.name} due to API limitations")
            
            # Validate and enhance pattern_info if needed
            pattern_info = self._enhance_pattern_info(pattern_info)
            
            # Generate metadata
            timestamp = int(time.time())
            file_stats = image_path.stat()
            
            metadata = {
                'id': str(image_path.stem),
                'filename': image_path.name,
                'path': str(rel_image_path),
                'thumbnail_path': str(rel_thumbnail_path),
                'timestamp': timestamp,
                'added_date': datetime.fromtimestamp(timestamp).isoformat(),
                'last_modified': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                'file_size': file_stats.st_size,
                'width': width,
                'height': height,
                'patterns': pattern_info,
                'colors': color_info,
                'has_fallback_analysis': is_fallback
            }
            
            # Generate embedding for similarity search
            # embedding = self.get_image_embedding(image)
            # if embedding is not None:
            #     metadata['embedding'] = embedding.tolist()
            
            # Add to metadata store
            self.metadata[str(image_path)] = metadata
            self.save_metadata()
            
            # Index in Elasticsearch if available
            if self.use_elasticsearch:
                result = self.es_client.index_document(metadata)
                if result:
                    logger.info(f"Successfully indexed {image_path.name} in Elasticsearch")
                else:
                    logger.error(f"Failed to index {image_path.name} in Elasticsearch")
            
            logger.info("Image processed successfully: %s", image_path)
            return metadata
        except Exception as e:
            logger.error("Error processing image %s: %s", image_path, str(e), exc_info=True)
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
            timestamp = int(time.time())
            log_entry = {
                "timestamp": timestamp,
                "query": query,
                "result_count": result_count,
                "search_time": search_time,
                "session_id": session_id,
                "user_id": user_id,
                "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
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
            timestamp = int(time.time())
            log_entry = {
                "timestamp": timestamp,
                "query": query,
                "result_id": result_id,
                "rank": rank,
                "session_id": session_id,
                "user_id": user_id,
                "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
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
            cutoff = int(time.time()) - (days * 24 * 60 * 60)
            
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
        search_start_time = int(time.time())
        
        # Check cache first
        cache_key = f"search:{query}:{k}"
        cached_results = self.cache.get(cache_key)
        
        if cached_results:
            logger.info(f"Cache hit for '{query}'")
            # Log the search for analytics
            search_time = int(time.time()) - search_start_time
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
            search_time = int(time.time()) - search_start_time
            self.log_search_query(query, len(results), search_time, session_id, user_id)
            
            return results
        else:
            logger.warning("Elasticsearch is not available. Falling back to in-memory search.")
            # Fall back to in-memory search
            results = self._in_memory_search(query, k)
            
            # Cache the results
            self.cache.set(cache_key, results)
            
            # Log the search for analytics
            search_time = int(time.time()) - search_start_time
            self.log_search_query(query, len(results), search_time, session_id, user_id)
            
            return results

    def _in_memory_search(self, query: str, k: int = 10) -> List[Dict]:
        """
        Perform in-memory search when Elasticsearch is not available
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of matching documents
        """
        if not self.metadata:
            return []
        
        if query == "*" or not query.strip():
            # For empty queries or wildcards, return all sorted by timestamp
            results = sorted(
                self.metadata.values(), 
                key=lambda x: x.get("timestamp", 0),
                reverse=True
            )
            return results[:k]
        
        # Calculate match scores for all images
        scored_results = []
        
        # Lowercase the query for case-insensitive matching
        query = query.lower()
        
        # Split query into terms for term matching
        query_terms = [term.strip() for term in query.split() if term.strip()]
        
        for doc in self.metadata.values():
            # Skip if missing required data
            if not doc.get("patterns") or not doc.get("colors"):
                continue
            
            # Start with a base score
            score = 0.0
            pattern_matches = 0
            
            # Extract pattern info
            patterns = doc.get("patterns", {})
            
            # Check if this is a fallback analysis
            is_fallback = doc.get("has_fallback_analysis", False) or patterns.get("is_fallback", False)
            
            # Get the main theme and primary pattern
            main_theme = str(patterns.get("main_theme", "")).lower()
            primary_pattern = str(patterns.get("primary_pattern", "")).lower()
            
            # Get content details and style info
            content_details = patterns.get("content_details", [])
            stylistic_attributes = patterns.get("stylistic_attributes", [])
            prompt = patterns.get("prompt", {}).get("final_prompt", "").lower()
            style_keywords = [str(kw).lower() for kw in patterns.get("style_keywords", [])]
            
            # Get confidence scores
            main_theme_confidence = patterns.get("main_theme_confidence", 0.8)
            pattern_confidence = patterns.get("pattern_confidence", 0.7)
            
            # Check exact matches on main theme and primary pattern
            if query in main_theme:
                score += 10.0 * main_theme_confidence
                pattern_matches += 1
            
            if query in primary_pattern:
                score += 8.0 * pattern_confidence
                pattern_matches += 1
            
            # Check content details for matches
            for item in content_details:
                item_name = str(item.get("name", "")).lower()
                item_confidence = item.get("confidence", 0.7)
                
                if query in item_name:
                    score += 7.0 * item_confidence
                    pattern_matches += 1
                
                # Check for partial term matches
                for term in query_terms:
                    if term in item_name:
                        score += 3.0 * item_confidence
                        pattern_matches += 1
            
            # Check stylistic attributes for matches
            for item in stylistic_attributes:
                item_name = str(item.get("name", "")).lower()
                item_confidence = item.get("confidence", 0.7)
                
                if query in item_name:
                    score += 6.0 * item_confidence
                    pattern_matches += 1
                
                # Check for partial term matches
                for term in query_terms:
                    if term in item_name:
                        score += 2.5 * item_confidence
                        pattern_matches += 1
            
            # Check style keywords
            for keyword in style_keywords:
                if query in keyword:
                    score += 5.0
                    pattern_matches += 1
                
                # Check for partial term matches
                for term in query_terms:
                    if term in keyword:
                        score += 2.0
                        pattern_matches += 1
            
            # Check the prompt
            if query in prompt:
                score += 4.0
                pattern_matches += 1
            
            # Check for partial term matches in prompt
            for term in query_terms:
                if term in prompt:
                    score += 1.5
                    pattern_matches += 1
            
            # Check dominant colors
            colors = doc.get("colors", {})
            color_matches = 0
            
            for color in colors.get("dominant_colors", []):
                color_name = str(color.get("name", "")).lower()
                proportion = color.get("proportion", 0.0)
                
                if query in color_name:
                    score += 3.0 * proportion
                    color_matches += 1
                
                # Check for partial term matches
                for term in query_terms:
                    if term in color_name:
                        score += 1.0 * proportion
                        color_matches += 1
            
            # Add bonus if we matched both patterns and colors
            if pattern_matches > 0 and color_matches > 0:
                score += 2.0
            
            # Apply recency boost
            recency_boost = 1.0
            timestamp = doc.get("timestamp", 0)
            current_time = int(time.time())
            days_old = (current_time - timestamp) / (24 * 60 * 60)
            if days_old < 30:  # Less than 30 days old
                recency_boost = 1.0 + (1.0 - days_old / 30) * 0.5  # Up to 50% boost for very recent
            
            # Apply penalties for fallback results
            fallback_penalty = 0.75 if is_fallback else 1.0
            
            # Calculate final score
            final_score = score * recency_boost * fallback_penalty
            
            # Only include results that have some relevance
            if final_score > 0:
                similarity = min(1.0, final_score / 20.0)  # Normalize to 0-1 range
                
                # Add to results with normalized score
                result = doc.copy()
                result["similarity"] = similarity
                result["raw_score"] = final_score
                scored_results.append(result)
        
        # Sort by score (descending)
        scored_results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        
        # Return top k results
        return scored_results[:k]

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
            thumbnails_recreated = 0
            
            # Check each metadata entry for missing files
            for rel_path, metadata in self.metadata.items():
                # Get the full paths to check
                image_path = config.UPLOAD_DIR / rel_path if isinstance(rel_path, str) else None
                thumbnail_path = config.THUMBNAIL_DIR / metadata.get('filename') if metadata.get('filename') else None
                
                # Check if the image file exists
                if not image_path or not image_path.exists():
                    # Try to find image by filename
                    filename = metadata.get('filename')
                    if filename:
                        alternate_path = config.UPLOAD_DIR / filename
                        if alternate_path.exists():
                            logger.info(f"Found image at alternate path: {alternate_path}")
                            image_path = alternate_path
                        else:
                            entries_to_remove.append(rel_path)
                            logger.info(f"Adding missing image to cleanup: {rel_path}")
                            continue
                    else:
                        entries_to_remove.append(rel_path)
                        logger.info(f"Adding missing image to cleanup: {rel_path}")
                        continue
                    
                # If image exists but thumbnail doesn't, recreate the thumbnail
                if thumbnail_path and not thumbnail_path.exists():
                    try:
                        logger.info(f"Recreating missing thumbnail for: {rel_path}")
                        new_thumbnail = self.create_thumbnail(image_path)
                        if new_thumbnail:
                            thumbnails_recreated += 1
                        else:
                            logger.error(f"Failed to recreate thumbnail for {rel_path}")
                            # If we can't recreate the thumbnail, the entry should be removed
                            entries_to_remove.append(rel_path)
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
            if entries_to_remove or thumbnails_recreated > 0:
                self.save_metadata()
                # Invalidate cache to ensure consistency
                self.cache.invalidate_all()
                
            logger.info(f"Cleanup complete: {len(entries_to_remove)} entries removed, {thumbnails_recreated} thumbnails recreated")
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