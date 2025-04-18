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
import re

# Relative imports from the same package
from .analyzers.gemini_analyzer import GeminiAnalyzer
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
            
            # Don't load CLIP model immediately, defer until needed
            self.clip_model = None
            self.clip_processor = None
            self.device = None
            
            # Flag to track if CLIP model is loaded
            self._clip_model_loaded = False
            
            # No Elasticsearch client
            self.use_elasticsearch = False
            
            # Initialize Redis cache
            self.cache = SearchCache()
            
            # Load existing metadata if available
            self.metadata = self.load_metadata()
            
            # Initialize search analytics
            self.search_logs = self._load_search_logs()
            
            logger.info("SearchEngine initialization completed")
        except Exception as e:
            logger.error(f"Error initializing SearchEngine: {e}")
            raise

    def _mask_api_key(self, key):
        if not key or len(key) < 8:
            return "INVALID_KEY"
        # Show only first 4 and last 4 characters
        return f"{key[:4]}...{key[-4:]}"

    def load_metadata(self):
        """Load metadata from JSON file or initialize empty dict."""
        metadata_path = Path(config.BASE_DIR) / "metadata.json"
        if metadata_path.exists():
            try:
                logger.info("Loading metadata from %s", metadata_path)
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                logger.info("Loaded metadata for %d images", len(metadata))
                
                # Build the search index for faster lookups
                self._build_search_index(metadata)
                
                return metadata
            except Exception as e:
                logger.error("Error loading metadata: %s", e, exc_info=True)
                return {}
        else:
            logger.info("No metadata file found, initializing empty metadata")
            return {}
    
    def _build_search_index(self, metadata):
        """
        Build an inverted index for faster text search.
        
        This creates a dictionary mapping words to lists of image IDs
        containing those words in various fields.
        """
        logger.info("Building search index for metadata")
        self.search_index = {}
        
        # Weights for different fields to prioritize matches
        # Higher number = higher priority in results
        field_weights = {
            "patterns.primary_pattern": 5.0,
            "patterns.main_theme": 4.0,
            "patterns.keywords": 3.0,
            "colors.dominant_colors.name": 2.0,
            "filename": 1.5
        }
        
        # Index each metadata entry
        for image_path, entry in metadata.items():
            # Get the image ID (using path as fallback)
            image_id = entry.get('id', image_path)
            
            # Index each field with its weight
            for field_path, weight in field_weights.items():
                # Get the value using nested field path
                value = self._get_nested_field(entry, field_path)
                
                # Skip if no value found
                if not value:
                    continue
                    
                # Handle different value types
                if isinstance(value, list):
                    # For lists (like keywords), index each item
                    for item in value:
                        if isinstance(item, str):
                            self._index_text(item, image_id, weight)
                elif isinstance(value, str):
                    # For strings, index the full text
                    self._index_text(value, image_id, weight)
        
        # Get stats about the index
        total_keys = len(self.search_index)
        total_entries = sum(len(entries) for entries in self.search_index.values())
        logger.info(f"Search index built with {total_keys} unique terms and {total_entries} total entries")
    
    def _get_nested_field(self, data, field_path):
        """Get a value from a nested dictionary using dot notation path."""
        if not data or not field_path:
            return None
            
        parts = field_path.split('.')
        value = data
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None
                
        return value
    
    def _index_text(self, text, image_id, weight=1.0):
        """
        Index a text string by tokenizing it and mapping tokens to image IDs.
        Also indexes bigrams (2-word phrases) for phrase matching.
        """
        if not text or not isinstance(text, str):
            return
            
        # Normalize text: lowercase and remove special chars
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize
        words = text.split()
        
        # Index individual words
        for word in words:
            if len(word) > 2:  # Skip very short words
                if word not in self.search_index:
                    self.search_index[word] = {}
                
                # Store the image ID with its weight for this term
                # If already indexed, use the higher weight
                self.search_index[word][image_id] = max(
                    weight, 
                    self.search_index[word].get(image_id, 0)
                )
        
        # Index bigrams for phrase matching
        for i in range(len(words) - 1):
            if len(words[i]) > 2 and len(words[i+1]) > 2:
                bigram = f"{words[i]} {words[i+1]}"
                if bigram not in self.search_index:
                    self.search_index[bigram] = {}
                
                # Give bigrams a slightly higher weight than individual terms
                bigram_weight = weight * 1.2
                self.search_index[bigram][image_id] = max(
                    bigram_weight, 
                    self.search_index[bigram].get(image_id, 0)
                )

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
            logger.info("Gemini API key updated in SearchEngine")
        else:
            logger.warning("Attempted to set empty API key")

    def _load_clip_model(self):
        """Lazy load CLIP model when needed."""
        if self._clip_model_loaded:
            return True
            
        try:
            logger.info("Loading CLIP model for embeddings...")
            from transformers import CLIPProcessor, CLIPModel
            
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            
            # Move to GPU if available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model = self.clip_model.to(self.device)
            
            self._clip_model_loaded = True
            logger.info(f"CLIP model loaded successfully (using {self.device})")
            return True
        except Exception as e:
            logger.error(f"Error loading CLIP model: {e}")
            self._clip_model_loaded = False
            return False

    def cleanup_resources(self):
        """Clean up resources to free memory."""
        try:
            # Clean up CLIP model to free GPU memory
            if self._clip_model_loaded and self.clip_model is not None:
                logger.info("Cleaning up CLIP model resources")
                self.clip_model = None
                self.clip_processor = None
                
                # Explicitly run garbage collector
                import gc
                gc.collect()
                
                # Clear CUDA cache if using GPU
                if self.device == "cuda":
                    try:
                        torch.cuda.empty_cache()
                        logger.info("CUDA cache cleared")
                    except Exception as e:
                        logger.error(f"Error clearing CUDA cache: {e}")
                
                self._clip_model_loaded = False
            
            # Clear any temp files
            temp_dir = Path(config.BASE_DIR) / "temp"
            if temp_dir.exists():
                import shutil
                try:
                    for file in temp_dir.glob("*"):
                        if file.is_file():
                            file.unlink()
                        elif file.is_dir():
                            shutil.rmtree(file)
                    logger.info("Temporary files cleaned up")
                except Exception as e:
                    logger.error(f"Error cleaning up temporary files: {e}")
            
            # Force save any pending metadata
            self.save_metadata()
            
            logger.info("Resources cleaned up successfully")
            return True
        except Exception as e:
            logger.error(f"Error cleaning up resources: {e}")
            return False

    def get_image_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Generate CLIP embeddings for an image.
        
        Args:
            image: PIL Image to process
            
        Returns:
            Normalized embedding vector as numpy array
        """
        try:
            # Make sure CLIP model is loaded
            if not self._clip_model_loaded:
                if not self._load_clip_model():
                    logger.error("Failed to load CLIP model")
                    return None
                
            # Ensure the image is in RGB mode
            if image.mode != "RGB":
                image = image.convert("RGB")
                logger.debug("Converted image to RGB mode for CLIP processing")
                
            # Process image through CLIP
            with torch.no_grad():
                try:
                    # Prepare image for CLIP
                    inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
                    
                    # Get image features
                    outputs = self.clip_model.get_image_features(**inputs)
                    
                    # Convert to numpy and normalize to unit vector
                    embedding = outputs.cpu().numpy()[0]
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                    
                    # Verify embedding shape
                    if embedding.shape[0] != 512:
                        logger.warning(f"Unexpected embedding shape: {embedding.shape}, expected (512,)")
                    
                    logger.info(f"Successfully generated CLIP embedding with shape {embedding.shape}")
                    return embedding
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        logger.error("CUDA out of memory error. Trying with CPU fallback.")
                        # Clean up GPU memory
                        torch.cuda.empty_cache()
                        
                        # Try with CPU fallback
                        self.device = "cpu"
                        self.clip_model = self.clip_model.to("cpu")
                        # Retry with CPU
                        inputs = self.clip_processor(images=image, return_tensors="pt")
                        outputs = self.clip_model.get_image_features(**inputs)
                        embedding = outputs.cpu().numpy()[0]
                        norm = np.linalg.norm(embedding)
                        if norm > 0:
                            embedding = embedding / norm
                        logger.info(f"Successfully generated CLIP embedding on CPU with shape {embedding.shape}")
                        return embedding
                    else:
                        raise
        except Exception as e:
            logger.error(f"Error generating CLIP embedding: {e}", exc_info=True)
            return None

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
            
            # Generate embedding for similarity search - use the original image for best quality
            embedding = None
            # Only generate embeddings if the CLIP model is available or can be loaded
            if self._clip_model_loaded or self._load_clip_model():
                logger.info(f"Generating embedding for {image_path.name}")
                try:
                    # Use the full resolution image for embedding generation
                    original_image = Image.open(image_path).convert('RGB')
                    embedding = self.get_image_embedding(original_image)
                    
                    if embedding is None:
                        logger.error(f"Failed to generate embedding for {image_path.name}")
                    elif not isinstance(embedding, np.ndarray):
                        logger.error(f"Invalid embedding type for {image_path.name}: {type(embedding)}")
                        embedding = None
                    else:
                        logger.info(f"Successfully generated embedding for {image_path.name} with shape {embedding.shape}")
                except Exception as e:
                    logger.error(f"Error during embedding generation for {image_path.name}: {str(e)}", exc_info=True)
                    embedding = None
            else:
                logger.warning(f"Skipping embedding generation for {image_path.name} due to CLIP model unavailability")
            
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
            
            # Add embedding if available
            if embedding is not None:
                try:
                    # Convert numpy array to list
                    embedding_list = embedding.tolist()
                    metadata['embedding'] = embedding_list
                    logger.info(f"Added embedding with {len(embedding_list)} dimensions to metadata for {image_path.name}")
                except Exception as e:
                    logger.error(f"Error converting embedding to list for {image_path.name}: {str(e)}")
            else:
                logger.warning(f"No embedding available for {image_path.name}")
            
            # Add to metadata store
            self.metadata[str(image_path)] = metadata
            self.save_metadata()
            
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
        # Return default empty color analysis structure
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
        Search for images matching the query using in-memory search.
        
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
        
        # Cache miss, perform in-memory search
        logger.info(f"Using in-memory search for: '{query}'")
        results = self._in_memory_search(query, k)
        
        # Cache the results
        self.cache.set(cache_key, results)
        
        # Log the search for analytics
        search_time = int(time.time()) - search_start_time
        self.log_search_query(query, len(results), search_time, session_id, user_id)
        
        return results

    def _in_memory_search(self, query: str, k: int = 10) -> List[Dict]:
        """
        Perform in-memory search for the given query
        
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
        
        # Use the metadata_search method which provides more comprehensive text search
        return self.metadata_search(query, limit=k)

    def delete_image(self, image_path: str) -> bool:
        """
        Delete image metadata from memory and invalidate the cache.
        
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
                self.metadata.pop(key)
                self.save_metadata()
                
                # Invalidate cache
                self.cache.invalidate_all()
                    
                logger.info(f"Deleted image metadata for: {filename}")
                return True
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
                    self.metadata.pop(rel_path)
            
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
        Update image metadata and invalidate the cache to ensure consistency.
        
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

    def find_similar_images(self, image_path: str = None, image: Image.Image = None, 
                         text_query: str = None, k: int = 20, exclude_source: bool = True,
                         image_weight: float = 0.7, text_weight: float = 0.3) -> List[Dict]:
        """
        Find images similar to the provided image and/or matching the text query.
        This implements a hybrid search approach using both visual and textual similarity.
        
        Args:
            image_path: Path to the reference image
            image: PIL Image object (alternative to image_path)
            text_query: Optional text query to combine with image similarity
            k: Maximum number of results to return
            exclude_source: Whether to exclude the source image from results
            image_weight: Weight for image similarity (when using hybrid search)
            text_weight: Weight for text similarity (when using hybrid search)
            
        Returns:
            List of similar images sorted by similarity score
        """
        try:
            # Validate inputs
            if not image_path and image is None:
                logger.error("Either image_path or image must be provided")
                return []
                
            # Load the image if path is provided
            if image_path:
                if not os.path.exists(image_path):
                    logger.error(f"Image not found: {image_path}")
                    return []
                    
                image = Image.open(image_path).convert('RGB')
                image_id = os.path.basename(image_path)
            else:
                # Generate a temporary ID for the query image
                image_id = f"query_image_{int(time.time())}"
                
            logger.info(f"Finding similar images to {image_id}" + 
                       (f" with text query '{text_query}'" if text_query else ""))
                
            # Generate embedding using CLIP
            embedding = self.get_image_embedding(image)
            
            if embedding is None:
                logger.error("Failed to generate image embedding")
                return []
                
            # Perform in-memory similarity search
            logger.info("Using in-memory similarity search")
            return self._in_memory_similarity_search(embedding, text_query, k, exclude_source, image_id)
        except Exception as e:
            logger.error(f"Error finding similar images: {str(e)}", exc_info=True)
            return []
            
    def _in_memory_similarity_search(self, query_embedding, text_query=None, k=20, 
                                  exclude_source=True, source_id=None) -> List[Dict]:
        """
        Perform in-memory similarity search when Elasticsearch is not available
        
        Args:
            query_embedding: The embedding vector to compare against
            text_query: Optional text query to filter results
            k: Number of results to return
            exclude_source: Whether to exclude the source image
            source_id: ID of source image to exclude
            
        Returns:
            List of similar documents
        """
        if not self.metadata:
            return []
            
        # Calculate similarity scores
        results = []
        
        for doc_id, doc in self.metadata.items():
            # Skip if it's the source image
            if exclude_source and source_id and (
                source_id == doc_id or 
                source_id == doc.get("id") or 
                source_id == doc.get("filename")
            ):
                continue
                
            # Get embedding if available
            doc_embedding = doc.get("embedding")
            
            if doc_embedding is None:
                # Skip documents without embeddings
                continue
                
            # Convert to numpy array if it's a list
            if isinstance(doc_embedding, list):
                doc_embedding = np.array(doc_embedding)
                
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, doc_embedding)
            
            # If text query is provided, also perform text matching
            text_match_score = 0.0
            if text_query:
                text_match_score = self._calculate_text_match(doc, text_query.lower())
                
                # Combine scores (70% visual, 30% text by default)
                final_score = 0.7 * similarity + 0.3 * text_match_score
            else:
                final_score = similarity
                
            # Create result object
            result = doc.copy()
            result["similarity"] = float(final_score)
            results.append(result)
            
        # Sort by similarity (descending)
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Return top k results
        return results[:k]
        
    def _calculate_text_match(self, doc, query):
        """Calculate a simple text match score for in-memory similarity search"""
        if not doc.get("patterns"):
            return 0.0
            
        patterns = doc["patterns"]
        score = 0.0
        
        # Check exact matches on main fields
        for field, weight in [
            ("main_theme", 5.0),
            ("primary_pattern", 4.5),
            ("category", 4.0)
        ]:
            value = str(patterns.get(field, "")).lower()
            if query in value:
                score += weight
                # Bonus for exact match
                if query == value:
                    score += weight * 0.5
                    
        # Check content details
        for item in patterns.get("content_details", []):
            if query in str(item.get("name", "")).lower():
                score += 3.0 * item.get("confidence", 0.7)
                
        # Check style keywords
        for keyword in patterns.get("style_keywords", []):
            if query in str(keyword).lower():
                score += 2.0
                
        # Normalize to 0-1 range
        return min(1.0, score / 10.0) 

    def reindex_all_with_embeddings(self, force=False):
        """Regenerate embeddings for all images."""
        # Check if CLIP model can be loaded first to avoid processing images when embeddings can't be generated
        if not self._clip_model_loaded and not self._load_clip_model():
            logger.error("Cannot reindex embeddings as CLIP model failed to load")
            return {
                "error": "CLIP model failed to load",
                "success": False,
                "total": 0,
                "processed": 0,
                "success": 0,
                "failed": 0
            }
            
        logger.info(f"Starting regeneration of embeddings for {len(self.metadata)} images")
        
        # Statistics to track progress
        stats = {
            "total": len(self.metadata),
            "processed": 0,
            "success": 0,
            "failed": 0,
            "already_had_embedding": 0,
            "missing_file": 0,
            "invalid_path": 0
        }
        
        # Process each image
        for image_path_str, metadata in self.metadata.items():
            stats["processed"] += 1
            
            try:
                # Check if this entry already has an embedding and we're not forcing regeneration
                if not force and 'embedding' in metadata and metadata['embedding'] is not None:
                    logger.info(f"Image {image_path_str} already has embedding and force=False")
                    stats["already_had_embedding"] += 1
                    continue
                
                # Convert string path to Path object
                try:
                    image_path = Path(image_path_str)
                    if not image_path.exists():
                        logger.warning(f"Image file not found: {image_path}")
                        stats["missing_file"] += 1
                        continue
                except Exception as e:
                    logger.error(f"Invalid path: {image_path_str} - {str(e)}")
                    stats["invalid_path"] += 1
                    continue
                
                # Generate embedding
                logger.info(f"Regenerating embedding for {image_path.name}")
                try:
                    # Use the full resolution image for embedding generation
                    original_image = Image.open(image_path).convert('RGB')
                    embedding = self.get_image_embedding(original_image)
                    
                    if embedding is None:
                        logger.error(f"Failed to generate embedding for {image_path.name}")
                        stats["failed"] += 1
                        continue
                    
                    # Update metadata with new embedding
                    metadata['embedding'] = embedding.tolist()
                    
                    # Update in-memory metadata store
                    self.metadata[image_path_str] = metadata
                    
                    stats["success"] += 1
                    logger.info(f"Successfully regenerated embedding for {image_path.name}")
                except Exception as e:
                    logger.error(f"Error regenerating embedding for {image_path.name}: {str(e)}", exc_info=True)
                    stats["failed"] += 1
            except Exception as e:
                logger.error(f"Unexpected error processing {image_path_str}: {str(e)}", exc_info=True)
                stats["failed"] += 1
                
            # Save metadata periodically (every 10 processed)
            if stats["processed"] % 10 == 0:
                self.save_metadata()
                logger.info(f"Progress: {stats['processed']}/{stats['total']} images processed")
        
        # Save updated metadata
        self.save_metadata()
        logger.info(f"Saved updated metadata with {stats['success']} new embeddings")
        
        # Invalidate cache
        self.cache.invalidate_all()
        
        # Clean up resources after batch processing
        self.cleanup_resources()
        
        # Return statistics
        stats["success_overall"] = stats["success"] > 0
        return stats

    def metadata_search(self, query, max_results=100, filter_criteria=None):
        """
        Search images directly from metadata using the inverted index.
        
        Args:
            query: Search text
            max_results: Maximum number of results to return
            filter_criteria: Dictionary of metadata fields to filter results
            
        Returns:
            List of matching image metadata
        """
        # Check for cached results first
        cache_key = f"metadata_search:{query}:{max_results}:{json.dumps(filter_criteria) if filter_criteria else 'none'}"
        cached_results = self.cache.get(cache_key)
        if cached_results:
            logger.info(f"Using cached results for query: {query}")
            return cached_results
            
        # Start timing search
        start_time = time.time()
        
        # Handle empty query or missing metadata
        if not query or not self.metadata:
            return []
        
        # Handle wildcard queries
        if query == "*" or query == "all":
            # For wildcard, get all items (respecting filter)
            results = self._filter_metadata(list(self.metadata.values()), filter_criteria)
            # Sort by timestamp, newest first
            results = sorted(results, key=lambda x: x.get('timestamp', 0), reverse=True)
            results = results[:max_results]
            # Cache these results
            self.cache.set(cache_key, results)
            
            elapsed = time.time() - start_time
            logger.info(f"Wildcard search returned {len(results)} results in {elapsed:.3f}s")
            
            return results
            
        # Normalize query and split into terms
        query = query.lower()
        query = re.sub(r'[^\w\s]', ' ', query)
        query_terms = [term for term in query.split() if len(term) > 2]
        
        # Create a dictionary to store relevance scores for each image
        scores = {}
        
        # First try to match the full query as is (including phrases)
        if query in self.search_index:
            for image_id, weight in self.search_index[query].items():
                # Higher weighting for exact phrase matches
                scores[image_id] = weight * 2.0
        
        # Then search for individual terms and their weights
        for term in query_terms:
            if term in self.search_index:
                for image_id, weight in self.search_index[term].items():
                    # Add to existing score or create new entry
                    scores[image_id] = scores.get(image_id, 0) + weight
        
        # Get full metadata for matched images
        results = []
        for image_path, item in self.metadata.items():
            image_id = item.get('id', image_path)
            
            # If this image has a score, add it to results with the score
            if image_id in scores:
                item_copy = item.copy()  # Avoid modifying original
                item_copy['_score'] = scores[image_id]
                results.append(item_copy)
        
        # Apply filters if provided
        if filter_criteria:
            results = self._filter_metadata(results, filter_criteria)
        
        # Sort by score (highest first)
        results = sorted(results, key=lambda x: x.get('_score', 0), reverse=True)
        
        # Limit results
        results = results[:max_results]
        
        # Cache the results
        self.cache.set(cache_key, results)
        
        # Log and return
        elapsed = time.time() - start_time
        logger.info(f"Metadata search for '{query}' returned {len(results)} results in {elapsed:.3f}s")
        
        return results

    def _filter_metadata(self, metadata_list, filter_criteria):
        """Filter metadata items based on criteria."""
        if not filter_criteria:
            return metadata_list
            
        filtered = []
        for item in metadata_list:
            match = True
            
            # Apply each filter criteria
            for field, value in filter_criteria.items():
                field_value = self._get_nested_field(item, field)
                
                # Skip items that don't match
                if field_value is None or field_value != value:
                    match = False
                    break
            
            if match:
                filtered.append(item)
                
        return filtered 