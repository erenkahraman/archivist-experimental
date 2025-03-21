from pathlib import Path
from typing import List, Dict, Any
import torch
from transformers import CLIPProcessor, CLIPModel
import json
from PIL import Image
import numpy as np
import logging
import time as import_time

# Relative imports from the same package
from .analyzers.color_analyzer import ColorAnalyzer
from .gemini_analyzer import GeminiAnalyzer
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
            
            # Load existing metadata if available
            self.metadata = self.load_metadata()
            
            logger.info("SearchEngine initialization completed")
        except Exception as e:
            logger.error(f"Error initializing SearchEngine: {e}")
            raise

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
            
            # Store metadata
            self.metadata[str(rel_image_path)] = metadata
            self.save_metadata()
            
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

    def search(self, query: str, k: int = 10) -> List[Dict]:
        """Search for images based on a text query using text-based filtering."""
        try:
            # Get all images from metadata
            if not self.metadata:
                return []

            results = []
            query_terms = query.lower().split()
            
            for path, metadata in self.metadata.items():
                # Initialize match score
                score = 0
                
                # Check pattern matches
                if 'patterns' in metadata:
                    patterns = metadata['patterns']
                    
                    # Check category/primary pattern
                    if 'category' in patterns:
                        category = patterns['category'].lower()
                        for term in query_terms:
                            if term in category:
                                score += 2
                                
                    # Check elements
                    for element in patterns.get('elements', []):
                        element_name = element.get('name', '').lower()
                        for term in query_terms:
                            if term in element_name:
                                score += 1
                
                # Check color matches
                if 'colors' in metadata:
                    for color in metadata['colors'].get('dominant_colors', []):
                        color_name = color.get('name', '').lower()
                        for term in query_terms:
                            if term in color_name:
                                score += 1
                
                if score > 0:
                    # Add metadata with score
                    results.append({
                        **metadata,
                        'similarity': min(score / (len(query_terms) * 2), 1.0)  # Normalize to 0-1
                    })
            
            # Sort by score
            results.sort(key=lambda x: x['similarity'], reverse=True)
            
            return results[:k]
        except Exception as e:
            logger.error(f"Error in search: {e}")
            return []
            
    def get_image_metadata(self, image_path: str) -> Dict[str, Any]:
        """
        Get metadata for a specific image
        
        Args:
            image_path: Path to the image (can be absolute or relative)
            
        Returns:
            Dictionary containing image metadata or None if not found
        """
        try:
            logger.info(f"Getting metadata for image: {image_path}")
            
            # Handle both absolute and relative paths
            if Path(image_path).is_absolute():
                # Convert absolute path to relative path
                try:
                    rel_path = str(Path(image_path).relative_to(config.UPLOAD_DIR))
                except ValueError:
                    # If the path is not relative to UPLOAD_DIR, use as is
                    rel_path = image_path
            else:
                rel_path = image_path
                
            logger.info(f"Looking for metadata with path: {rel_path}")
                
            # Try to find the metadata
            if rel_path in self.metadata:
                logger.info(f"Found metadata using exact path match: {rel_path}")
                return self.metadata[rel_path]
                
            # If not found directly, try to find by filename
            filename = Path(image_path).name
            logger.info(f"Trying to find metadata by filename: {filename}")
            
            for path, metadata in self.metadata.items():
                if Path(path).name == filename:
                    logger.info(f"Found metadata using filename match: {path}")
                    return metadata
            
            # Try with different path formats
            # 1. Try with just the filename
            if filename in self.metadata:
                logger.info(f"Found metadata using just filename: {filename}")
                return self.metadata[filename]
                
            # 2. Try with uploads/filename
            uploads_path = f"uploads/{filename}"
            if uploads_path in self.metadata:
                logger.info(f"Found metadata using uploads/filename: {uploads_path}")
                return self.metadata[uploads_path]
                
            # 3. Try with full path but different base
            for path in self.metadata:
                if path.endswith(f"/{filename}"):
                    logger.info(f"Found metadata using path ending: {path}")
                    return self.metadata[path]
                    
            logger.warning(f"Metadata not found for image: {image_path}")
            return None
        except Exception as e:
            logger.error(f"Error getting image metadata: {e}")
            return None 