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
from .search.searcher import ImageSearcher
import config
from .embedding_extractor import EmbeddingExtractor
from .gemini_analyzer import GeminiAnalyzer

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
            # Load models and move to GPU if available
            logger.info("Loading CLIP model: openai/clip-vit-large-patch14")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
            logger.info("CLIP model loaded successfully")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            logger.info("CLIP processor loaded successfully")
            
            # Initialize analyzers - securely pass API key without logging it
            if gemini_api_key:
                # Mask the key for logging
                masked_key = self._mask_api_key(gemini_api_key)
                logger.info(f"Using Gemini API key: {masked_key}")
            
            # Initialize analyzers without storing the key in this class
            self.gemini_analyzer = GeminiAnalyzer(api_key=gemini_api_key)
            self.color_analyzer = ColorAnalyzer(max_clusters=config.N_CLUSTERS, api_key=gemini_api_key)
            self.searcher = ImageSearcher(self.model, self.processor, self.device)
            
            # Initialize embedding extractor
            self.embedding_extractor = EmbeddingExtractor(self.model, self.processor, self.device)
            
            # Load existing metadata if available
            self.metadata = self.load_metadata()
            
            logger.info("SearchEngine initialization completed")
        except Exception as e:
            logger.error(f"Error initializing SearchEngine: {e}")
            raise

    def _mask_api_key(self, key):
        """Safely mask API key for logging purposes."""
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
            
            # Extract image features for search
            image = Image.open(image_path).convert('RGB')
            image_features = self.embedding_extractor.extract_image_features(image)
            
            # Convert to numpy array for color analysis
            image_np = np.array(image)
            
            # Analyze colors
            color_info = self.analyze_colors(image_np)
            
            # Use Gemini for pattern analysis
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
        """Search for images based on a text query."""
        try:
            # Get image paths and features
            image_paths = list(self.metadata.keys())
            if not image_paths:
                return []
            
            # Perform search
            results = self.searcher.search(query, image_paths, self.metadata, k)
            
            return results
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