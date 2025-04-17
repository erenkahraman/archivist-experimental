from pathlib import Path
from typing import Dict, Any, List, Optional
import json
from PIL import Image
import numpy as np
import logging
import time
import os
import shutil

# Import just what's needed
from src.analyzers.gemini_analyzer import GeminiAnalyzer
from src.config.config import BASE_DIR, UPLOAD_DIR, THUMBNAIL_DIR, METADATA_DIR
from src.config.prompts import IMAGE_SIZE, THUMBNAIL_QUALITY

# Configure logging
logger = logging.getLogger(__name__)

class PatternAnalyzer:
    """
    Core service for analyzing textile patterns and managing metadata.
    Focused on image processing, pattern recognition, and storage management.
    """
    
    def __init__(self, api_key=None):
        """Initialize the analyzer with the Gemini API key"""
        logger.info("Initializing PatternAnalyzer...")
        try:
            # Ensure directories exist
            Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
            Path(THUMBNAIL_DIR).mkdir(parents=True, exist_ok=True)
            Path(METADATA_DIR).mkdir(parents=True, exist_ok=True)
            
            # Initialize Gemini analyzer
            if api_key:
                masked_key = self._mask_api_key(api_key)
                logger.info(f"Using Gemini API key: {masked_key}")
            
            # Initialize the Gemini analyzer
            self.gemini_analyzer = GeminiAnalyzer(api_key=api_key)
            
            # Set metadata path
            self.metadata_file = Path(METADATA_DIR) / "patterns_metadata.json"
            
            # Load existing metadata if available
            self.metadata = self.load_metadata()
            
            # Track last metadata load time
            self._last_load_time = time.time()
            
            logger.info("PatternAnalyzer initialization completed")
        except Exception as e:
            logger.error(f"Error initializing PatternAnalyzer: {e}")
            raise

    def _mask_api_key(self, key):
        """Mask the API key for logging purposes"""
        if not key or len(key) < 8:
            return "INVALID_KEY"
        return f"{key[:4]}...{key[-4:]}"

    def load_metadata(self) -> Dict:
        """Load metadata from file"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                logger.info(f"Loaded {len(metadata)} metadata entries")
                return metadata
            logger.info("No metadata file found, starting with empty metadata")
            return {}
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return {}

    def save_metadata(self) -> bool:
        """Save metadata to file"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2)
            logger.info("Metadata saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
            return False

    def set_api_key(self, api_key: str):
        """Set or update the Gemini API key"""
        if api_key:
            masked_key = self._mask_api_key(api_key)
            logger.info(f"Updating Gemini API key: {masked_key}")
            self.gemini_analyzer.set_api_key(api_key)
            logger.info("Gemini API key updated")
        else:
            logger.warning("Attempted to set empty API key")

    def process_image(self, image_path: Path) -> Dict[str, Any]:
        """
        Process an image to extract pattern analysis and metadata
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing analysis results and metadata
        """
        try:
            logger.info(f"Processing image: {image_path}")
            
            # Check if file exists
            if not isinstance(image_path, Path):
                image_path = Path(image_path)
                
            if not image_path.exists():
                logger.error(f"Image file not found: {image_path}")
                return None
            
            # Create thumbnail
            thumbnail_path = self.create_thumbnail(image_path)
            if not thumbnail_path:
                logger.error(f"Failed to create thumbnail for: {image_path}")
                return None
            
            # Get relative paths for storage
            rel_image_path = str(image_path.relative_to(UPLOAD_DIR)) if str(UPLOAD_DIR) in str(image_path) else image_path.name
            rel_thumbnail_path = str(thumbnail_path.relative_to(THUMBNAIL_DIR)) if str(THUMBNAIL_DIR) in str(thumbnail_path) else thumbnail_path.name
            
            # Analyze the image with Gemini
            pattern_info = self.gemini_analyzer.analyze_image(str(image_path))
            
            # Generate metadata
            metadata = {
                'id': str(image_path.stem),
                'filename': image_path.name,
                'path': rel_image_path,
                'thumbnail_path': rel_thumbnail_path,
                'patterns': pattern_info,
                'timestamp': time.time()
            }
            
            # Store metadata
            self.metadata[rel_image_path] = metadata
            self.save_metadata()
            
            logger.info(f"Image processed successfully: {image_path}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error processing image: {e}", exc_info=True)
            return None

    def create_thumbnail(self, image_path: Path) -> Path:
        """
        Create a thumbnail for an image
        
        Args:
            image_path: Path to the original image
            
        Returns:
            Path to the created thumbnail
        """
        try:
            # Create thumbnail directory if it doesn't exist
            Path(THUMBNAIL_DIR).mkdir(parents=True, exist_ok=True)
            
            # Open the image
            image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Use the centralized image size configuration
            thumbnail_size = IMAGE_SIZE
            
            # Resize to thumbnail size
            image.thumbnail((thumbnail_size, thumbnail_size))
            
            # Create thumbnail path
            thumbnail_path = Path(THUMBNAIL_DIR) / image_path.name
            
            # Save thumbnail with configured quality
            image.save(thumbnail_path, quality=THUMBNAIL_QUALITY)
            
            return thumbnail_path
        except Exception as e:
            logger.error(f"Error creating thumbnail: {e}")
            return None

    def get_image_metadata(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific image
        
        Args:
            image_path: Path or filename of the image
            
        Returns:
            Metadata for the image or None if not found
        """
        # Ensure metadata is loaded
        self._ensure_metadata_loaded()
        
        # Try to get by exact path first
        if image_path in self.metadata:
            return self.metadata[image_path]
        
        # Try to get by filename
        filename = Path(image_path).name
        for path, metadata in self.metadata.items():
            if metadata.get('filename') == filename:
                return metadata
        
        logger.warning(f"No metadata found for image: {image_path}")
        return None
    
    def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all image metadata
        
        Returns:
            Dictionary of all image metadata
        """
        # Ensure metadata is loaded
        self._ensure_metadata_loaded()
        return self.metadata
        
    def _ensure_metadata_loaded(self, max_age_seconds: int = 60) -> None:
        """Reload metadata if it's older than max_age_seconds"""
        current_time = time.time()
        if current_time - self._last_load_time > max_age_seconds:
            self.metadata = self.load_metadata()
            self._last_load_time = time.time()
            
    def update_metadata(self, path: str, updated_metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for an image
        
        Args:
            path: Image path
            updated_metadata: Updated metadata
            
        Returns:
            bool: Success status
        """
        # Ensure metadata is loaded
        self._ensure_metadata_loaded()
        
        try:
            if path in self.metadata:
                # Update existing metadata
                self.metadata[path].update(updated_metadata)
                
                # Save to file
                return self.save_metadata()
                    
            # Try to find by filename
            filename = Path(path).name
            for cached_path in self.metadata:
                if Path(cached_path).name == filename:
                    # Update existing metadata
                    self.metadata[cached_path].update(updated_metadata)
                    
                    # Save to file
                    return self.save_metadata()
                    
            logger.warning(f"No metadata found for {path}")
            return False
        except Exception as e:
            logger.error(f"Error updating metadata: {e}")
            return False
    
    def delete_metadata(self, path: str) -> bool:
        """
        Delete metadata for an image
        
        Args:
            path: Image path
            
        Returns:
            bool: Success status
        """
        try:
            # Remove from metadata
            if path in self.metadata:
                del self.metadata[path]
                return self.save_metadata()
                
            # Try to find by filename
            filename = Path(path).name
            for cached_path in list(self.metadata.keys()):
                if Path(cached_path).name == filename:
                    del self.metadata[cached_path]
                    return self.save_metadata()
                    
            logger.warning(f"No metadata found for {path}")
            return False
        except Exception as e:
            logger.error(f"Error deleting metadata: {e}")
            return False 