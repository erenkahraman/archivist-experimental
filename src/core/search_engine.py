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
from src.analyzers.color_analyzer import ColorAnalyzer
from src.analyzers.gemini_analyzer import GeminiAnalyzer
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
        """
        Advanced search for images based on pattern, color, and other metadata.
        Supports compound queries with comma-separated terms like 'red paisley, flower'
        as well as space-separated terms like 'red paisley'.
        Results are sorted by relevance to the query.
        
        Args:
            query: Search query string (can include commas to separate distinct terms)
            k: Maximum number of results to return
            
        Returns:
            List of image metadata dictionaries with similarity scores
        """
        try:
            if not self.metadata:
                logger.info("No metadata available for search")
                return []

            # Parse the query into components
            query = query.lower().strip()
            
            # Split on commas first to get distinct search "phrases"
            query_phrases = [phrase.strip() for phrase in query.split(',')]
            
            # Then split each phrase into individual terms
            all_query_terms = []
            for phrase in query_phrases:
                all_query_terms.extend(phrase.split())
            
            # Remove duplicates while preserving order
            query_terms = []
            for term in all_query_terms:
                if term not in query_terms:
                    query_terms.append(term)
            
            # Common color names to help with color matching
            color_names = [
                "red", "blue", "green", "yellow", "orange", "purple", "pink", 
                "brown", "black", "white", "gray", "grey", "teal", "turquoise", 
                "gold", "silver", "bronze", "maroon", "navy", "olive", "mint",
                "cyan", "magenta", "lavender", "violet", "indigo", "coral", "peach"
            ]
            
            # Common pattern types
            pattern_types = [
                "paisley", "floral", "geometric", "abstract", "animal", "stripe", 
                "dots", "plaid", "check", "chevron", "herringbone", "tropical", 
                "diamond", "swirl", "damask", "toile", "ikat", "medallion", "flower"
            ]
            
            # Extract color and pattern terms from the query
            color_terms = [term for term in query_terms if term in color_names]
            pattern_terms = [term for term in query_terms if term in pattern_types]
            
            # Also check each phrase for pattern matches that might be multi-word
            for phrase in query_phrases:
                phrase = phrase.strip()
                for pattern_type in pattern_types:
                    if pattern_type in phrase and pattern_type not in pattern_terms:
                        pattern_terms.append(pattern_type)
                for color_name in color_names:
                    if color_name in phrase and color_name not in color_terms:
                        color_terms.append(color_name)
            
            # Any terms that aren't colors or patterns
            other_terms = [term for term in query_terms 
                          if term not in color_names 
                          and term not in pattern_types]
            
            # Log what we're searching for to debug
            logger.info(f"Searching for patterns: {pattern_terms}, colors: {color_terms}, other: {other_terms}")
            
            # If no specific search terms were identified, treat all terms as general
            if not pattern_terms and not color_terms and not other_terms:
                # Try the full phrases first
                other_terms = query_phrases
                
                # Also treat individual words as search terms
                for term in query_terms:
                    if term not in other_terms:
                        other_terms.append(term)
                        
                # Log that we're using generic search terms
                logger.info(f"No specific pattern/color terms identified. Using general terms: {other_terms}")
                
            # Filter results based on color/pattern/other terms
            scored_results = []
            
            for path, metadata in self.metadata.items():
                # Initialize score components
                color_score = 0.0
                pattern_score = 0.0
                other_score = 0.0
                confidence_boost = 0.0
                proportion_boost = 0.0
                
                # PATTERN MATCHING
                if 'patterns' in metadata and metadata['patterns']:
                    patterns = metadata['patterns']
                    
                    # Primary pattern match is weighted heavily
                    primary_pattern = patterns.get('primary_pattern', '').lower()
                    if primary_pattern and primary_pattern != 'unknown':
                        primary_confidence = patterns.get('pattern_confidence', 0.5)
                        
                        # Check for exact primary pattern matches
                        for pattern_term in pattern_terms:
                            if pattern_term == primary_pattern:
                                # Exact match with high confidence
                                pattern_score += 10.0 * primary_confidence
                            elif pattern_term in primary_pattern or primary_pattern in pattern_term:
                                # Partial match
                                pattern_score += 5.0 * primary_confidence
                    
                    # Check secondary patterns
                    secondary_patterns = patterns.get('secondary_patterns', [])
                    for secondary in secondary_patterns:
                        sec_name = secondary.get('name', '').lower()
                        sec_confidence = secondary.get('confidence', 0.5)
                        
                        for pattern_term in pattern_terms:
                            if pattern_term == sec_name:
                                # Exact match in secondary pattern
                                pattern_score += 3.0 * sec_confidence
                            elif pattern_term in sec_name or sec_name in pattern_term:
                                # Partial match in secondary pattern
                                pattern_score += 1.5 * sec_confidence
                    
                    # Check elements for pattern matches
                    elements = patterns.get('elements', [])
                    for element in elements:
                        # Handle case where element might be a string instead of a dictionary
                        if isinstance(element, dict):
                            element_name = element.get('name', '').lower()
                            element_confidence = element.get('confidence', 0.5)
                        else:
                            element_name = str(element).lower()
                            element_confidence = 0.5
                        
                        for pattern_term in pattern_terms:
                            if pattern_term in element_name:
                                pattern_score += 2.0 * element_confidence
                    
                    # Check the final prompt for pattern terms
                    if 'prompt' in patterns and 'final_prompt' in patterns['prompt']:
                        prompt_text = patterns['prompt']['final_prompt'].lower()
                        for pattern_term in pattern_terms:
                            if pattern_term in prompt_text:
                                pattern_score += 0.5
                
                # COLOR MATCHING
                if 'colors' in metadata and metadata['colors']:
                    colors = metadata['colors']
                    dominant_colors = colors.get('dominant_colors', [])
                    
                    for color_data in dominant_colors:
                        color_name = color_data.get('name', '').lower()
                        # Higher weight for colors with higher proportion
                        proportion = color_data.get('proportion', 0.1)
                        
                        for color_term in color_terms:
                            if color_term == color_name or color_term in color_name:
                                # Weight by the color's proportion in the image
                                color_score += 5.0 * proportion
                                proportion_boost += proportion
                    
                    # Also check shades for more subtle color matches
                    for color_data in dominant_colors:
                        shades = color_data.get('shades', [])
                        for shade in shades:
                            shade_name = shade.get('name', '').lower()
                            for color_term in color_terms:
                                if color_term in shade_name:
                                    color_score += 1.0 * color_data.get('proportion', 0.1)
                
                # OTHER TERMS MATCHING (for any other descriptive terms)
                if 'patterns' in metadata and metadata['patterns']:
                    # Check mood, cultural influence, style keywords, etc
                    patterns = metadata['patterns']
                    
                    # Check style keywords
                    style_keywords = patterns.get('style_keywords', [])
                    for keyword in style_keywords:
                        keyword = keyword.lower()
                        for term in other_terms:
                            # Check if the term is a full phrase that appears in the keyword
                            # or if the keyword contains the term
                            if (term.strip() and (term == keyword or term in keyword)):
                                other_score += 2.0
                    
                    # Check cultural influence
                    if 'cultural_influence' in patterns:
                        cultural = patterns['cultural_influence'].get('type', '').lower()
                        cultural_conf = patterns['cultural_influence'].get('confidence', 0.5)
                        for term in other_terms:
                            if term.strip() and term in cultural:
                                other_score += 1.5 * cultural_conf
                    
                    # Check mood
                    if 'mood' in patterns:
                        mood = patterns['mood'].get('type', '').lower()
                        mood_conf = patterns['mood'].get('confidence', 0.5)
                        for term in other_terms:
                            if term.strip() and term in mood:
                                other_score += 1.5 * mood_conf
                    
                    # Check the prompt text for any other terms - this is important for general searches
                    if 'prompt' in patterns and 'final_prompt' in patterns['prompt']:
                        prompt_text = patterns['prompt']['final_prompt'].lower()
                        for term in other_terms:
                            if term.strip() and term in prompt_text:
                                # Give higher weight to full phrase matches in the prompt
                                if len(term.split()) > 1:
                                    other_score += 3.0
                                else:
                                    other_score += 1.0
                
                # Calculate final score (weighted sum)
                final_score = 0.0
                
                # Weight pattern matches most heavily when pattern terms are present
                if pattern_terms:
                    final_score += pattern_score * 2.0
                
                # Weight color matches heavily when color terms are present 
                if color_terms:
                    final_score += color_score * 1.5
                
                # Add other terms score
                final_score += other_score
                
                # Normalize score based on number of query terms to avoid bias toward longer queries
                if query_terms:  # Prevent division by zero
                    final_score = final_score / (len(query_terms) * 3)  # Normalize to roughly 0-1 range
                
                # Only include results with non-zero scores
                if final_score > 0:
                    scored_results.append({
                        **metadata, 
                        'similarity': min(final_score, 1.0),  # Cap at 1.0
                        'pattern_score': pattern_score,
                        'color_score': color_score,
                        'other_score': other_score
                    })
            
            # Sort results by similarity score (descending)
            scored_results.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Log search results for debugging
            logger.info(f"Search for '{query}' found {len(scored_results)} results")
            
            return scored_results[:k]  # Return top k results
            
        except Exception as e:
            logger.error(f"Error in search: {e}", exc_info=True)
            return []

    def get_image_metadata(self, image_path: str) -> Dict[str, Any]:
        """Get metadata for a specific image."""
        try:
            # Normalize the path format to match what's stored in metadata
            norm_path = str(image_path).replace("\\", "/")
            
            # Try direct match first
            if norm_path in self.metadata:
                return self.metadata[norm_path]
            
            # Try with just the filename
            filename = os.path.basename(norm_path)
            for path, meta in self.metadata.items():
                if os.path.basename(path) == filename:
                    return meta
            
            # Try to match with relative path
            for path, meta in self.metadata.items():
                if path.endswith(norm_path) or norm_path.endswith(path):
                    return meta
            
            logger.warning(f"No metadata found for image: {image_path}")
            return {}
            
        except Exception as e:
            logger.error(f"Error getting image metadata: {e}")
            return {} 