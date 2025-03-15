from pathlib import Path
from typing import List, Dict, Any
import torch
from transformers import CLIPProcessor, CLIPModel
import json
from PIL import Image
import numpy as np
import logging
from sklearn.cluster import KMeans
import colorsys

# Relative imports from the same package
from .analyzers.pattern_analyzer import PatternAnalyzer
from .analyzers.color_analyzer import ColorAnalyzer
from .search.searcher import ImageSearcher
import config
from .prompt_builder import PromptBuilder
from .embedding_extractor import EmbeddingExtractor
from .config_prompts import THRESHOLDS
from .text_refiner import refine_text

# Sadece önemli logları göster
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Werkzeug loglarını kapat
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.setLevel(logging.ERROR)

class SearchEngine:
    def __init__(self):
        logger.info("Initializing SearchEngine...")
        try:
            # Load models and move to GPU if available
            logger.info("Loading CLIP model: openai/clip-vit-large-patch14")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
            logger.info("CLIP model loaded successfully")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            logger.info("CLIP processor loaded successfully")
            
            # Initialize analyzers
            self.pattern_analyzer = PatternAnalyzer(self.model, self.processor, self.device)
            self.color_analyzer = ColorAnalyzer(max_clusters=config.N_CLUSTERS)
            self.searcher = ImageSearcher(self.model, self.processor, self.device)
            
            # Initialize embedding extractor
            self.embedding_extractor = EmbeddingExtractor(self.model, self.processor, self.device)
            
            # Initialize other components
            self.metadata_file = config.BASE_DIR / "metadata.json"
            self.metadata = self.load_metadata()
            
            # Add max_clusters attribute
            self.max_clusters = config.N_CLUSTERS
            
            logger.info("SearchEngine initialization completed")
        except Exception as e:
            logger.error(f"Error initializing SearchEngine: {str(e)}")
            logger.error("Stack trace:")
            import traceback
            traceback.print_exc()
            raise

    def load_metadata(self) -> Dict:
        """Load metadata from file if it exists."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def save_metadata(self):
        """Save metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f)

    def detect_patterns(self, image_path: Path) -> Dict:
        """Detect patterns in an image and generate a detailed description."""
        try:
            # Load and preprocess the image
            image_features = self.embedding_extractor.extract_features(image_path)
            
            # Analyze patterns
            pattern_info = self.pattern_analyzer._analyze_patterns(image_features)
            
            # Add detailed style analysis
            style_info = self._analyze_detailed_style(image_features, pattern_info)
            pattern_info['style'] = style_info  # Add style results to pattern_info
            
            # Load image for color analysis
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image_array = np.array(image)
            
            # Analyze colors with the image array
            color_info = self.color_analyzer.analyze_colors(image_array)
            
            # Generate detailed prompt
            prompt_data = self.generate_detailed_prompt(image_features, pattern_info, color_info)
            
            return {
                "pattern_info": pattern_info,
                "color_info": color_info,
                "prompt": prompt_data
            }
        except Exception as e:
            logger.error(f"Error detecting patterns: {str(e)}")
            return {"error": str(e)}

    def search(self, query: str, k: int = 10) -> List[Dict]:
        """Search for images based on query."""
        return self.searcher.search(query, list(self.metadata.values()), k)

    def _get_default_pattern_info(self) -> Dict:
        """Return default pattern info when analysis fails."""
        return {
            'category': 'Unknown',
            'category_confidence': 0.0,
            'primary_pattern': 'Unknown',
            'pattern_confidence': 0.0,
            'secondary_patterns': [],
            'layout': {'type': 'balanced', 'confidence': 0.0},
            'repeat_type': {'type': 'regular', 'confidence': 0.0},
            'scale': {'type': 'medium', 'confidence': 0.0},
            'texture_type': {'type': 'smooth', 'confidence': 0.0},
            'cultural_influence': {'type': 'contemporary', 'confidence': 0.0},
            'historical_period': {'type': 'modern', 'confidence': 0.0},
            'mood': {'type': 'balanced', 'confidence': 0.0},
            'style_keywords': ['balanced'],
            'prompt': {
                'final_prompt': 'Unable to analyze pattern',
                'components': {},
                'completeness_score': 0
            }
        }

    def _analyze_patterns(self, image_features) -> Dict:
        """Analyze patterns in the image features."""
        try:
            logger.info("Starting pattern analysis...")
            
            # Basic pattern categories
            pattern_categories = [
                "geometric", "floral", "abstract", "stripes", "polka dots", 
                "chevron", "paisley", "plaid", "animal print", "tribal",
                "damask", "herringbone", "houndstooth", "ikat", "lattice",
                "medallion", "moroccan", "ogee", "quatrefoil", "trellis"
            ]
            
            # Analyze pattern categories
            scores = {}
            with torch.no_grad():
                for category in pattern_categories:
                    text_inputs = self.processor(
                        text=[f"this is a {category} pattern"],
                        return_tensors="pt",
                        padding=True
                    ).to(self.device)
                    
                    text_features = self.model.get_text_features(**text_inputs)
                    similarity = torch.nn.functional.cosine_similarity(
                        image_features.to(self.device),
                        text_features,
                        dim=1
                    )
                    scores[category] = float(similarity[0].cpu())

            # Find highest scoring category
            primary_pattern = max(scores.items(), key=lambda x: x[1])
            
            # Find secondary patterns (above threshold)
            threshold = 0.2
            secondary_patterns = [
                {"name": pattern, "confidence": score}
                for pattern, score in scores.items()
                if score > threshold and pattern != primary_pattern[0]
            ]
            
            # Return results
            return {
                "category": primary_pattern[0],
                "category_confidence": float(primary_pattern[1]),
                "secondary_patterns": sorted(secondary_patterns, 
                                          key=lambda x: x["confidence"], 
                                          reverse=True)[:3]  # Top 3 secondary patterns
            }

        except Exception as e:
            logger.error(f"Error in pattern analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "category": "Unknown",
                "category_confidence": 0.0,
                "secondary_patterns": []
            }

    def _analyze_detailed_style(self, image_features, pattern_info):
        """Analyze detailed style aspects of the pattern with improved accuracy."""
        try:
            pattern_type = pattern_info.get('category', 'unknown')
            
            aspects = {
                'layout': [
                    'scattered', 'aligned', 'symmetrical', 'repeating', 
                    'random', 'clustered', 'all-over', 'border', 'trailing'
                ],
                'scale': [
                    'large', 'medium', 'small', 'varied', 'proportional', 
                    'oversized', 'miniature', 'mixed'
                ],
                'texture_type': [
                    'smooth', 'textured', 'raised', 'flat', 'embossed', 
                    'dimensional', 'woven', 'printed'
                ]
            }
            
            results = {}
            
            for aspect, attributes in aspects.items():
                logger.info(f"Analyzing {aspect} for pattern type: {pattern_type}")
                
                best_attribute = None
                best_score = 0.0
                all_scores = {}  # Store all scores for logging
                
                for attr in attributes:
                    phrasings = [
                        f"a {pattern_type} pattern with {attr} {aspect}",
                        f"{attr} {aspect} in a {pattern_type} pattern",
                        f"textile with {attr} {aspect}",
                        f"{pattern_type} design with {attr} {aspect}"
                    ]
                    
                    attr_score = 0.0
                    valid_checks = 0
                    phrase_scores = []  # Store individual phrase scores
                    
                    with torch.no_grad():
                        for phrase in phrasings:
                            text_inputs = self.processor(
                                text=[phrase],
                                return_tensors="pt",
                                padding=True
                            ).to(self.device)
                            
                            text_features = self.model.get_text_features(**text_inputs)
                            similarity = torch.nn.functional.cosine_similarity(
                                image_features.to(self.device),
                                text_features,
                                dim=1
                            )
                            score = float(similarity[0].cpu())
                            phrase_scores.append((phrase, score))
                            
                            # Use threshold from config
                            aspect_key = aspect.split('_')[0]  # Handle texture_type -> texture
                            threshold = THRESHOLDS.get(aspect_key, 0.2)
                            
                            if score > threshold:
                                attr_score += score
                                valid_checks += 1
                    
                    # Log all phrase scores for debugging
                    logger.debug(f"Phrase scores for {attr}: {phrase_scores}")
                    
                    if valid_checks > 0:
                        avg_score = attr_score / valid_checks
                        all_scores[attr] = avg_score
                        logger.info(f"Average score for {attr}: {avg_score:.4f}")
                        if avg_score > best_score:
                            best_score = avg_score
                            best_attribute = attr
                            logger.info(f"New best attribute for {aspect}: {attr} (score: {avg_score:.4f})")

                # Log all attribute scores for this aspect
                logger.debug(f"All {aspect} scores: {all_scores}")
                
                results[aspect] = {
                    'type': best_attribute or 'balanced',
                    'confidence': best_score,
                    'all_scores': all_scores  # Include all scores for potential UI display
                }
                logger.info(f"Final {aspect} result: {best_attribute or 'balanced'} (confidence: {best_score:.4f})")

            return results

        except Exception as e:
            logger.error(f"Error in _analyze_detailed_style: {str(e)}")
            logger.error("Stack trace:")
            import traceback
            traceback.print_exc()
            return {}

    def generate_detailed_prompt(self, image_features, pattern_info, color_info):
        """Generate precise, descriptive prompts using the PromptBuilder."""
        try:
            # Initialize the prompt builder
            prompt_builder = PromptBuilder()
            
            # Extract style info if available
            style_info = pattern_info.get('style', {})
            
            # Build the prompt with style information
            prompt_result = prompt_builder.build_prompt(pattern_info, color_info, style_info)
            
            # Refine the prompt text
            final_prompt = prompt_result.get("final_prompt", "")
            refined_prompt = refine_text(final_prompt)
            prompt_result["final_prompt"] = refined_prompt
            
            return prompt_result
        
        except Exception as e:
            logger.error(f"Error generating detailed prompt: {str(e)}")
            return {
                "final_prompt": "Unable to generate detailed prompt",
                "components": {},
                "completeness_score": 0
            }

    def get_color_name(self, hsv):
        """Get color name from HSV values."""
        h, s, v = hsv
        
        if s < 0.1 and v > 0.9:
            return "White"
        if s < 0.1 and v < 0.1:
            return "Black"
        if s < 0.1:
            return "Gray"

        h *= 360
        if h < 30:
            return "Red"
        elif h < 90:
            return "Yellow"
        elif h < 150:
            return "Green"
        elif h < 210:
            return "Cyan"
        elif h < 270:
            return "Blue"
        elif h < 330:
            return "Magenta"
        else:
            return "Red"

    def analyze_colors(self, image: np.ndarray) -> Dict:
        """Enhanced color analysis with better color naming."""
        try:
            # Convert image to RGB if it's not
            if len(image.shape) == 2:  # Grayscale
                image = np.stack((image,) * 3, axis=-1)
            elif image.shape[2] == 4:  # RGBA
                image = image[:, :, :3]

            # Reshape image for clustering
            pixels = image.reshape(-1, 3)
            
            # Determine optimal number of clusters
            n_colors = min(max(3, len(np.unique(pixels, axis=0)) // 100), self.max_clusters)
            
            # Perform k-means clustering
            kmeans = KMeans(n_clusters=n_colors, random_state=42)
            kmeans.fit(pixels)
            
            # Get cluster centers and their proportions
            colors = kmeans.cluster_centers_
            labels = kmeans.labels_
            
            # Calculate color proportions
            unique_labels, counts = np.unique(labels, return_counts=True)
            proportions = counts / len(labels)

            # Define basic color references with their RGB values
            color_references = {
                'Red': (255, 0, 0),
                'Dark Red': (139, 0, 0),
                'Pink': (255, 192, 203),
                'Orange': (255, 165, 0),
                'Yellow': (255, 255, 0),
                'Green': (0, 128, 0),
                'Light Green': (144, 238, 144),
                'Blue': (0, 0, 255),
                'Light Blue': (173, 216, 230),
                'Purple': (128, 0, 128),
                'Brown': (165, 42, 42),
                'Gray': (128, 128, 128),
                'Light Gray': (211, 211, 211),
                'Black': (0, 0, 0),
                'White': (255, 255, 255),
                'Beige': (245, 245, 220),
                'Navy': (0, 0, 128),
                'Teal': (0, 128, 128),
                'Maroon': (128, 0, 0),
                'Gold': (255, 215, 0)
            }

            # Function to convert RGB to HSV
            def rgb_to_hsv(rgb):
                return colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)

            # Function to find closest color name
            def get_color_name(rgb):
                min_distance = float('inf')
                closest_color = 'Unknown'
                
                # Convert target color to HSV
                hsv = rgb_to_hsv(rgb)
                
                for name, ref_rgb in color_references.items():
                    # Convert reference color to HSV
                    ref_hsv = rgb_to_hsv(ref_rgb)
                    
                    # Calculate distance in HSV space with weighted components
                    h_diff = min(abs(hsv[0] - ref_hsv[0]), 1 - abs(hsv[0] - ref_hsv[0])) * 2.0
                    s_diff = abs(hsv[1] - ref_hsv[1])
                    v_diff = abs(hsv[2] - ref_hsv[2])
                    
                    distance = (h_diff * 2) + (s_diff * 1) + (v_diff * 1)
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_color = name

                return closest_color

            # Process each dominant color
            dominant_colors = []
            for color, proportion in zip(colors, proportions):
                rgb = color.astype(int)
                hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb)
                color_name = get_color_name(rgb)
                
                dominant_colors.append({
                    'rgb': rgb.tolist(),
                    'hex': hex_color,
                    'name': color_name,
                    'proportion': float(proportion)
                })

            # Sort colors by proportion
            dominant_colors.sort(key=lambda x: x['proportion'], reverse=True)

            # Calculate overall brightness
            brightness = np.mean(image) / 255.0

            # Calculate color contrast
            contrast = np.std(image) / 255.0

            return {
                'dominant_colors': dominant_colors,
                'overall_brightness': float(brightness),
                'color_contrast': float(contrast)
            }

        except Exception as e:
            logger.error(f"Error in analyze_colors: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def create_thumbnail(self, image_path: Path) -> Path:
        """Create a thumbnail for the image."""
        try:
            from PIL import Image
            
            # Open image
            img = Image.open(image_path)
            
            # Calculate new dimensions
            aspect_ratio = img.width / img.height
            new_width = min(400, img.width)
            new_height = int(new_width / aspect_ratio)
            
            # Resize image
            thumbnail = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Save thumbnail
            thumbnail_path = config.THUMBNAIL_DIR / image_path.name
            thumbnail.save(thumbnail_path, 'JPEG', quality=85)
            
            return thumbnail_path
            
        except Exception as e:
            logger.error(f"Error creating thumbnail: {str(e)}")
            return None

    def process_images(self, image_paths: List[Path]) -> List[Dict]:
        """Process multiple images and return their metadata."""
        results = []
        for path in image_paths:
            try:
                metadata = self.process_image(path)
                if metadata:
                    results.append(metadata)
            except Exception as e:
                logger.error(f"Error processing {path}: {str(e)}")
                continue
        return results

    def process_image(self, image_path: Path) -> Dict[str, Any]:
        """
        Process a single image and extract its patterns and metadata.
        
        Args:
            image_path (Path): Path to the image file
            
        Returns:
            Dict[str, Any]: Processed image metadata including patterns and colors
        """
        try:
            logger.info(f"\n=== Processing image: {image_path} ===")
            
            # Check if image exists
            if not image_path.exists():
                logger.error(f"Error: Image file not found at {image_path}")
                return None
                
            # Create thumbnail
            thumbnail_path = self.create_thumbnail(image_path)
            if not thumbnail_path:
                logger.error(f"Error: Failed to create thumbnail for {image_path}")
                return None
                
            # Open and process the image
            try:
                image = Image.open(image_path)
                # Convert to RGB if needed (for PNG with transparency)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
            except Exception as e:
                logger.error(f"Error opening image: {str(e)}")
                return None
                
            # Extract patterns
            pattern_data = self.detect_patterns(image_path)
            if not pattern_data or "error" in pattern_data:
                logger.warning(f"Warning: No patterns detected in {image_path}")
                pattern_data = {
                    "pattern_info": {
                        "category": "Unknown",
                        "category_confidence": 0.0
                    },
                    "prompt": {"final_prompt": "No pattern detected"}
                }
            
            # Extract the pattern info and prompt from the pattern_data
            pattern_info = pattern_data.get("pattern_info", {})
            prompt_data = pattern_data.get("prompt", {"final_prompt": "No pattern detected"})
            color_info = pattern_data.get("color_info", {})
            
            # Get the final prompt text
            prompt_text = prompt_data.get("final_prompt", "No pattern detected")
            
            # Create metadata with the structure expected by the frontend
            metadata = {
                "original_path": str(image_path),
                "thumbnail_path": str(thumbnail_path.name),
                "patterns": {
                    "primary_pattern": pattern_info.get("category", "Unknown"),
                    "confidence": pattern_info.get("category_confidence", 0.0),
                    "elements": [e["name"] for e in pattern_info.get("elements", [])[:3]],
                    "style": pattern_info.get("style", {}),
                    "prompt": prompt_text,
                    "secondary_patterns": pattern_info.get("secondary_patterns", [])
                },
                "colors": color_info,
                "timestamp": str(Path(image_path).stat().st_mtime)
            }
            
            # Log the prompt for debugging
            logger.info(f"Generated prompt: {prompt_text}")
            
            # Save to metadata store
            self.metadata[str(image_path)] = metadata
            self.save_metadata()
            
            logger.info(f"Successfully processed image: {image_path}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _analyze_style(self, image_features) -> Dict:
        """Analyze style attributes of the pattern."""
        style_types = ["traditional", "modern", "abstract", "naturalistic", 
                      "geometric", "organic", "minimalist", "ornate"]
        return self._analyze_attributes(image_features, style_types)

    def _analyze_layout(self, image_features) -> Dict:
        """Analyze layout of the pattern."""
        layout_types = ["regular", "random", "directional", "symmetrical", 
                       "all-over", "border", "central", "scattered"]
        return self._analyze_attributes(image_features, layout_types)

    def _analyze_repetition_type(self, image_features) -> Dict:
        """Analyze pattern repetition type."""
        repeat_types = ["block repeat", "half-drop", "mirror repeat", 
                       "brick repeat", "diamond repeat", "ogee repeat"]
        return self._analyze_attributes(image_features, repeat_types)

    def _analyze_scale(self, image_features) -> Dict:
        """Analyze scale of the pattern."""
        scale_types = ["small-scale", "medium-scale", "large-scale", 
                      "multi-scale", "micro pattern", "oversized"]
        return self._analyze_attributes(image_features, scale_types)

    def _analyze_texture(self, image_features) -> Dict:
        """Analyze texture characteristics."""
        texture_types = ["smooth", "textured", "dimensional", "flat", 
                        "embossed", "woven", "printed"]
        return self._analyze_attributes(image_features, texture_types)

    def _analyze_cultural_context(self, image_features) -> Dict:
        """Analyze cultural influences."""
        cultural_styles = ["western", "eastern", "tribal", "contemporary", 
                         "traditional", "fusion", "ethnic"]
        return self._analyze_attributes(image_features, cultural_styles)

    def _analyze_historical_context(self, image_features) -> Dict:
        """Analyze historical context."""
        historical_periods = ["classical", "modern", "contemporary", 
                            "vintage", "retro", "timeless"]
        return self._analyze_attributes(image_features, historical_periods)

    def _suggest_use_cases(self, image_features) -> Dict:
        """Suggest potential use cases."""
        use_cases = ["apparel", "upholstery", "accessories", 
                    "home decor", "wallcovering", "technical"]
        return self._analyze_attributes(image_features, use_cases)

    def _suggest_applications(self, image_features) -> Dict:
        """Suggest specific applications."""
        applications = ["fashion", "interior design", "textile art", 
                       "commercial", "residential", "industrial"]
        return self._analyze_attributes(image_features, applications)

    def _analyze_mood(self, image_features) -> Dict:
        """Analyze emotional mood."""
        moods = ["elegant", "playful", "sophisticated", "casual", 
                "dramatic", "serene", "energetic"]
        return self._analyze_attributes(image_features, moods)

    def _extract_style_keywords(self, image_features) -> List[str]:
        """Extract key style descriptors."""
        style_words = ["refined", "bold", "delicate", "dynamic", 
                      "balanced", "harmonious", "striking"]
        results = self._analyze_attributes(image_features, style_words)
        return [results['type']] if results else []

    def _analyze_attributes(self, features, attributes) -> Dict:
        """Generic method to analyze pattern attributes."""
        try:
            scores = {}
            for attr in attributes:
                text_inputs = self.processor(
                    text=[f"this pattern is {attr}", f"this pattern is not {attr}"],
                    return_tensors="pt",
                    padding=True
                ).to(self.device)  # Move text inputs to GPU
                
                with torch.no_grad():
                    text_features = self.model.get_text_features(**text_inputs)
                    similarity = torch.nn.functional.cosine_similarity(
                        features.to(self.device),  # Move features to GPU
                        text_features,
                        dim=1
                    )
                    scores[attr] = float(similarity[0].cpu())  # Move result back to CPU

            # Get the most likely attribute
            top_attr = max(scores.items(), key=lambda x: x[1])
            return {
                'type': top_attr[0],
                'confidence': top_attr[1]
            }

        except Exception as e:
            logger.error(f"Error in _analyze_attributes: {str(e)}")
            return {
                'type': 'unknown',
                'confidence': 0.0
            }

    def _get_attributes_for_aspect(self, aspect):
        """Get possible attributes for a given aspect."""
        logger.info(f"\nGetting attributes for aspect: {aspect}")
        attributes = {
            'layout': ["balanced", "asymmetric", "radial", "linear", "scattered", "grid-like", "concentric"],
            'scale': ["small", "medium", "large", "varied", "proportional", "intricate", "bold"],
            'texture': ["smooth", "rough", "layered", "flat", "dimensional", "textured", "embossed"]
        }
        result = attributes.get(aspect, ["balanced"])
        logger.info(f"Available attributes: {result}")
        return result 