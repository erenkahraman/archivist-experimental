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

# Only show important logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Disable Werkzeug logs
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

    def detect_patterns(self, image) -> Dict:
        """Detect patterns in the image."""
        try:
            logger.info("\n=== Starting pattern detection ===")
            
            # Process image through CLIP
            inputs = self.processor(
                images=image,
                return_tensors="pt",
                padding=True
            )
            
            with torch.no_grad():
                image_features = self.model.get_image_features(
                    pixel_values=inputs['pixel_values'].to(self.device)
                )
                
                # Analyze patterns using pattern_analyzer
                pattern_info = self.pattern_analyzer._analyze_patterns(image_features)
                
                # Analyze colors using color_analyzer
                color_info = self.color_analyzer.analyze_colors(np.array(image))
                
                # Generate prompt using generate_detailed_prompt instead of _generate_detailed_prompt
                prompt_data = self.generate_detailed_prompt(image_features, pattern_info, color_info)
                pattern_info['prompt'] = prompt_data
                
            return pattern_info

        except Exception as e:
            logger.error(f"\nError in detect_patterns: {str(e)}")
            return self._get_default_pattern_info()

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
        """Analyze style with context-aware prompts."""
        try:
            pattern_type = pattern_info['primary_pattern']
            logger.info(f"\nAnalyzing detailed style for pattern type: {pattern_type}")
            
            style_queries = {
                'layout': [
                    "This {} pattern has a {} layout",
                    "The arrangement of {} elements is {}",
                    "The {} motifs are organized in a {} way"
                ],
                'scale': [
                    "The {} elements are {} in scale",
                    "This is a {} scale {} pattern",
                    "The size of {} motifs is {}"
                ],
                'texture': [
                    "The texture of this {} pattern is {}",
                    "This {} design has a {} surface quality",
                    "The {} elements create a {} texture"
                ]
            }

            results = {}
            for aspect, queries in style_queries.items():
                logger.info(f"\nAnalyzing {aspect}...")
                best_score = 0
                best_attribute = None
                
                attributes = self._get_attributes_for_aspect(aspect)
                logger.info(f"Testing {len(attributes)} possible {aspect} attributes")
                
                for attr in attributes:
                    attr_score = 0
                    valid_checks = 0
                    
                    for query in queries:
                        # First {} for pattern type, second {} for attribute
                        formatted_query = query.format(pattern_type, attr)
                        logger.info(f"Testing query: {formatted_query}")
                        
                        text_inputs = self.processor(
                            text=[formatted_query],
                            return_tensors="pt",
                            padding=True
                        )
                        
                        with torch.no_grad():
                            text_features = self.model.get_text_features(**text_inputs)
                            similarity = torch.nn.functional.cosine_similarity(
                                image_features, text_features, dim=1
                            )
                            score = float(similarity[0])
                            logger.info(f"Similarity score: {score:.4f}")
                            
                            if score > 0.2:
                                attr_score += score
                                valid_checks += 1
                    
                    if valid_checks > 0:
                        avg_score = attr_score / valid_checks
                        logger.info(f"Average score for {attr}: {avg_score:.4f}")
                        if avg_score > best_score:
                            best_score = avg_score
                            best_attribute = attr
                            logger.info(f"New best attribute for {aspect}: {attr} (score: {avg_score:.4f})")

                results[aspect] = {
                    'type': best_attribute or 'balanced',
                    'confidence': best_score
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
        """Generate a comprehensive prompt based on pattern analysis components."""
        try:
            # Get the key information we need
            pattern_category = pattern_info.get('category', 'unknown').lower()
            
            # Get colors
            colors = []
            if color_info and 'dominant_colors' in color_info:
                colors = [c['name'].lower() for c in color_info['dominant_colors'][:3]]
            
            # Get specific elements
            elements = []
            if 'elements' in pattern_info and pattern_info['elements']:
                elements = [e['name'] for e in pattern_info['elements'][:5]]
            
            # Get secondary patterns
            secondary_patterns = []
            if pattern_info.get('secondary_patterns'):
                secondary_patterns = [p['name'].lower() for p in pattern_info['secondary_patterns'][:2]]
            
            # Build a more natural description
            description_parts = []
            
            # Start with "This image appears to be a..."
            opening_phrases = [
                f"This image appears to be a {pattern_category} pattern",
                f"This is a {pattern_category} design",
                f"The image shows a {pattern_category} pattern"
            ]
            description_parts.append(opening_phrases[0])  # Use the first option by default
            
            # Add elements with more natural language
            if elements:
                # Check if any elements contain descriptive adjectives
                descriptive_elements = [e for e in elements if " " in e]
                regular_elements = [e for e in elements if " " not in e]
                
                # Prioritize descriptive elements
                featured_elements = descriptive_elements + regular_elements
                
                if len(featured_elements) == 1:
                    description_parts.append(f"featuring {featured_elements[0]}")
                elif len(featured_elements) == 2:
                    description_parts.append(f"featuring {featured_elements[0]} and {featured_elements[1]}")
                else:
                    elements_text = ", ".join(featured_elements[:-1]) + f", and {featured_elements[-1]}"
                    description_parts.append(f"featuring {elements_text}")
            
            # Add color description with more natural language
            if colors:
                # Add adjectives to colors based on brightness/saturation
                brightness = color_info.get('overall_brightness', 0.5)
                contrast = color_info.get('color_contrast', 0.5)
                
                color_adjective = "rich" if contrast > 0.4 else "subtle"
                if brightness > 0.7:
                    color_adjective = "bright" if contrast > 0.4 else "light"
                elif brightness < 0.3:
                    color_adjective = "deep" if contrast > 0.4 else "dark"
                
                if len(colors) == 1:
                    description_parts.append(f"with a prominent {color_adjective} {colors[0]} color palette")
                elif len(colors) == 2:
                    description_parts.append(f"with a {color_adjective} color palette of {colors[0]} and {colors[1]} tones")
                else:
                    colors_text = ", ".join(colors[:-1]) + f", and {colors[-1]}"
                    description_parts.append(f"with a {color_adjective} color palette of {colors_text} tones")
            
            # Add texture and detail information
            texture = pattern_info.get('texture_type', {}).get('type', 'detailed')
            texture_confidence = pattern_info.get('texture_type', {}).get('confidence', 0)
            
            # Only include texture if we have reasonable confidence
            if texture_confidence > 0.2:
                description_parts.append(f"The {pattern_category} elements have {texture} textures")
            
            # Add layout information in natural language
            layout = pattern_info.get('layout', {}).get('type', 'balanced')
            scale = pattern_info.get('scale', {}).get('type', 'medium')
            layout_confidence = pattern_info.get('layout', {}).get('confidence', 0)
            
            # Only include layout if we have reasonable confidence
            if layout_confidence > 0.2:
                description_parts.append(f"arranged in a {layout} composition at {scale} scale")
            
            # Add style influence if relevant
            if secondary_patterns:
                patterns_text = " and ".join(secondary_patterns)
                description_parts.append(f"with influences of {patterns_text} style")
            
            # Add mood/feeling
            mood = pattern_info.get('mood', {}).get('type', 'vibrant')
            mood_confidence = pattern_info.get('mood', {}).get('confidence', 0)
            
            # Only include mood if we have reasonable confidence
            if mood_confidence > 0.2:
                description_parts.append(f"creating a {mood} visual effect")
            
            # Combine into a natural paragraph
            final_prompt = ". ".join(description_parts) + "."
            
            # Replace multiple spaces with single space
            final_prompt = " ".join(final_prompt.split())
            
            # Fix any awkward phrasings
            final_prompt = final_prompt.replace(" .", ".")
            final_prompt = final_prompt.replace("..", ".")
            
            # Add more natural language variations
            final_prompt = final_prompt.replace("featuring", "showcasing")
            final_prompt = final_prompt.replace("with a", "having a")
            final_prompt = final_prompt.replace("arranged in", "organized in")
            
            return {
                "final_prompt": final_prompt,
                "components": {
                    "pattern_category": pattern_category,
                    "elements": elements,
                    "colors": colors,
                    "texture": texture,
                    "layout": layout,
                    "scale": scale,
                    "secondary_patterns": secondary_patterns,
                    "mood": mood
                },
                "completeness_score": min(len(description_parts) / 7, 1.0)
            }

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
            patterns = self.detect_patterns(image)
            if not patterns:
                logger.warning(f"Warning: No patterns detected in {image_path}")
                patterns = {
                    "category": "Unknown",
                    "category_confidence": 0.0,
                    "prompt": {"final_prompt": "No pattern detected"}
                }
                
            # Extract colors
            image_array = np.array(image)
            colors = self.analyze_colors(image_array)
            
            # Create metadata
            metadata = {
                "original_path": str(image_path),
                "thumbnail_path": str(thumbnail_path.name),
                "patterns": patterns,
                "colors": colors,
                "timestamp": str(Path(image_path).stat().st_mtime)
            }
            
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