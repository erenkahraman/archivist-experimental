from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
import faiss
import json
import config
import colorsys
from sklearn.cluster import KMeans
from PIL import Image

class SearchEngine:
    def __init__(self):
        print("Initializing SearchEngine...")
        try:
            # Load models and move to GPU if available
            print(f"Loading CLIP model from: {config.CLIP_MODEL_NAME}")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = CLIPModel.from_pretrained(config.CLIP_MODEL_NAME).to(self.device)
            print("CLIP model loaded successfully")
            self.processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL_NAME)
            print("CLIP processor loaded successfully")
            
            # Initialize other components
            self.min_clusters = 3
            self.max_clusters = config.N_CLUSTERS
            self.kmeans = None
            self.index = None
            self.metadata_file = config.BASE_DIR / "metadata.json"
            self.metadata = self.load_metadata()
            print("SearchEngine initialization completed")
        except Exception as e:
            print(f"Error initializing SearchEngine: {str(e)}")
            print("Stack trace:")
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
            print("\n=== Starting pattern detection ===")
            print("Checking model and processor...")
            if not self.model or not self.processor:
                raise Exception("CLIP model or processor not initialized properly")
            
            # Process image through CLIP - optimized for batch processing
            print("Processing image through CLIP...")
            try:
                # Convert PIL Image to appropriate format before passing to processor
                inputs = self.processor(
                    images=image,
                    return_tensors="pt",
                    padding=True
                )
                print("Image processed successfully")
                print(f"Input shape: {inputs['pixel_values'].shape}")
            except Exception as e:
                print(f"Error processing image: {str(e)}")
                raise
            
            with torch.no_grad():
                print("Getting image features...")
                try:
                    image_features = self.model.get_image_features(pixel_values=inputs['pixel_values'].to(self.device))
                    print(f"Image features shape: {image_features.shape}")
                    # Analyze patterns before moving to CPU
                    print("\nAnalyzing patterns...")
                    pattern_info = self._analyze_patterns(image_features)
                    
                    # Now we can move to CPU
                    image_features = image_features.cpu()
                    print("Image features extracted successfully")
                except Exception as e:
                    print(f"Error extracting image features: {str(e)}")
                    raise

            # Style analysis can be done asynchronously or in a lighter way
            print("\nStarting detailed style analysis...")
            style_info = self._analyze_detailed_style(image_features, pattern_info)
            if style_info:  # Null check
                pattern_info.update(style_info)
                print("Style analysis completed and updated")
            else:
                print("Style analysis returned no results")

            # Generate prompt
            print("\nAnalyzing colors...")
            color_info = self.analyze_colors(np.array(image))
            if color_info:
                print("Color analysis completed")
            else:
                print("Color analysis failed")
                
            print("\nGenerating detailed prompt...")
            prompt_data = self.generate_detailed_prompt(image_features, pattern_info, color_info)
            pattern_info['prompt'] = prompt_data
            print("=== Pattern analysis completed successfully ===\n")

            return pattern_info

        except Exception as e:
            print(f"\nError in detect_patterns: {str(e)}")
            print("Stack trace:")
            import traceback
            traceback.print_exc()
            print("\nReturning default pattern info")
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
            print("Starting pattern analysis...")
            
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
            print(f"Error in pattern analysis: {str(e)}")
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
            print(f"\nAnalyzing detailed style for pattern type: {pattern_type}")
            
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
                print(f"\nAnalyzing {aspect}...")
                best_score = 0
                best_attribute = None
                
                attributes = self._get_attributes_for_aspect(aspect)
                print(f"Testing {len(attributes)} possible {aspect} attributes")
                
                for attr in attributes:
                    attr_score = 0
                    valid_checks = 0
                    
                    for query in queries:
                        # First {} for pattern type, second {} for attribute
                        formatted_query = query.format(pattern_type, attr)
                        print(f"Testing query: {formatted_query}")
                        
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
                            print(f"Similarity score: {score:.4f}")
                            
                            if score > 0.2:
                                attr_score += score
                                valid_checks += 1
                    
                    if valid_checks > 0:
                        avg_score = attr_score / valid_checks
                        print(f"Average score for {attr}: {avg_score:.4f}")
                        if avg_score > best_score:
                            best_score = avg_score
                            best_attribute = attr
                            print(f"New best attribute for {aspect}: {attr} (score: {avg_score:.4f})")

                results[aspect] = {
                    'type': best_attribute or 'balanced',
                    'confidence': best_score
                }
                print(f"Final {aspect} result: {best_attribute or 'balanced'} (confidence: {best_score:.4f})")

            return results

        except Exception as e:
            print(f"Error in _analyze_detailed_style: {str(e)}")
            print("Stack trace:")
            import traceback
            traceback.print_exc()
            return {}

    def generate_detailed_prompt(self, image_features, pattern_info, color_info):
        """Generate a comprehensive prompt based on pattern analysis components."""
        try:
            prompt_parts = []

            # 1. Textile Design Pattern
            if pattern_info.get('category'):
                prompt_parts.append(
                    f"A {pattern_info['category'].lower()} textile design"
                )

            # 2. Color Harmony
            if color_info and 'dominant_colors' in color_info:
                colors = [c['name'].lower() for c in color_info['dominant_colors'][:3]]
                prompt_parts.append(
                    f"featuring a harmonious blend of {', '.join(colors)}"
                )

            # 3. Motifs and Themes
            if pattern_info.get('secondary_patterns'):
                motifs = [p['name'].lower() for p in pattern_info['secondary_patterns'][:2]]
                prompt_parts.append(
                    f"with {' and '.join(motifs)} motifs"
                )

            # 4. Pattern Repetition and Layout
            layout = pattern_info.get('layout', {}).get('type', 'balanced')
            repeat = pattern_info.get('repeat_type', {}).get('type', 'regular')
            prompt_parts.append(
                f"arranged in {layout} layout with {repeat} repetition"
            )

            # 5. Scale and Proportion
            scale = pattern_info.get('scale', {}).get('type', 'medium')
            prompt_parts.append(
                f"designed at {scale} scale"
            )

            # 6. Texture and Detailing
            texture = pattern_info.get('texture_type', {}).get('type', 'smooth')
            prompt_parts.append(
                f"with {texture} textural details"
            )

            # 7. Cultural/Historical Context
            cultural = pattern_info.get('cultural_influence', {}).get('type', 'contemporary')
            historical = pattern_info.get('historical_period', {}).get('type', 'modern')
            prompt_parts.append(
                f"inspired by {cultural} {historical} traditions"
            )

            # 8. Emotional Appeal
            mood = pattern_info.get('mood', {}).get('type', 'balanced')
            prompt_parts.append(
                f"conveying a {mood} aesthetic"
            )

            # 9. Originality
            style = pattern_info.get('style_keywords', ['unique'])[0]
            prompt_parts.append(
                f"with {style} interpretation"
            )

            # Combine all parts into final prompt
            final_prompt = " ".join(prompt_parts)

            return {
                "final_prompt": final_prompt,
                "components": {
                    "textile_design": prompt_parts[0] if prompt_parts else None,
                    "color_harmony": prompt_parts[1] if len(prompt_parts) > 1 else None,
                    "motifs_themes": prompt_parts[2] if len(prompt_parts) > 2 else None,
                    "pattern_layout": prompt_parts[3] if len(prompt_parts) > 3 else None,
                    "scale_proportion": prompt_parts[4] if len(prompt_parts) > 4 else None,
                    "texture_details": prompt_parts[5] if len(prompt_parts) > 5 else None,
                    "cultural_context": prompt_parts[6] if len(prompt_parts) > 6 else None,
                    "emotional_appeal": prompt_parts[7] if len(prompt_parts) > 7 else None,
                    "originality": prompt_parts[8] if len(prompt_parts) > 8 else None
                },
                "completeness_score": len(prompt_parts) / 9
            }

        except Exception as e:
            print(f"Error generating detailed prompt: {str(e)}")
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
            print(f"Error in analyze_colors: {str(e)}")
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
            print(f"Error creating thumbnail: {str(e)}")
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
                print(f"Error processing {path}: {str(e)}")
                continue
        return results

    def process_image(self, image_path: Path) -> Dict:
        """Process a single image and extract its patterns and metadata."""
        try:
            print(f"\n=== Processing image: {image_path} ===")
            
            # Check if image exists
            if not image_path.exists():
                print(f"Error: Image file not found at {image_path}")
                return None
                
            # Create thumbnail
            thumbnail_path = self.create_thumbnail(image_path)
            if not thumbnail_path:
                print(f"Error: Failed to create thumbnail for {image_path}")
                return None
                
            # Open and process the image
            try:
                image = Image.open(image_path)
                # Convert to RGB if needed (for PNG with transparency)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
            except Exception as e:
                print(f"Error opening image: {str(e)}")
                return None
                
            # Extract patterns
            patterns = self.detect_patterns(image)
            if not patterns:
                print(f"Warning: No patterns detected in {image_path}")
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
            
            print(f"Successfully processed image: {image_path}")
            return metadata
            
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def search(self, query: str, k: int = 10) -> List[Dict]:
        """
        Enhanced search for images based on query with hierarchical ranking.
        
        This search algorithm uses a multi-tier approach:
        1. Exact pattern matches (highest priority)
        2. Semantic similarity
        3. Color matches
        4. Attribute matches (style, layout, etc.)
        
        Results are normalized to provide realistic match percentages.
        """
        try:
            if not query.strip():
                # Return all images if query is empty
                metadata_list = list(self.metadata.values())
                return metadata_list[:k]
                
            # Split query into terms and normalize
            query_terms = [term.lower().strip() for term in query.lower().split() if term.strip()]
            
            # Get all metadata as a list for processing
            metadata_list = []
            for path, metadata in self.metadata.items():
                metadata_list.append(metadata)

            if not metadata_list:
                print("No images in database")
                return []

            # Process the query with CLIP to get semantic features
            with torch.no_grad():
                text_inputs = self.processor(
                    text=[query],
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                query_features = self.model.get_text_features(**text_inputs).cpu()

            # Calculate scores for each image
            results = []
            max_scores = {
                'exact_match': 0.0,
                'semantic': 0.0,
                'pattern': 0.0,
                'color': 0.0,
                'attribute': 0.0,
                'term_matches': 0
            }
            
            for metadata in metadata_list:
                # Initialize score components with hierarchical weighting
                scores = {
                    'exact_match': 0.0,  # Exact pattern match (highest priority)
                    'semantic': 0.0,     # Semantic similarity score
                    'pattern': 0.0,      # Pattern match score
                    'color': 0.0,        # Color match score
                    'attribute': 0.0,    # Attribute match score
                    'term_matches': 0,   # Number of query terms matched
                    'term_coverage': 0.0 # Percentage of query terms matched
                }
                
                # 1. EXACT PATTERN MATCHING (highest priority)
                if 'patterns' in metadata:
                    patterns = metadata['patterns']
                    
                    # Check for exact category match (highest priority)
                    if 'category' in patterns:
                        category = patterns['category'].lower()
                        category_confidence = patterns.get('category_confidence', 0.8)
                        
                        # Check for exact match with full query
                        if query.lower() == category:
                            scores['exact_match'] = 10.0 * category_confidence  # Very high weight for exact matches
                        
                        # Check for exact matches with individual terms
                        for term in query_terms:
                            if term == category:
                                scores['exact_match'] += 5.0 * category_confidence
                                scores['term_matches'] += 1
                            elif term in category:
                                scores['exact_match'] += 2.0 * category_confidence
                                scores['term_matches'] += 1
                
                # 2. SEMANTIC MATCHING using prompt
                if 'patterns' in metadata and 'prompt' in metadata['patterns']:
                    prompt_text = metadata['patterns']['prompt'].get('final_prompt', '')
                    if prompt_text:
                        # Calculate semantic similarity between query and prompt
                        with torch.no_grad():
                            prompt_inputs = self.processor(
                                text=[prompt_text],
                                return_tensors="pt",
                                padding=True,
                                truncation=True
                            )
                            prompt_features = self.model.get_text_features(**prompt_inputs).cpu()
                            
                            # Compute cosine similarity
                            similarity = torch.nn.functional.cosine_similarity(
                                query_features, prompt_features
                            ).item()
                            
                            scores['semantic'] = similarity * 3.0  # Scale semantic similarity
                        
                        # Also check for term matches in prompt
                        prompt_lower = prompt_text.lower()
                        for term in query_terms:
                            if f" {term} " in f" {prompt_lower} ":  # Match whole words
                                scores['term_matches'] += 1
                
                # 3. PATTERN TYPE MATCHING with term weighting
                if 'patterns' in metadata:
                    patterns = metadata['patterns']
                    
                    # Primary pattern matching
                    if 'category' in patterns:
                        primary_pattern = patterns['category'].lower()
                        primary_confidence = patterns.get('category_confidence', 0.8)
                        
                        # Check for partial matches with primary pattern
                        for term in query_terms:
                            if term in primary_pattern and term != primary_pattern:
                                # Higher weight for primary pattern matches
                                scores['pattern'] += primary_confidence * 2.0
                                scores['term_matches'] += 1
                    
                    # Secondary patterns matching
                    if 'secondary_patterns' in patterns:
                        for pattern in patterns['secondary_patterns']:
                            pattern_name = pattern['name'].lower()
                            pattern_conf = pattern['confidence']
                            
                            for term in query_terms:
                                if term == pattern_name:
                                    scores['pattern'] += pattern_conf * 1.5
                                    scores['term_matches'] += 1
                                elif term in pattern_name:
                                    scores['pattern'] += pattern_conf
                                    scores['term_matches'] += 1
                    
                    # Style keywords matching
                    if 'style_keywords' in patterns:
                        for keyword in patterns['style_keywords']:
                            keyword_lower = keyword.lower()
                            for term in query_terms:
                                if term == keyword_lower:
                                    scores['attribute'] += 0.8  # Higher weight for exact keyword match
                                    scores['term_matches'] += 1
                                elif term in keyword_lower:
                                    scores['attribute'] += 0.4  # Lower weight for partial keyword match
                                    scores['term_matches'] += 1
                    
                    # Match other attributes (layout, scale, texture, etc.)
                    for attr_type in ['layout', 'scale', 'texture_type', 'cultural_influence', 'historical_period', 'mood']:
                        if attr_type in patterns:
                            attr = patterns[attr_type]
                            if 'type' in attr:
                                attr_type_value = attr['type'].lower()
                                attr_conf = attr.get('confidence', 0.5)
                                
                                for term in query_terms:
                                    if term == attr_type_value:
                                        scores['attribute'] += attr_conf * 0.8
                                        scores['term_matches'] += 1
                                    elif term in attr_type_value:
                                        scores['attribute'] += attr_conf * 0.4
                                        scores['term_matches'] += 1

                # 4. COLOR MATCHING with term weighting
                if 'colors' in metadata:
                    colors = metadata['colors']
                    if 'dominant_colors' in colors:
                        for color in colors['dominant_colors']:
                            color_name = color['name'].lower()
                            proportion = color['proportion']
                            
                            for term in query_terms:
                                if term == color_name:
                                    scores['color'] += proportion * 2.0  # Higher weight for exact color match
                                    scores['term_matches'] += 1
                                elif term in color_name:
                                    scores['color'] += proportion * 1.0  # Weight by color proportion
                                    scores['term_matches'] += 1
                
                # Calculate term coverage - what percentage of query terms were matched
                scores['term_coverage'] = scores['term_matches'] / len(query_terms) if query_terms else 0
                
                # Calculate final score with hierarchical weighting
                # The weights establish a clear hierarchy of matching criteria
                final_score = (
                    (scores['exact_match'] * 0.40) +  # Exact matches are highest priority
                    (scores['semantic'] * 0.25) +     # Semantic similarity is second priority
                    (scores['pattern'] * 0.15) +      # Pattern matches are third priority
                    (scores['color'] * 0.10) +        # Color matches are fourth priority
                    (scores['attribute'] * 0.05) +    # Attribute matches are fifth priority
                    (scores['term_coverage'] * 0.05)  # Reward matching more query terms
                )
                
                # Track maximum scores for normalization
                for key in max_scores:
                    if key in scores and scores[key] > max_scores[key]:
                        max_scores[key] = scores[key]
                
                # Only include results with a match
                if final_score > 0:
                    results.append({
                        **metadata,
                        'raw_score': final_score,
                        'scores': scores,
                        'matched_terms': scores['term_matches']
                    })
            
            # Normalize scores to provide realistic match percentages
            # This prevents inflated 100% matches
            if results:
                # Find the maximum raw score
                max_raw_score = max(result['raw_score'] for result in results)
                
                # Normalize scores to a realistic range (max 95%)
                for result in results:
                    # Calculate normalized similarity as a percentage
                    # Cap at 95% to avoid misleading 100% matches
                    normalized_score = (result['raw_score'] / max_raw_score) * 0.95
                    result['similarity'] = normalized_score
                    
                    # Remove raw scores and detailed scores from final results
                    del result['raw_score']
                    del result['scores']
            
            # Sort by similarity score
            results.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Debug information
            if results:
                print(f"Search for '{query}' found {len(results)} results")
                print(f"Top match: {results[0]['patterns']['category']} with score {results[0]['similarity']:.2f}")
            
            return results[:k]

        except Exception as e:
            print(f"Error in search: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

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
                ).to(self.device)  # Text inputs'u GPU'ya taşı
                
                with torch.no_grad():
                    text_features = self.model.get_text_features(**text_inputs)
                    similarity = torch.nn.functional.cosine_similarity(
                        features.to(self.device),  # Features'ı GPU'ya taşı
                        text_features,
                        dim=1
                    )
                    scores[attr] = float(similarity[0].cpu())  # Sonucu CPU'ya taşı

            # Get the most likely attribute
            top_attr = max(scores.items(), key=lambda x: x[1])
            return {
                'type': top_attr[0],
                'confidence': top_attr[1]
            }

        except Exception as e:
            print(f"Error in _analyze_attributes: {str(e)}")
            return {
                'type': 'unknown',
                'confidence': 0.0
            }

    def _get_attributes_for_aspect(self, aspect):
        """Get possible attributes for a given aspect."""
        print(f"\nGetting attributes for aspect: {aspect}")
        attributes = {
            'layout': ["balanced", "asymmetric", "radial", "linear", "scattered", "grid-like", "concentric"],
            'scale': ["small", "medium", "large", "varied", "proportional", "intricate", "bold"],
            'texture': ["smooth", "rough", "layered", "flat", "dimensional", "textured", "embossed"]
        }
        result = attributes.get(aspect, ["balanced"])
        print(f"Available attributes: {result}")
        return result 