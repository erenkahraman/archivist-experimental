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
        # Load models
        self.model = CLIPModel.from_pretrained(config.CLIP_MODEL_NAME)
        self.processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL_NAME)
        
        # Initialize other components
        self.min_clusters = 3
        self.max_clusters = config.N_CLUSTERS
        self.kmeans = None
        self.index = None
        self.metadata_file = config.BASE_DIR / "metadata.json"
        self.metadata = self.load_metadata()

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
            # Process image through CLIP - batch processing için optimize edildi
            inputs = self.processor(images=image, return_tensors="pt", padding=True)
            
            with torch.no_grad():  # Memory kullanımını optimize et
                image_features = self.model.get_image_features(**inputs)
                image_features = image_features.cpu()  # GPU memory'i hemen boşalt

            # Pattern tespitini optimize et - batch processing
            pattern_types = [
                "floral pattern", "geometric pattern", "abstract pattern",
                "striped pattern", "polka dot pattern", "paisley pattern",
                "animal print pattern", "tribal pattern", "damask pattern",
                "plaid pattern", "checkered pattern", "organic pattern"
            ]

            # Batch processing için tüm pattern'ları tek seferde işle
            all_texts = [f"this is a {pattern}" for pattern in pattern_types]
            text_inputs = self.processor(
                text=all_texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            )

            with torch.no_grad():
                text_features = self.model.get_text_features(**text_inputs)
                text_features = text_features.cpu()
                similarities = torch.nn.functional.cosine_similarity(
                    image_features.unsqueeze(1),
                    text_features.unsqueeze(0),
                    dim=2
                )
                pattern_scores = {
                    pattern.replace(" pattern", ""): float(score)
                    for pattern, score in zip(pattern_types, similarities[0])
                }

            # Get top patterns
            top_patterns = sorted(pattern_scores.items(), key=lambda x: x[1], reverse=True)
            primary_pattern = top_patterns[0]

            # Create pattern info with null checks
            pattern_info = {
                'category': primary_pattern[0].capitalize(),
                'category_confidence': float(primary_pattern[1]),  # Convert to native Python float
                'primary_pattern': primary_pattern[0],
                'pattern_confidence': float(primary_pattern[1]),
                'secondary_patterns': [
                    {'name': pattern, 'confidence': float(score)}  # Convert to native Python float
                    for pattern, score in top_patterns[1:4]
                    if score > 0.2
                ],
                'layout': {'type': 'balanced', 'confidence': 0.0},  # Default değerler
                'repeat_type': {'type': 'regular', 'confidence': 0.0},
                'scale': {'type': 'medium', 'confidence': 0.0},
                'texture_type': {'type': 'smooth', 'confidence': 0.0},
                'cultural_influence': {'type': 'contemporary', 'confidence': 0.0},
                'historical_period': {'type': 'modern', 'confidence': 0.0},
                'mood': {'type': 'balanced', 'confidence': 0.0},
                'style_keywords': ['balanced']
            }

            # Style analizi asenkron yapılabilir veya daha hafif bir şekilde yapılabilir
            style_info = self._analyze_detailed_style(image_features, pattern_info)
            if style_info:  # Null check
                pattern_info.update(style_info)

            # Generate prompt
            color_info = self.analyze_colors(np.array(image))
            prompt_data = self.generate_detailed_prompt(image_features, pattern_info, color_info)
            pattern_info['prompt'] = prompt_data

            return pattern_info

        except Exception as e:
            print(f"Error in detect_patterns: {str(e)}")
            # Tüm gerekli alanları içeren default response
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

    def _analyze_detailed_style(self, image_features, pattern_info):
        """Analyze style with context-aware prompts."""
        try:
            pattern_type = pattern_info['primary_pattern']
            
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
                best_score = 0
                best_attribute = None
                
                attributes = self._get_attributes_for_aspect(aspect)
                for attr in attributes:
                    attr_score = 0
                    valid_checks = 0
                    
                    for query in queries:
                        # İlk {} pattern type için, ikinci {} attribute için
                        formatted_query = query.format(pattern_type, attr)
                        
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
                            
                            if score > 0.2:
                                attr_score += score
                                valid_checks += 1
                    
                    if valid_checks > 0:
                        avg_score = attr_score / valid_checks
                        if avg_score > best_score:
                            best_score = avg_score
                            best_attribute = attr

                results[aspect] = {
                    'type': best_attribute or 'balanced',
                    'confidence': best_score
                }

            return results

        except Exception as e:
            print(f"Error in _analyze_detailed_style: {str(e)}")
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
        """Process a single image and return its metadata."""
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Create thumbnail
            thumbnail_path = self.create_thumbnail(image_path)
            if not thumbnail_path:
                raise Exception("Failed to create thumbnail")
            
            # Get image dimensions
            width, height = image.size
            
            # Analyze patterns and colors
            pattern_info = self.detect_patterns(image)
            color_info = self.analyze_colors(np.array(image))
            
            # Create metadata
            metadata = {
                'original_path': str(image_path),
                'thumbnail_path': str(thumbnail_path.relative_to(config.BASE_DIR)),
                'dimensions': {'width': width, 'height': height},
                'colors': color_info,
                'patterns': pattern_info,
                'isUploading': False
            }
            
            # Save metadata
            self.metadata[str(image_path)] = metadata
            self.save_metadata()
            
            return metadata
            
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None

    def search(self, query: str, k: int = 10) -> List[Dict]:
        """Search for images based on query."""
        try:
            # Split query into terms
            query_terms = query.lower().split()
            
            # Get all metadata as a list for processing
            metadata_list = []
            for path, metadata in self.metadata.items():
                metadata_list.append(metadata)

            if not metadata_list:
                print("No images in database")
                return []

            # Calculate scores for each image
            results = []
            for metadata in metadata_list:
                # Initialize score components
                scores = {
                    'pattern': 0.0,
                    'color': 0.0
                }
                
                # Pattern Type Matching
                if 'patterns' in metadata:
                    patterns = metadata['patterns']
                    if 'primary_pattern' in patterns:
                        pattern = patterns['primary_pattern'].lower()
                        confidence = patterns['pattern_confidence']
                        
                        for term in query_terms:
                            if term in pattern:
                                scores['pattern'] = confidence
                    
                    # Check secondary patterns
                    if 'secondary_patterns' in patterns:
                        for pattern in patterns['secondary_patterns']:
                            pattern_name = pattern['name'].lower()
                            pattern_conf = pattern['confidence']
                            for term in query_terms:
                                if term in pattern_name:
                                    scores['pattern'] = max(scores['pattern'], pattern_conf)

                # Color Matching
                if 'colors' in metadata:
                    colors = metadata['colors']
                    if 'dominant_colors' in colors:
                        for color in colors['dominant_colors']:
                            color_name = color['name'].lower()
                            proportion = color['proportion']
                            for term in query_terms:
                                if term in color_name:
                                    scores['color'] = max(scores['color'], proportion)

                # Calculate final score (weighted average)
                final_score = (scores['pattern'] * 0.6) + (scores['color'] * 0.4)
                
                if final_score > 0:  # Only include results with a match
                    results.append({
                        **metadata,
                        'similarity': final_score
                    })

            # Sort by similarity score
            results.sort(key=lambda x: x['similarity'], reverse=True)
            
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
                )
                
                with torch.no_grad():
                    text_features = self.model.get_text_features(**text_inputs)
                    similarity = torch.nn.functional.cosine_similarity(
                        features, text_features, dim=1
                    )
                    scores[attr] = float(similarity[0])

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