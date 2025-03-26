from typing import Dict, List, Any, Tuple
import numpy as np
import colorsys
from sklearn.cluster import KMeans, MiniBatchKMeans
import logging
import google.generativeai as genai
from PIL import Image
import os
import json
import io
import base64
from src.config.prompts import GEMINI_CONFIG
import cv2
from collections import Counter
import math

logger = logging.getLogger(__name__)

class ColorAnalyzer:
    def __init__(self, max_clusters: int = 15, api_key=None):
        self.max_clusters = max_clusters
        # Load expanded color reference database for accurate matching
        self._load_reference_colors()
        
        # Use provided API key or get from environment
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if self._api_key:
            # Mask API key in logs and never store the full key in instance variables
            masked_key = self._mask_api_key(self._api_key)
            logger.info(f"Using Gemini API key: {masked_key}")
            # Configure the API client but don't store the raw key
            genai.configure(api_key=self._api_key)
            self.use_gemini = True
        else:
            logger.warning("No Gemini API key provided for color analysis. Will use fallback method.")
            self.use_gemini = False
        
        # Precompute values for faster processing
        self._precompute_reference_colors()

    def _load_reference_colors(self):
        """Load comprehensive color reference database."""
        # Include essential design-focused colors
        self.color_references = {
            # Reds
            'Red': (255, 0, 0),
            'Crimson': (220, 20, 60),
            'Dark Red': (139, 0, 0),
            'Burgundy': (128, 0, 32),
            'Maroon': (128, 0, 0),
            'Ruby Red': (224, 17, 95),
            'Cherry': (222, 49, 99),
            'Cardinal Red': (196, 30, 58),
            'Scarlet': (255, 36, 0),
            'Vermilion': (227, 66, 52),
            
            # Pinks
            'Pink': (255, 192, 203),
            'Hot Pink': (255, 105, 180),
            'Deep Pink': (255, 20, 147),
            'Rose': (255, 0, 127),
            'Fuchsia': (255, 0, 255),
            'Magenta': (255, 0, 255),
            'Coral Pink': (248, 131, 121),
            'Salmon': (250, 128, 114),
            
            # Oranges
            'Orange': (255, 165, 0),
            'Dark Orange': (255, 140, 0),
            'Coral': (255, 127, 80),
            'Peach': (255, 218, 185),
            'Tangerine': (242, 133, 0),
            'Amber': (255, 191, 0),
            
            # Yellows
            'Yellow': (255, 255, 0),
            'Gold': (255, 215, 0),
            'Goldenrod': (218, 165, 32),
            'Khaki': (240, 230, 140),
            'Mustard': (225, 173, 1),
            'Lemon': (255, 250, 205),
            
            # Greens
            'Green': (0, 128, 0),
            'Lime': (0, 255, 0),
            'Olive': (128, 128, 0),
            'Forest Green': (34, 139, 34),
            'Dark Green': (0, 100, 0),
            'Light Green': (144, 238, 144),
            'Mint': (189, 252, 201),
            'Emerald': (80, 200, 120),
            'Sage': (188, 184, 138),
            'Chartreuse': (127, 255, 0),
            'Avocado': (86, 130, 3),
            
            # Cyans
            'Teal': (0, 128, 128),
            'Turquoise': (64, 224, 208),
            'Cyan': (0, 255, 255),
            'Aqua': (0, 255, 255),
            'Sky Blue': (135, 206, 235),
            
            # Blues
            'Blue': (0, 0, 255),
            'Navy': (0, 0, 128),
            'Royal Blue': (65, 105, 225),
            'Light Blue': (173, 216, 230),
            'Indigo': (75, 0, 130),
            'Cobalt': (0, 71, 171),
            'Azure': (0, 127, 255),
            'Cerulean': (42, 82, 190),
            'Sapphire': (15, 82, 186),
            
            # Purples
            'Purple': (128, 0, 128),
            'Violet': (238, 130, 238),
            'Lavender': (230, 230, 250),
            'Plum': (221, 160, 221),
            'Mauve': (204, 153, 204),
            'Amethyst': (153, 102, 204),
            'Periwinkle': (204, 204, 255),
            
            # Browns
            'Brown': (165, 42, 42),
            'Dark Brown': (101, 67, 33),
            'Chocolate': (210, 105, 30),
            'Sienna': (160, 82, 45),
            'Tan': (210, 180, 140),
            'Beige': (245, 245, 220),
            'Taupe': (72, 60, 50),
            'Camel': (193, 154, 107),
            
            # Grays
            'Gray': (128, 128, 128),
            'Light Gray': (211, 211, 211),
            'Dark Gray': (169, 169, 169),
            'Silver': (192, 192, 192),
            'Charcoal': (54, 69, 79),
            'Slate': (112, 128, 144),
            
            # White/Black
            'Black': (0, 0, 0),
            'White': (255, 255, 255),
            'Ivory': (255, 255, 240),
            'Cream': (255, 253, 208),
        }

    def _precompute_reference_colors(self):
        """Precompute color spaces for reference colors to enable accurate matching."""
        # Store both HSV and Lab values for better matching
        self.reference_hsv = {}
        self.reference_lab = {}
        
        for name, rgb in self.color_references.items():
            r, g, b = rgb
            # HSV for basic matching
            self.reference_hsv[name] = colorsys.rgb_to_hsv(r/255, g/255, b/255)
            
            # Lab for perceptual matching (delta-E calculations)
            # Convert to BGR for OpenCV
            bgr = np.uint8([[[b, g, r]]])
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            self.reference_lab[name] = lab[0, 0]

    def _mask_api_key(self, key):
        """Safely mask API key for logging purposes."""
        if not key or len(key) < 8:
            return "INVALID_KEY"
        # Show only first 4 and last 4 characters
        return f"{key[:4]}...{key[-4:]}"

    def set_api_key(self, api_key: str):
        """Set or update the Gemini API key"""
        if api_key:
            masked_key = self._mask_api_key(api_key)
            logger.info(f"Updating Gemini API key: {masked_key}")
            # Configure the API client
            genai.configure(api_key=api_key)
            self._api_key = api_key
            self.use_gemini = True
            logger.info("Gemini API key updated for color analysis")
        else:
            logger.warning("Attempted to set empty API key")
            self._api_key = None
            self.use_gemini = False

    def analyze_colors(self, image: np.ndarray) -> Dict:
        """Analyze colors in the image with token-optimized methods."""
        try:
            logger.info("Starting color analysis...")
            
            # Convert image to RGB if needed
            if len(image.shape) == 2:  # Grayscale
                image = np.stack((image,) * 3, axis=-1)
            elif image.shape[2] == 4:  # RGBA
                image = image[:, :, :3]
            
            # Downsize the image for faster processing 
            h, w = image.shape[:2]
            target_pixels = 60_000  # Further reduced resolution to save tokens
            scale = min(1.0, np.sqrt(target_pixels / (h * w)))
            if scale < 1.0:
                new_h, new_w = int(h * scale), int(w * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # First try a local-only approach to avoid API calls when possible
            # Only use Gemini for complex, high-color-variety images
            local_result = self._quick_color_check(image)
            
            # Lower threshold to use local processing more often
            if local_result.get('color_complexity', 1.0) < 0.55:
                logger.info("Image has simple color palette, using local analysis to save tokens")
                return self._precision_color_analysis(image)
            
            if self.use_gemini and self._api_key:  # Check if API key flag is true, not the key itself
                # Try using Gemini API with minimal prompting
                try:
                    # Convert to PIL Image for Gemini
                    pil_image = Image.fromarray(image.astype('uint8'))
                    gemini_result = self._analyze_with_gemini(pil_image)
                    if gemini_result and 'dominant_colors' in gemini_result and len(gemini_result['dominant_colors']) > 0:
                        logger.info(f"Successfully analyzed colors with Gemini API, found {len(gemini_result['dominant_colors'])} colors")
                        return gemini_result
                except Exception as e:
                    logger.error(f"Error analyzing colors with Gemini API: {str(e)}")
                    logger.info("Falling back to precision color analysis")
            
            # High precision fallback method
            return self._precision_color_analysis(image)
            
        except Exception as e:
            logger.error(f"Error in color analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "dominant_colors": [],
                "overall_brightness": 0.5,
                "color_contrast": 0.2
            }
            
    def _quick_color_check(self, image: np.ndarray) -> Dict:
        """Quickly evaluate if an image has a simple or complex color palette to decide if Gemini is needed."""
        # Take image samples to check color complexity
        h, w = image.shape[:2]
        samples = []
        
        # Sample grid points
        for y in range(0, h, max(1, h//10)):
            for x in range(0, w, max(1, w//10)):
                samples.append(image[y, x])
                
        # Convert to LAB for perceptual comparison
        lab_samples = []
        for rgb in samples:
            bgr = np.uint8([[[rgb[2], rgb[1], rgb[0]]]])
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0, 0]
            lab_samples.append(lab)
            
        # Calculate color variations
        if len(lab_samples) > 1:
            # Calculate standard deviation of colors
            lab_std = np.std(lab_samples, axis=0)
            # Average the standard deviations of L, a, b channels
            color_complexity = float(np.mean(lab_std) / 30)  # Normalize to approx 0-1 scale
        else:
            color_complexity = 0.0
            
        return {
            'color_complexity': color_complexity,
            'sample_count': len(samples)
        }

    def _post_process_gemini_colors(self, gemini_result: Dict, original_image: np.ndarray) -> Dict:
        """Post-process Gemini colors to ensure they match actual colors in the image."""
        # Convert image to LAB color space for better color matching
        lab_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2LAB)
        
        # For each color returned by Gemini
        for color in gemini_result.get('dominant_colors', []):
            if 'rgb' in color and isinstance(color['rgb'], list) and len(color['rgb']) == 3:
                # Get the RGB values
                r, g, b = [int(x) for x in color['rgb']]
                
                # Find the closest actual color in the image
                closest_rgb = self._find_closest_real_color(r, g, b, original_image)
                
                # Update the RGB value with the actual color from the image
                if closest_rgb:
                    color['rgb'] = closest_rgb
                    # Update hex code to match
                    r, g, b = closest_rgb
                    color['hex'] = f"#{r:02x}{g:02x}{b:02x}"
                    
                    # Update color name with high precision matching
                    color['name'] = self._get_precise_color_name(closest_rgb)
                    
                    # Update shades with precise values
                    color['shades'] = self._generate_precise_shades(closest_rgb)
                
        # Ensure minimum of 10 colors
        if len(gemini_result['dominant_colors']) < 10:
            # Add more colors from precision analysis
            additional_colors = self._precision_color_analysis(original_image)
            existing_rgbs = [tuple(c['rgb']) for c in gemini_result['dominant_colors']]
            
            for new_color in additional_colors['dominant_colors']:
                new_rgb = tuple(new_color['rgb'])
                if not any(self._color_distance(new_rgb, existing_rgb) < 15 for existing_rgb in existing_rgbs):
                    gemini_result['dominant_colors'].append(new_color)
                    existing_rgbs.append(new_rgb)
                    
                    if len(gemini_result['dominant_colors']) >= 15:
                        break
        
        # Sort by proportion
        gemini_result['dominant_colors'].sort(key=lambda x: x['proportion'], reverse=True)
        
        return gemini_result

    def _find_closest_real_color(self, r: int, g: int, b: int, image: np.ndarray) -> List[int]:
        """Find the closest actual color in the image to the specified RGB."""
        # Convert image to appropriate shape for processing
        pixels = image.reshape(-1, 3)
        
        # Convert target to LAB
        target_bgr = np.uint8([[[b, g, r]]])
        target_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB)[0, 0]
        
        # Calculate color distances to a sample of pixels
        if len(pixels) > 5000:
            # Sample pixels for efficiency
            indices = np.random.choice(len(pixels), 5000, replace=False)
            sample = pixels[indices]
        else:
            sample = pixels
            
        # Convert sample to LAB
        sample_bgr = sample[:, [2, 1, 0]]  # Convert RGB to BGR
        sample_lab = np.zeros_like(sample_bgr)
        for i, bgr in enumerate(sample_bgr):
            lab = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0, 0]
            sample_lab[i] = lab
            
        # Calculate Delta E (color difference)
        delta_e = np.sqrt(np.sum((sample_lab - target_lab)**2, axis=1))
        
        # Get the closest color
        closest_idx = np.argmin(delta_e)
        closest_rgb = sample[closest_idx].tolist()
        
        return closest_rgb

    def _analyze_with_gemini(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze colors using Google's Gemini API with token-optimized prompting."""
        try:
            # Further reduce image size to minimize tokens while keeping quality
            max_dim = 200  # Even smaller image = fewer tokens
            width, height = image.size
            if width > max_dim or height > max_dim:
                if width > height:
                    new_width = max_dim
                    new_height = int(height * (max_dim / width))
                else:
                    new_height = max_dim
                    new_width = int(width * (max_dim / height))
                image = image.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert to JPEG with higher compression to reduce size further
            buffer = io.BytesIO()
            image.convert('RGB').save(buffer, format="JPEG", quality=75)
            buffer.seek(0)
            optimized_image = Image.open(buffer)
            
            # Extremely concise prompt to minimize tokens
            prompt = """
            List exactly 10 main colors in this image with RGB values, hex codes, and proportions.
            Output only valid JSON:
            {
              "dominant_colors": [
                {
                  "name": "color name",
                  "rgb": [R,G,B],
                  "hex": "#RRGGBB",
                  "proportion": 0.45
                }
              ],
              "overall_brightness": 0.75,
              "color_contrast": 0.65
            }
            """
            
            # Configure for minimal token usage
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",  # Lowest-cost Gemini model
                generation_config={
                    "max_output_tokens": 600,
                    "temperature": 0.0,
                    "response_mime_type": "application/json"  # Explicitly request JSON response
                }
            )
            
            # Generate content
            response = model.generate_content([prompt, optimized_image])
            
            # Extract and parse JSON
            result_text = response.text
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_content = result_text[json_start:json_end]
                try:
                    result = json.loads(json_content)
                    # Enhance the basic results with our local processing
                    enhanced_result = self._enhance_gemini_result(result, image)
                    return enhanced_result
                except json.JSONDecodeError:
                    logger.error("Failed to parse JSON from Gemini response, falling back to local analysis")
                    return self._precision_color_analysis(np.array(image))
            else:
                logger.error("No JSON found in Gemini response, falling back to local analysis")
                return self._precision_color_analysis(np.array(image))
                
        except Exception as e:
            logger.error(f"Error in Gemini color analysis: {str(e)}")
            return None
            
    def _enhance_gemini_result(self, gemini_result: Dict, image: Image.Image) -> Dict:
        """Enhance basic Gemini results with locally-generated precise color info."""
        # Make sure dominant_colors exists
        if "dominant_colors" not in gemini_result or not gemini_result["dominant_colors"]:
            return self._precision_color_analysis(np.array(image))
            
        # For each color, add precise name and generate shades locally
        for color in gemini_result.get("dominant_colors", []):
            if "rgb" in color and isinstance(color["rgb"], list) and len(color["rgb"]) == 3:
                # Get better color name
                color["name"] = self._get_precise_color_name(color["rgb"])
                
                # Generate shades locally instead of asking Gemini
                color["shades"] = self._generate_precise_shades(color["rgb"])
                
        return gemini_result

    def _precision_color_analysis(self, image: np.ndarray) -> Dict:
        """Perform high-precision color analysis using efficient local methods."""
        try:
            # Convert to LAB color space for better perceptual analysis
            lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            
            # Reshape for clustering
            pixels_lab = lab_image.reshape(-1, 3).astype(np.float32)
            
            # Determine optimal number of clusters
            n_colors = min(15, len(np.unique(image.reshape(-1, 3), axis=0)) // 50 + 5)
            
            # Use MiniBatchKMeans for better efficiency
            kmeans = MiniBatchKMeans(n_clusters=n_colors, batch_size=1000, random_state=42, n_init=3)
            labels = kmeans.fit_predict(pixels_lab)
            
            # Convert cluster centers back to RGB
            centers_lab = kmeans.cluster_centers_.astype(np.uint8)
            centers_rgb = []
            
            # Process batch conversion for efficiency
            for lab_center in centers_lab:
                # Reshape for cv2.cvtColor
                lab_pixel = np.array([[[lab_center[0], lab_center[1], lab_center[2]]]], dtype=np.uint8)
                rgb_pixel = cv2.cvtColor(lab_pixel, cv2.COLOR_LAB2RGB)
                centers_rgb.append(rgb_pixel[0, 0])
            
            # Get proportions
            counts = np.bincount(labels, minlength=n_colors)
            proportions = counts / len(labels)
            
            # Process colors with high precision
            dominant_colors = []
            for i, (rgb, proportion) in enumerate(zip(centers_rgb, proportions)):
                rgb = rgb.tolist()
                hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb)
                
                # Get precise color name
                color_name = self._get_precise_color_name(rgb)
                
                # Generate accurate shades
                shades = self._generate_precise_shades(rgb)
                
                dominant_colors.append({
                    'name': color_name,
                    'rgb': rgb,
                    'hex': hex_color,
                    'proportion': float(proportion),
                    'shades': shades
                })
            
            # Sort by proportion
            dominant_colors.sort(key=lambda x: x['proportion'], reverse=True)
            
            # Take top colors but ensure we have at least 10
            result_colors = dominant_colors[:min(10, len(dominant_colors))]
            
            # Calculate statistics
            return {
                'dominant_colors': result_colors,
                'overall_brightness': float(np.mean(image) / 255.0),
                'color_contrast': float(np.std(image) / 255.0)
            }
        except Exception as e:
            logger.error(f"Error in precision color analysis: {str(e)}")
            # Return a minimal valid structure
            return {
                'dominant_colors': [],
                'overall_brightness': 0.5,
                'color_contrast': 0.5
            }

    def _generate_precise_shades(self, rgb: List[int]) -> List[Dict]:
        """Generate precise lighter and darker shades based on color theory."""
        r, g, b = rgb if isinstance(rgb, list) else rgb.tolist()
        
        # Convert to LAB for perceptual adjustments
        bgr = np.uint8([[[b, g, r]]])
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0, 0]
        
        # Create lighter shade by increasing L value
        lighter_lab = lab.copy()
        lighter_lab[0] = min(255, lighter_lab[0] + 30)  # Increase lightness
        
        # Create darker shade by decreasing L value
        darker_lab = lab.copy()
        darker_lab[0] = max(0, darker_lab[0] - 30)  # Decrease lightness
        
        # Convert back to RGB
        lighter_bgr = cv2.cvtColor(np.uint8([[[lighter_lab[0], lighter_lab[1], lighter_lab[2]]]]), cv2.COLOR_LAB2BGR)
        darker_bgr = cv2.cvtColor(np.uint8([[[darker_lab[0], darker_lab[1], darker_lab[2]]]]), cv2.COLOR_LAB2BGR)
        
        # Convert BGR to RGB
        lighter_rgb = lighter_bgr[0, 0][::-1].tolist()
        darker_rgb = darker_bgr[0, 0][::-1].tolist()
        
        # Get names for shades
        base_name = self._get_precise_color_name(rgb)
        
        return [
            {
                'name': f"Light {base_name}",
                'rgb': lighter_rgb,
                'hex': '#{:02x}{:02x}{:02x}'.format(*lighter_rgb)
            },
            {
                'name': f"Dark {base_name}",
                'rgb': darker_rgb,
                'hex': '#{:02x}{:02x}{:02x}'.format(*darker_rgb)
            }
        ]

    def _get_precise_color_name(self, rgb: List[int]) -> str:
        """Get precise color name using Delta-E color difference algorithm."""
        r, g, b = rgb if isinstance(rgb, list) else rgb.tolist()
        
        # Convert to LAB color space
        bgr = np.uint8([[[b, g, r]]])
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0, 0]
        
        # Calculate Delta E 2000 (perceptual color difference) for all reference colors
        min_delta_e = float('inf')
        closest_color = 'Unknown'
        
        for name, ref_lab in self.reference_lab.items():
            # Calculate color difference
            delta_e = np.sqrt(np.sum((lab - ref_lab)**2))
            
            if delta_e < min_delta_e:
                min_delta_e = delta_e
                closest_color = name
        
        # Add descriptive modifiers based on HSV values
        h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
        
        # Determine modifiers
        modifiers = []
        
        # Lightness modifiers
        if v < 0.2:
            modifiers.append("Very Dark")
        elif v < 0.4:
            modifiers.append("Dark")
        elif v > 0.9:
            modifiers.append("Very Light")
        elif v > 0.7:
            modifiers.append("Light")
            
        # Saturation modifiers
        if s < 0.1:
            if v > 0.7:
                modifiers = ["Pale"]
            elif v < 0.3:
                modifiers = ["Charcoal"]
            else:
                modifiers = ["Grayish"]
        elif s < 0.3:
            modifiers.append("Muted")
        elif s > 0.9:
            modifiers.append("Vibrant")
        elif s > 0.7:
            modifiers.append("Rich")
            
        # Warmth modifiers (based on hue)
        if 0.95 <= h or h < 0.1:  # Reds
            if s > 0.6 and v > 0.6:
                modifiers.append("Warm")
        elif 0.5 <= h < 0.65:  # Blues
            if s > 0.4:
                modifiers.append("Cool")
                
        # Combine modifiers with color name
        if modifiers:
            return f"{' '.join(modifiers)} {closest_color}"
        return closest_color

    def _color_distance(self, rgb1: Tuple[int, int, int], rgb2: Tuple[int, int, int]) -> float:
        """Calculate perceptually accurate color distance."""
        # Convert colors to LAB space
        bgr1 = np.uint8([[[rgb1[2], rgb1[1], rgb1[0]]]])
        bgr2 = np.uint8([[[rgb2[2], rgb2[1], rgb2[0]]]])
        
        lab1 = cv2.cvtColor(bgr1, cv2.COLOR_BGR2LAB)[0, 0]
        lab2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2LAB)[0, 0]
        
        # Calculate Delta E
        return np.sqrt(np.sum((lab1 - lab2)**2))

    def _rgb_to_hsv(self, rgb: List[int]) -> Tuple[float, float, float]:
        """Convert RGB to HSV."""
        return colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255) 