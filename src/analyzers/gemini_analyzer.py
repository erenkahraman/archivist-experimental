import os
import logging
import json
from typing import Dict, Any, List
import google.generativeai as genai
from PIL import Image
import base64
import io
import time
from src.config.prompts import GEMINI_CONFIG

logger = logging.getLogger(__name__)

class GeminiAnalyzer:
    """Class to handle pattern analysis using Google's Gemini API"""
    
    def __init__(self, api_key=None):
        """Initialize with Gemini API key"""
        # Use provided API key or get from environment
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self._api_key:
            logger.warning("No Gemini API key provided. Set GEMINI_API_KEY environment variable or pass api_key parameter.")
        else:
            # Mask API key in logs and never store the full key in instance variables
            masked_key = self._mask_api_key(self._api_key)
            logger.info(f"Using Gemini API key: {masked_key}")
            # Configure the API client but don't store the raw key
            genai.configure(api_key=self._api_key)
    
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
            logger.info("Gemini API key updated")
        else:
            logger.warning("Attempted to set empty API key")
            self._api_key = None
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze an image using Google's Gemini API to identify patterns and details
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing pattern analysis results
        """
        try:
            if not self._api_key:
                logger.error("Gemini API key not set")
                return self._get_default_response()
            
            # Load the image and resize for token efficiency
            image = Image.open(image_path)
            
            # Get image dimensions
            width, height = image.size
            
            # Resize to smaller dimensions for token efficiency
            max_dim = 220
            if width > max_dim or height > max_dim:
                if width > height:
                    new_width = max_dim
                    new_height = int(height * (max_dim / width))
                else:
                    new_height = max_dim
                    new_width = int(width * (max_dim / height))
                image = image.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert to JPEG with compression for token reduction
            buffer = io.BytesIO()
            image.convert('RGB').save(buffer, format="JPEG", quality=75)
            buffer.seek(0)
            optimized_image = Image.open(buffer)
            
            # Get file name from path
            file_name = os.path.basename(image_path)
            
            # Define the analysis prompt
            analysis_prompt = f"""
            Identify the most specific textile pattern with high precision, capturing detailed motifs and cultural cues even if they aren't explicitly listed.

            Taxonomy:
            - BASE: geometric, floral, abstract, figurative, ethnic/tribal, typographic.
            - SPECIFIC (non-exhaustive): paisley, batik, chevron, ditsy, lace, polka dot, plaid, tartan, ikat, toile, damask, patchwork, houndstooth, gingham, border, tropical, folk, brocade, graffiti, doodle.

            Output JSON:
            {{
              "main_theme": "Most specific pattern name",
              "main_theme_confidence": 0.95,
              "category": "Base category",
              "category_confidence": 0.95,
              "secondary_patterns": [{{"name": "One secondary pattern if clearly present", "confidence": 0.9}}],
              "style_keywords": ["up to 5 relevant keywords"],
              "prompt": {{"final_prompt": "Concise detailed description including motifs, arrangement, and cultural cues"}}
            }}

            Rules:
            1. Always select the most specific pattern name if confidence >90%.
            2. Do not default to generic base terms (e.g., 'abstract') if further detail is available.
            3. Generate a descriptive, specific name using detailed cues when the pattern doesn't exactly match the predefined list.
            4. Include one secondary pattern only if clearly distinguishable.
            5. Limit style_keywords to a maximum of 5 that capture the unique design details.
            6. Report only high-confidence identifications.
            """
            
            # Set up the Gemini model
            generation_config = {
                "temperature": 0.0,
                "top_p": 0.95,
                "top_k": 10,
                "max_output_tokens": 1024,
            }
            
            # Load the model
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                generation_config=generation_config
            )
            
            # Generate content
            response = model.generate_content([analysis_prompt, optimized_image])
            
            # Extract and parse JSON
            result_text = response.text
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_content = result_text[json_start:json_end]
                try:
                    result = json.loads(json_content)
                    # Add image dimensions and path
                    result["dimensions"] = {"width": width, "height": height}
                    result["original_path"] = image_path
                    # Validate and fix the response
                    validated_result = self._validate_and_fix_response(result)
                    return validated_result
                except json.JSONDecodeError:
                    logger.error("Failed to parse JSON from Gemini response")
                    return self._get_default_response(image_path, width, height)
            else:
                logger.error("No JSON found in Gemini response")
                return self._get_default_response(image_path, width, height)
                
        except Exception as e:
            logger.error(f"Error in Gemini analysis: {str(e)}")
            # Try to get dimensions if image was loaded
            width, height = 0, 0
            try:
                if 'image' in locals() and image:
                    width, height = image.size
            except:
                pass
            return self._get_default_response(image_path, width, height)
    
    def _validate_and_fix_response(self, response: Dict) -> Dict:
        """Validate and fix the Gemini response to ensure it has all required fields"""
        default = self._get_default_response("", 0, 0)
        
        # Ensure all required fields exist
        for key in default.keys():
            if key not in response:
                response[key] = default[key]
        
        # Ensure new fields exist
        if "main_theme" not in response:
            response["main_theme"] = response.get("category", "Unknown")
        
        if "main_theme_confidence" not in response:
            response["main_theme_confidence"] = response.get("category_confidence", 0.8)
        
        if "content_details" not in response or not isinstance(response["content_details"], list):
            response["content_details"] = []
            # Generate from elements if available
            if "elements" in response and isinstance(response["elements"], list):
                for element in response["elements"]:
                    if isinstance(element, dict) and "name" in element:
                        response["content_details"].append({
                            "name": element["name"],
                            "confidence": element.get("confidence", 0.7)
                        })
        
        if "stylistic_attributes" not in response or not isinstance(response["stylistic_attributes"], list):
            response["stylistic_attributes"] = []
            # Generate from style_keywords if available
            if "style_keywords" in response and isinstance(response["style_keywords"], list):
                for keyword in response["style_keywords"]:
                    response["stylistic_attributes"].append({
                        "name": keyword,
                        "confidence": 0.7
                    })
        
        # Ensure dimensions has proper structure
        if not isinstance(response.get("dimensions"), dict):
            response["dimensions"] = {"width": 0, "height": 0}
        else:
            if "width" not in response["dimensions"]:
                response["dimensions"]["width"] = 0
            if "height" not in response["dimensions"]:
                response["dimensions"]["height"] = 0
        
        # Ensure original_path exists
        if "original_path" not in response:
            response["original_path"] = ""
        
        # Ensure confidence values are floats between 0 and 1
        if isinstance(response.get("category_confidence"), str):
            try:
                response["category_confidence"] = float(response["category_confidence"])
            except:
                response["category_confidence"] = 0.8
        elif response.get("category_confidence") is None:
            response["category_confidence"] = 0.8
        
        # Ensure main_theme_confidence is a float
        if isinstance(response.get("main_theme_confidence"), str):
            try:
                response["main_theme_confidence"] = float(response["main_theme_confidence"])
            except:
                response["main_theme_confidence"] = 0.8
        elif response.get("main_theme_confidence") is None:
            response["main_theme_confidence"] = 0.8
        
        # Validate content_details items
        for item in response.get("content_details", []):
            if not isinstance(item, dict):
                continue
            
            if "name" not in item:
                item["name"] = ""
            
            if "confidence" not in item:
                item["confidence"] = 0.7
            elif isinstance(item["confidence"], str):
                try:
                    item["confidence"] = float(item["confidence"])
                except:
                    item["confidence"] = 0.7
        
        # Validate stylistic_attributes items
        for item in response.get("stylistic_attributes", []):
            if not isinstance(item, dict):
                continue
            
            if "name" not in item:
                item["name"] = ""
            
            if "confidence" not in item:
                item["confidence"] = 0.7
            elif isinstance(item["confidence"], str):
                try:
                    item["confidence"] = float(item["confidence"])
                except:
                    item["confidence"] = 0.7
        
        # Ensure secondary_patterns is a list with proper structure
        if not isinstance(response.get("secondary_patterns"), list):
            response["secondary_patterns"] = []
        
        for pattern in response.get("secondary_patterns", []):
            if not isinstance(pattern, dict):
                continue
            if "confidence" not in pattern:
                pattern["confidence"] = 0.7
            elif isinstance(pattern["confidence"], str):
                try:
                    pattern["confidence"] = float(pattern["confidence"])
                except:
                    pattern["confidence"] = 0.7
        
        # Ensure elements is a list with proper structure
        if not isinstance(response.get("elements"), list):
            response["elements"] = []
        
        for element in response.get("elements", []):
            if not isinstance(element, dict):
                continue
            # Ensure all required element fields exist
            for field in ["name", "sub_category", "color", "confidence"]:
                if field not in element:
                    element[field] = "" if field != "confidence" else 0.8
            
            # Add animal-specific fields if not present
            if "animal_type" not in element:
                element["animal_type"] = ""
            if "textural_detail" not in element:
                element["textural_detail"] = ""
                
            # Ensure confidence is a float
            if isinstance(element["confidence"], str):
                try:
                    element["confidence"] = float(element["confidence"])
                except:
                    element["confidence"] = 0.8
        
        # Ensure density has proper structure
        if not isinstance(response.get("density"), dict):
            response["density"] = default["density"]
        else:
            if "type" not in response["density"]:
                response["density"]["type"] = "regular"
            if "confidence" not in response["density"]:
                response["density"]["confidence"] = 0.7
            elif isinstance(response["density"]["confidence"], str):
                try:
                    response["density"]["confidence"] = float(response["density"]["confidence"])
                except:
                    response["density"]["confidence"] = 0.7
        
        # Ensure layout has proper structure
        if not isinstance(response.get("layout"), dict):
            response["layout"] = default["layout"]
        else:
            if "type" not in response["layout"]:
                response["layout"]["type"] = "balanced"
            if "confidence" not in response["layout"]:
                response["layout"]["confidence"] = 0.7
            elif isinstance(response["layout"]["confidence"], str):
                try:
                    response["layout"]["confidence"] = float(response["layout"]["confidence"])
                except:
                    response["layout"]["confidence"] = 0.7
        
        # Ensure scale has proper structure
        if not isinstance(response.get("scale"), dict):
            response["scale"] = {"type": "medium", "confidence": 0.7}
        else:
            if "type" not in response["scale"]:
                response["scale"]["type"] = "medium"
            if "confidence" not in response["scale"]:
                response["scale"]["confidence"] = 0.7
            elif isinstance(response["scale"]["confidence"], str):
                try:
                    response["scale"]["confidence"] = float(response["scale"]["confidence"])
                except:
                    response["scale"]["confidence"] = 0.7
        
        # Ensure texture_type has proper structure
        if not isinstance(response.get("texture_type"), dict):
            response["texture_type"] = {"type": "smooth", "confidence": 0.7}
        else:
            if "type" not in response["texture_type"]:
                response["texture_type"]["type"] = "smooth"
            if "confidence" not in response["texture_type"]:
                response["texture_type"]["confidence"] = 0.7
            elif isinstance(response["texture_type"]["confidence"], str):
                try:
                    response["texture_type"]["confidence"] = float(response["texture_type"]["confidence"])
                except:
                    response["texture_type"]["confidence"] = 0.7
        
        # Ensure cultural_influence has proper structure
        if not isinstance(response.get("cultural_influence"), dict):
            response["cultural_influence"] = {"type": "contemporary", "confidence": 0.7}
        else:
            if "type" not in response["cultural_influence"]:
                response["cultural_influence"]["type"] = "contemporary"
            if "confidence" not in response["cultural_influence"]:
                response["cultural_influence"]["confidence"] = 0.7
            elif isinstance(response["cultural_influence"]["confidence"], str):
                try:
                    response["cultural_influence"]["confidence"] = float(response["cultural_influence"]["confidence"])
                except:
                    response["cultural_influence"]["confidence"] = 0.7
        
        # Ensure historical_period has proper structure
        if not isinstance(response.get("historical_period"), dict):
            response["historical_period"] = {"type": "modern", "confidence": 0.7}
        else:
            if "type" not in response["historical_period"]:
                response["historical_period"]["type"] = "modern"
            if "confidence" not in response["historical_period"]:
                response["historical_period"]["confidence"] = 0.7
            elif isinstance(response["historical_period"]["confidence"], str):
                try:
                    response["historical_period"]["confidence"] = float(response["historical_period"]["confidence"])
                except:
                    response["historical_period"]["confidence"] = 0.7
        
        # Ensure mood has proper structure
        if not isinstance(response.get("mood"), dict):
            response["mood"] = {"type": "neutral", "confidence": 0.7}
        else:
            if "type" not in response["mood"]:
                response["mood"]["type"] = "neutral"
            if "confidence" not in response["mood"]:
                response["mood"]["confidence"] = 0.7
            elif isinstance(response["mood"]["confidence"], str):
                try:
                    response["mood"]["confidence"] = float(response["mood"]["confidence"])
                except:
                    response["mood"]["confidence"] = 0.7
        
        # Ensure style_keywords is a list
        if not isinstance(response.get("style_keywords"), list):
            response["style_keywords"] = []
        
        # Add fields expected by the gallery component
        if "main_theme" in response and response["main_theme"]:
            # Check if "paisley" is in the main_theme
            if "paisley" in response["main_theme"].lower():
                response["primary_pattern"] = "paisley pattern"
            else:
                response["primary_pattern"] = response["main_theme"]
        else:
            response["primary_pattern"] = response.get("category", "Unknown")
        
        # Ensure pattern_confidence is a valid number
        if "main_theme_confidence" in response and response["main_theme_confidence"] is not None:
            response["pattern_confidence"] = float(response["main_theme_confidence"])
        elif "category_confidence" in response and response["category_confidence"] is not None:
            response["pattern_confidence"] = float(response["category_confidence"])
        else:
            response["pattern_confidence"] = 0.8
        
        return response
    
    def _get_default_response(self, image_path: str, width: int, height: int) -> Dict[str, Any]:
        """Return a default response when analysis fails"""
        return {
            "main_theme": "Unknown",
            "main_theme_confidence": 0.0,
            "content_details": [],
            "stylistic_attributes": [],
            "category": "Unknown",
            "category_confidence": 0.0,
            "primary_pattern": "Unknown",
            "pattern_confidence": 0.0,
            "secondary_patterns": [],
            "elements": [
                {
                    "name": "",
                    "sub_category": "",
                    "color": "",
                    "animal_type": "",
                    "textural_detail": "",
                    "confidence": 0.0
                }
            ],
            "density": {
                "type": "regular",
                "confidence": 0.0
            },
            "layout": {
                "type": "balanced",
                "confidence": 0.0
            },
            "scale": {
                "type": "medium",
                "confidence": 0.0
            },
            "texture_type": {
                "type": "",
                "confidence": 0.0
            },
            "cultural_influence": {
                "type": "",
                "confidence": 0.0
            },
            "historical_period": {
                "type": "",
                "confidence": 0.0
            },
            "mood": {
                "type": "",
                "confidence": 0.0
            },
            "style_keywords": [],
            "prompt": {
                "final_prompt": "Unable to analyze pattern"
            },
            "dimensions": {"width": width, "height": height},
            "original_path": image_path
        } 