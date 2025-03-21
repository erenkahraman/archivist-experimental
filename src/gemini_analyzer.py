import os
import logging
import json
from typing import Dict, Any, List
import google.generativeai as genai
from PIL import Image
import base64
import io
from .config_prompts import GEMINI_CONFIG

logger = logging.getLogger(__name__)

class GeminiAnalyzer:
    """Class to handle pattern analysis using Google's Gemini API"""
    
    def __init__(self, api_key=None):
        """Initialize with Gemini API key"""
        # Use provided API key or get from environment
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning("No Gemini API key provided. Set GEMINI_API_KEY environment variable or pass api_key parameter.")
        else:
            # Mask API key in logs and never store the full key in instance variables
            masked_key = self._mask_api_key(self.api_key)
            logger.info(f"Using Gemini API key: {masked_key}")
            # Configure the API client but don't store the raw key
            genai.configure(api_key=self.api_key)
            # Don't store the actual key, just a flag that we have one
            self.api_key = True
    
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
            # Don't store the actual key, just a flag that we have one
            self.api_key = True
            logger.info("Gemini API key updated")
        else:
            logger.warning("Attempted to set empty API key")
            self.api_key = False
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze an image using Google's Gemini API to identify patterns and details
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing pattern analysis results
        """
        try:
            if not self.api_key or self.api_key is not True:
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
            
            # Prepare the prompt for pattern analysis
            prompt = """
            Analyze this image and provide a detailed analysis of any patterns present. Focus on the following aspects:

            1. **Primary Pattern Category:**  
               Identify the main pattern category (e.g., geometric, floral, abstract, animal print, etc.) and provide a confidence score (0 to 1).

            2. **Secondary Pattern Types:**  
               List any additional pattern types that appear in the image, each with its own confidence score.

            3. **Specific Elements:**  
               For each key element detected (e.g., a flower, a geometric shape, or an animal skin print), provide:  
               - The exact element name (e.g., "roses", "tulips", "circles", "leopard skin").  
               - For animal prints specifically, include the precise animal type (e.g., "leopard", "zebra", "tiger", "giraffe", "snake", "crocodile") and the characteristic textural detail (e.g., "distinctive rosettes", "bold stripes").  
               - A detailed sub-category if applicable (e.g., "garden roses", "double tulips").  
               - The dominant color or color description of the element (e.g., "vibrant pink", "pastel blue", "warm brown with black rosettes").  
               - A confidence score for the detection.

            4. **Layout and Distribution:**  
               Describe how the elements are arranged (e.g., scattered, clustered, symmetrical, trailing vines) and provide a confidence score.

            5. **Density and Scale:**  
               Specify the pattern density (e.g., dense, sparse, regular) and the scale of the elements (e.g., small, medium, large) with corresponding confidence scores.

            6. **Texture Type:**
               Describe the texture quality of the pattern (e.g., smooth, rough, embossed, flat, glossy, matte) with a confidence score.

            7. **Cultural and Historical Context:**
               - Identify any cultural influences in the pattern (e.g., Japanese, Moroccan, Scandinavian, Art Deco) with a confidence score.
               - Suggest a historical period the pattern might be associated with (e.g., Victorian, Mid-Century Modern, Contemporary) with a confidence score.
               - Describe the mood or emotional quality the pattern evokes (e.g., calm, energetic, sophisticated, playful) with a confidence score.

            8. **Style Keywords:**
               Provide 3-5 descriptive style keywords that best characterize the pattern (e.g., "minimalist", "bohemian", "tropical", "industrial").

            9. **Prompt Description:**  
               Finally, generate a coherent, human-readable prompt that summarizes the pattern. This description should integrate the above details into a fluid sentence. For example:  
               "Elegant floral pattern featuring vibrant pink garden roses and delicate pastel tulips, combined with bold leopard skin print exhibiting distinctive rosettes, arranged in a trailing vines layout with a dense overall distribution at a large scale."

            Format your response as a structured JSON with the following fields:

            {
              "category": "primary pattern category",
              "category_confidence": 0.95,
              "secondary_patterns": [
                {"name": "pattern name", "confidence": 0.8},
                {"name": "pattern name", "confidence": 0.6}
              ],
              "elements": [
                {
                  "name": "element name",
                  "sub_category": "detailed element type",
                  "color": "dominant color description",
                  "animal_type": "if applicable, specify animal type for animal prints",
                  "textural_detail": "if applicable, describe specific pattern detail (e.g., rosettes, stripes)",
                  "confidence": 0.9
                }
              ],
              "density": {
                "type": "dense/scattered/regular",
                "confidence": 0.85
              },
              "layout": {
                "type": "layout type",
                "confidence": 0.8
              },
              "scale": {
                "type": "small/medium/large",
                "confidence": 0.8
              },
              "texture_type": {
                "type": "texture description (e.g., smooth, rough, embossed)",
                "confidence": 0.8
              },
              "cultural_influence": {
                "type": "cultural style (e.g., Japanese, Moroccan, Art Deco)",
                "confidence": 0.7
              },
              "historical_period": {
                "type": "historical era (e.g., Victorian, Mid-Century, Contemporary)",
                "confidence": 0.7
              },
              "mood": {
                "type": "emotional quality (e.g., calm, energetic, sophisticated)",
                "confidence": 0.8
              },
              "style_keywords": ["keyword1", "keyword2", "keyword3"],
              "prompt": {
                "final_prompt": "A detailed, coherent description of the pattern"
              }
            }
            """
            
            # Configure the model
            model = genai.GenerativeModel(
                model_name=GEMINI_CONFIG['model'],
                generation_config={
                    "max_output_tokens": GEMINI_CONFIG['max_tokens'],
                    "temperature": GEMINI_CONFIG['temperature'],
                    "response_mime_type": GEMINI_CONFIG.get('response_mime_type', 'application/json')
                }
            )
            
            # Generate content with the image
            response = model.generate_content([prompt, optimized_image])
            
            # Extract and parse the JSON response
            result_text = response.text
            
            # Try to find JSON within the response
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_content = result_text[json_start:json_end]
                try:
                    # Parse JSON and validate result
                    result = json.loads(json_content)
                    
                    # Add image metadata
                    result["dimensions"] = {"width": width, "height": height}
                    result["original_path"] = image_path
                    
                    # Validate and fix result
                    return self._validate_and_fix_response(result)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from Gemini response: {str(e)}")
                    logger.error(f"JSON content attempted to parse: {json_content[:100]}...")
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
        response["primary_pattern"] = response.get("category", "Unknown")
        
        # Ensure pattern_confidence is a valid number
        if "category_confidence" in response and response["category_confidence"] is not None:
            response["pattern_confidence"] = float(response["category_confidence"])
        else:
            response["pattern_confidence"] = 0.8
        
        return response
    
    def _get_default_response(self, image_path: str, width: int, height: int) -> Dict[str, Any]:
        """Return a default response when analysis fails"""
        return {
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