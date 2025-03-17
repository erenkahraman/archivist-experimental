import os
import logging
import json
from typing import Dict, Any, List
import openai
from PIL import Image
import base64
import io
from .config_prompts import OPENAI_CONFIG

logger = logging.getLogger(__name__)

class OpenAIAnalyzer:
    """Class to handle pattern analysis using OpenAI's API"""
    
    def __init__(self, api_key=None):
        """Initialize with OpenAI API key"""
        # Use provided API key or get from environment
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("No OpenAI API key provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        else:
            openai.api_key = self.api_key
    
    def set_api_key(self, api_key: str):
        """Set or update the OpenAI API key"""
        self.api_key = api_key
        openai.api_key = api_key
        logger.info("OpenAI API key updated")
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze an image using OpenAI's Vision API to identify patterns and details
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing pattern analysis results
        """
        try:
            if not self.api_key:
                logger.error("OpenAI API key not set")
                return self._get_default_response()
            
            # Encode image to base64
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                
            # Prepare the prompt for pattern analysis
            prompt = """
            Analyze this image and provide detailed information about any patterns present.
            Focus on the following aspects:
            1. Primary pattern category (geometric, floral, abstract, etc.)
            2. Secondary pattern types if present
            3. Specific elements in the pattern (e.g., roses, triangles)
            4. Pattern layout and distribution
            5. Pattern density and scale
            
            Format your response as a structured JSON with the following fields:
            {
                "category": "primary pattern category",
                "category_confidence": 0.95,
                "secondary_patterns": [
                    {"name": "pattern name", "confidence": 0.8},
                    {"name": "pattern name", "confidence": 0.6}
                ],
                "elements": [
                    {"name": "element name", "confidence": 0.9}
                ],
                "density": {
                    "type": "dense/scattered/regular",
                    "confidence": 0.85
                },
                "layout": {
                    "type": "layout type",
                    "confidence": 0.8
                },
                "prompt": {
                    "final_prompt": "A detailed description of the pattern that could be used as a prompt"
                }
            }
            """
            
            # Call OpenAI API using configuration from config_prompts.py
            response = openai.chat.completions.create(
                model=OPENAI_CONFIG['model'],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64.b64encode(image_data).decode('utf-8')}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=OPENAI_CONFIG['max_tokens'],
                temperature=OPENAI_CONFIG['temperature']
            )
            
            # Extract and parse the JSON response
            result_text = response.choices[0].message.content
            # Find JSON content in the response (it might be wrapped in markdown code blocks)
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_content = result_text[json_start:json_end]
                try:
                    result = json.loads(json_content)
                    # Ensure the result has all required fields
                    return self._validate_and_fix_response(result)
                except json.JSONDecodeError:
                    logger.error("Failed to parse JSON from OpenAI response")
                    return self._get_default_response()
            else:
                logger.error("No JSON found in OpenAI response")
                return self._get_default_response()
                
        except Exception as e:
            logger.error(f"Error in OpenAI analysis: {str(e)}")
            return self._get_default_response()
    
    def _validate_and_fix_response(self, response: Dict) -> Dict:
        """Validate and fix the OpenAI response to ensure it has all required fields"""
        default = self._get_default_response()
        
        # Ensure all required fields exist
        for key in default.keys():
            if key not in response:
                response[key] = default[key]
        
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
            if "confidence" not in element:
                element["confidence"] = 0.8
            elif isinstance(element["confidence"], str):
                try:
                    element["confidence"] = float(element["confidence"])
                except:
                    element["confidence"] = 0.8
        
        # Add fields expected by the gallery component
        response["primary_pattern"] = response.get("category", "Unknown")
        
        # Ensure pattern_confidence is a valid number
        if "category_confidence" in response and response["category_confidence"] is not None:
            response["pattern_confidence"] = float(response["category_confidence"])
        else:
            response["pattern_confidence"] = 0.8
        
        return response
    
    def _get_default_response(self) -> Dict[str, Any]:
        """Return a default response when analysis fails"""
        return {
            "category": "Unknown",
            "category_confidence": 0.0,
            "primary_pattern": "Unknown",
            "pattern_confidence": 0.0,
            "secondary_patterns": [],
            "elements": [],
            "density": {
                "type": "regular",
                "confidence": 0.0
            },
            "layout": {
                "type": "balanced",
                "confidence": 0.0
            },
            "prompt": {
                "final_prompt": "Unable to analyze pattern",
                "components": {},
                "completeness_score": 0
            }
        } 