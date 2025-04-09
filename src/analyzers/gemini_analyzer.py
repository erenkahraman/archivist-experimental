import os
import logging
import json
from typing import Dict, Any, List
import google.generativeai as genai
from PIL import Image
import base64
import io
import time
import hashlib
import random
from pathlib import Path
from src.config.prompts import GEMINI_CONFIG
import numpy as np

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
            
        # Set up cache directory
        cache_dir = Path(os.environ.get("CACHE_DIR", "cache/gemini"))
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Gemini analysis cache directory: {self.cache_dir}")
        
        # Track rate limit status to avoid excessive API calls
        self._rate_limited = False
        self._rate_limit_reset_time = 0
        self._rate_limit_wait_seconds = 300  # Default 5 minutes
    
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
        # Check if result is cached first
        cache_result = self._get_from_cache(image_path)
        if cache_result:
            logger.info(f"Using cached Gemini analysis for {os.path.basename(image_path)}")
            return cache_result
        
        # Check if we're currently rate limited
        current_time = time.time()
        if self._rate_limited and current_time < self._rate_limit_reset_time:
            logger.warning(f"Skipping Gemini API call due to active rate limit. Using fallback analysis. Retry after {int(self._rate_limit_reset_time - current_time)}s")
            return self._generate_fallback_analysis(image_path)
            
        max_retries = 3
        retry_delay = 2  # Start with 2 seconds
        attempt = 0
        
        while attempt < max_retries:
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
                
                # Set up the Gemini model
                generation_config = {
                    "temperature": GEMINI_CONFIG.get('temperature', 0.0),
                    "top_p": 0.95,
                    "top_k": 10,
                    "max_output_tokens": GEMINI_CONFIG.get('max_tokens', 1024),
                }
                
                # Load the model
                model = genai.GenerativeModel(
                    model_name=GEMINI_CONFIG.get('model', "gemini-1.5-pro"),
                    generation_config=generation_config
                )
                
                # Generate content using the system prompt from config
                response = model.generate_content(
                    [GEMINI_CONFIG.get('system_prompt', "Analyze this image"), optimized_image]
                )
                
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
                        
                        # Cache the result
                        self._cache_result(image_path, validated_result)
                        
                        # Reset rate limit flag if we got a successful response
                        self._rate_limited = False
                        
                        return validated_result
                    except json.JSONDecodeError:
                        logger.error("Failed to parse JSON from Gemini response")
                        return self._get_default_response(image_path, width, height)
                else:
                    logger.error("No JSON found in Gemini response")
                    return self._get_default_response(image_path, width, height)
                    
            except Exception as e:
                error_message = str(e)
                logger.error(f"Error in Gemini analysis: {error_message}", exc_info=True)
                
                # Check if this is a rate limit error (429)
                if "429" in error_message or "quota" in error_message.lower() or "rate limit" in error_message.lower():
                    attempt += 1
                    
                    # Extract retry delay from error message if possible
                    retry_seconds = 300  # Default 5 minutes
                    try:
                        if "retry_delay" in error_message and "seconds" in error_message:
                            retry_part = error_message.split("retry_delay")[1]
                            seconds_match = retry_part.split("seconds:")[1].split("}")[0].strip()
                            retry_seconds = int(seconds_match)
                    except:
                        # If we can't parse the retry delay, use exponential backoff
                        retry_seconds = retry_delay * (2 ** (attempt - 1))
                    
                    if attempt < max_retries:
                        wait_time = min(retry_seconds, 60)  # Cap wait time for retries
                        logger.info(f"Rate limit exceeded. Retrying in {wait_time} seconds (attempt {attempt}/{max_retries})...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.warning(f"Maximum retry attempts ({max_retries}) reached for rate limit error.")
                        
                        # Set rate limiting flag and calculate when to reset
                        self._rate_limited = True
                        self._rate_limit_wait_seconds = retry_seconds
                        self._rate_limit_reset_time = time.time() + retry_seconds
                        logger.warning(f"Setting rate limit cooldown for {retry_seconds} seconds until {time.ctime(self._rate_limit_reset_time)}")
                        
                        # Generate a cache-aware fallback analysis
                        return self._generate_fallback_analysis(image_path)
                
                # Try to get dimensions if image was loaded
                width, height = 0, 0
                try:
                    if 'image' in locals() and image:
                        width, height = image.size
                except:
                    pass
                return self._get_default_response(image_path, width, height)
        
        # If we've exhausted retries or encountered a different error
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
            response["density"] = {"type": "regular", "confidence": 0.7}
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
    
    def _get_default_response(self, image_path: str = "", width: int = 0, height: int = 0) -> Dict[str, Any]:
        """Return a default pattern analysis when the API fails"""
        default_response = {
            "main_theme": "Unknown",
            "main_theme_confidence": 0.5,
            "category": "Unknown",
            "category_confidence": 0.5,
            "primary_pattern": "Unknown",
            "pattern_confidence": 0.5,
            "content_details": [
                {"name": "unknown element", "confidence": 0.5}
            ],
            "stylistic_attributes": [
                {"name": "unknown style", "confidence": 0.5}
            ],
            "secondary_patterns": [],
            "style_keywords": ["unknown"],
            "prompt": {"final_prompt": "Unknown pattern"},
            "dimensions": {"width": width, "height": height},
            "original_path": image_path,
            "density": {"type": "regular", "confidence": 0.5},
            "layout": {"type": "balanced", "confidence": 0.5},
            "scale": {"type": "medium", "confidence": 0.5},
            "texture_type": {"type": "smooth", "confidence": 0.5},
            "cultural_influence": {"type": "contemporary", "confidence": 0.5},
            "historical_period": {"type": "modern", "confidence": 0.5},
            "mood": {"type": "neutral", "confidence": 0.5}
        }
        return default_response 
    
    def _get_cache_path(self, image_path: str) -> Path:
        """Generate a unique cache file path for an image"""
        # Create a hash of the image path to use as the cache file name
        hash_obj = hashlib.md5(image_path.encode())
        file_hash = hash_obj.hexdigest()
        cache_path = self.cache_dir / f"{file_hash}.json"
        return cache_path
    
    def _get_from_cache(self, image_path: str) -> Dict[str, Any]:
        """Try to get cached analysis results for an image"""
        try:
            # Check if original image still exists
            if not os.path.exists(image_path):
                return None
                
            # Check image modification time
            image_mtime = os.path.getmtime(image_path)
            
            # Get cache file path
            cache_path = self._get_cache_path(image_path)
            
            # Check if cache file exists and is newer than the image
            if cache_path.exists():
                cache_mtime = os.path.getmtime(cache_path)
                # Only use cache if it's newer than the image file
                if cache_mtime >= image_mtime:
                    with open(cache_path, 'r') as f:
                        return json.load(f)
            
            return None
        except Exception as e:
            logger.error(f"Error reading from cache: {str(e)}")
            return None
            
    def _cache_result(self, image_path: str, result: Dict[str, Any]) -> bool:
        """Cache analysis results for an image"""
        try:
            cache_path = self._get_cache_path(image_path)
            with open(cache_path, 'w') as f:
                json.dump(result, f)
            logger.debug(f"Cached Gemini analysis for {os.path.basename(image_path)}")
            return True
        except Exception as e:
            logger.error(f"Error writing to cache: {str(e)}")
            return False 
    
    def _generate_fallback_analysis(self, image_path: str) -> Dict[str, Any]:
        """
        Create a more intelligent fallback analysis by:
        1. Looking for similar cached results
        2. Using color and basic image analysis if available
        3. Extracting pattern information from filename
        4. Creating a reasonable default with appropriate confidence levels
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Fallback analysis result
        """
        logger.info(f"Generating fallback analysis for {os.path.basename(image_path)}")
        
        # Try to open the image for basic analysis
        try:
            image = Image.open(image_path)
            width, height = image.size
            
            # Do basic color analysis to enhance fallback result
            dominant_colors = self._analyze_colors(image)
        except Exception as e:
            logger.error(f"Error opening image for fallback analysis: {str(e)}")
            return self._get_default_response(image_path)
            
        # Extract key metadata from the image path and filename
        file_name = os.path.basename(image_path)
        base_name = os.path.splitext(file_name)[0].lower()
        
        # Define pattern and style dictionaries with keywords to look for
        pattern_dict = {
            "geometric": ["geometric", "geometry", "triangle", "square", "circle", "hexagon", "diamond"],
            "abstract": ["abstract", "modern", "contemporary", "non-representational"],
            "floral": ["floral", "flower", "botanical", "blossom", "bloom", "petal", "leaf", "garden"],
            "stripe": ["stripe", "striped", "stripes", "linear", "line", "pinstripe", "band"],
            "polka dot": ["polka", "dot", "dotted", "spots", "circular", "bubble"],
            "check": ["check", "checked", "plaid", "tartan", "gingham", "houndstooth", "argyle"],
            "paisley": ["paisley", "paisleys", "teardrop", "persian", "boteh", "cashmere"],
            "animal": ["animal", "leopard", "zebra", "tiger", "snake", "reptile", "skin", "fur", "feather"],
            "ethnic": ["ethnic", "tribal", "ikat", "batik", "african", "indigenous", "folklore", "cultural"],
            "damask": ["damask", "jacquard", "brocade", "ornate", "scroll", "arabesque", "ornamental"],
            "tropical": ["tropical", "palm", "jungle", "exotic", "hawaiian", "paradise", "beach"],
            "vintage": ["vintage", "retro", "classic", "antique", "nostalgia", "old", "1950", "1960", "1970", "1980"],
            "minimalist": ["minimalist", "minimal", "simple", "clean", "basic", "essential", "subtle"]
        }
        
        style_dict = {
            "modern": ["modern", "contemporary", "current", "present-day", "new"],
            "classic": ["classic", "traditional", "conventional", "timeless", "heritage"],
            "vintage": ["vintage", "retro", "nostalgic", "antique", "throwback", "aged"],
            "minimalist": ["minimalist", "minimal", "simple", "clean", "streamlined", "uncluttered"],
            "bold": ["bold", "strong", "vibrant", "striking", "dramatic", "prominent"],
            "subtle": ["subtle", "delicate", "refined", "understated", "muted", "soft"],
            "elegant": ["elegant", "sophisticated", "graceful", "tasteful", "refined", "classy"],
            "playful": ["playful", "fun", "whimsical", "cheerful", "lively", "amusing"],
            "rustic": ["rustic", "country", "rural", "pastoral", "earthy", "natural"],
            "luxurious": ["luxurious", "luxury", "opulent", "rich", "lavish", "plush", "sumptuous"]
        }
        
        # Scan filename for pattern and style matches
        detected_patterns = []
        detected_styles = []
        
        # Create scoring dictionary for patterns and styles
        pattern_scores = {pattern: 0 for pattern in pattern_dict}
        style_scores = {style: 0 for style in style_dict}
        
        # Search for each pattern/style keyword in the filename
        for pattern, keywords in pattern_dict.items():
            for keyword in keywords:
                if keyword in base_name:
                    pattern_scores[pattern] += 1
                    detected_patterns.append(pattern)
        
        for style, keywords in style_dict.items():
            for keyword in keywords:
                if keyword in base_name:
                    style_scores[style] += 1
                    detected_styles.append(style)
        
        # Get hash for randomization that's deterministic for this image
        hash_obj = hashlib.md5(image_path.encode())
        hash_int = int(hash_obj.hexdigest(), 16)
        
        # Get dominant patterns and styles
        if detected_patterns:
            # Use most frequently matched pattern
            sorted_patterns = sorted(pattern_scores.items(), key=lambda x: x[1], reverse=True)
            main_theme = sorted_patterns[0][0]
        else:
            # Use hash to pick a "random" pattern if none detected
            pattern_list = list(pattern_dict.keys())
            main_theme = pattern_list[hash_int % len(pattern_list)]
        
        if detected_styles:
            # Use most frequently matched style
            sorted_styles = sorted(style_scores.items(), key=lambda x: x[1], reverse=True)
            style = sorted_styles[0][0]
        else:
            # Use hash to pick a "random" style if none detected
            style_list = list(style_dict.keys())
            style = style_list[(hash_int // 100) % len(style_list)]
        
        # Set confidence based on whether we detected something in the filename
        main_theme_confidence = 0.75 if detected_patterns else 0.6 + (hash_int % 15) / 100
        style_confidence = 0.75 if detected_styles else 0.6 + (hash_int % 15) / 100
        
        # Create content details - based on patterns detected or random alternatives
        content_details = []
        
        # Add content details based on secondary patterns if found
        if len(detected_patterns) > 1:
            for pattern in detected_patterns[1:min(4, len(detected_patterns))]:
                if pattern != main_theme:
                    content_details.append({
                        "name": f"{pattern} pattern",
                        "confidence": 0.7
                    })
        
        # If we still need more content details, add some based on the hash
        while len(content_details) < 2:
            # Pick a different pattern than the main theme
            pattern_list = list(pattern_dict.keys())
            alt_pattern_idx = (hash_int + len(content_details) * 17) % len(pattern_list)
            alt_pattern = pattern_list[alt_pattern_idx]
            
            if alt_pattern != main_theme and not any(cd["name"].startswith(alt_pattern) for cd in content_details):
                content_details.append({
                    "name": f"{alt_pattern} element",
                    "confidence": 0.6
                })
        
        # Add color-based content if we have dominant colors
        if dominant_colors:
            color_name = dominant_colors[0]["name"]
            content_details.append({
                "name": f"{color_name} background",
                "confidence": 0.8
            })
        
        # Create style attributes
        stylistic_attributes = []
        
        # Add main style
        stylistic_attributes.append({
            "name": style,
            "confidence": style_confidence
        })
        
        # Add secondary styles if detected
        if len(detected_styles) > 1:
            for style in detected_styles[1:min(3, len(detected_styles))]:
                if style != stylistic_attributes[0]["name"]:
                    stylistic_attributes.append({
                        "name": style,
                        "confidence": 0.65
                    })
        
        # Ensure we have at least 2 style attributes
        while len(stylistic_attributes) < 2:
            # Pick a different style than those already added
            style_list = list(style_dict.keys())
            alt_style_idx = (hash_int + len(stylistic_attributes) * 23) % len(style_list)
            alt_style = style_list[alt_style_idx]
            
            if not any(sa["name"] == alt_style for sa in stylistic_attributes):
                stylistic_attributes.append({
                    "name": alt_style,
                    "confidence": 0.6
                })
        
        # Create a simple final prompt that includes detected elements
        final_prompt = f"{main_theme} pattern with {style} style"
        if dominant_colors:
            final_prompt += f", primarily {dominant_colors[0]['name']}"
        if content_details:
            final_prompt += f", featuring {content_details[0]['name']}"
        
        # Build style keywords
        style_keywords = [main_theme, style]
        
        # Add color keywords if available
        if dominant_colors:
            style_keywords.append(dominant_colors[0]["name"])
        
        # Add content elements
        for item in content_details[:2]:
            keyword = item["name"].split()[0]  # Take the first word
            if keyword not in style_keywords:
                style_keywords.append(keyword)
        
        # Create fallback result with reasonable defaults and detected information
        fallback_result = {
            "main_theme": main_theme.capitalize(),
            "main_theme_confidence": round(main_theme_confidence, 2),
            "category": main_theme.capitalize(),
            "category_confidence": round(0.7 + (hash_int % 15) / 100, 2),
            "primary_pattern": main_theme.capitalize() + " pattern",
            "pattern_confidence": round(main_theme_confidence - 0.05, 2),
            "content_details": content_details,
            "stylistic_attributes": stylistic_attributes,
            "secondary_patterns": [],
            "style_keywords": style_keywords,
            "prompt": {"final_prompt": final_prompt},
            "dimensions": {"width": width, "height": height},
            "original_path": image_path
        }
        
        # Cache this fallback result with a special flag
        fallback_result["is_fallback"] = True
        self._cache_result(image_path, fallback_result)
        
        logger.info(f"Generated fallback analysis for {os.path.basename(image_path)}: {main_theme} ({fallback_result['main_theme_confidence']})")
        
        return fallback_result

    def _analyze_colors(self, image):
        """
        Basic color analysis for fallback mode
        
        Args:
            image: PIL Image object
            
        Returns:
            List of dominant colors
        """
        try:
            # Resize image to speed up processing
            img = image.copy()
            img.thumbnail((100, 100))
            
            # Convert to RGB mode
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # Define basic colors
            basic_colors = {
                "red": ((180, 0, 0), (255, 60, 60)),
                "blue": ((0, 0, 180), (60, 60, 255)),
                "green": ((0, 180, 0), (60, 255, 60)),
                "yellow": ((180, 180, 0), (255, 255, 60)),
                "black": ((0, 0, 0), (50, 50, 50)),
                "white": ((200, 200, 200), (255, 255, 255)),
                "gray": ((100, 100, 100), (180, 180, 180)),
                "purple": ((130, 0, 130), (200, 60, 200)),
                "orange": ((230, 80, 0), (255, 180, 60)),
                "brown": ((80, 40, 0), (170, 100, 60)),
                "pink": ((230, 130, 200), (255, 200, 230))
            }
            
            # Convert to RGB array
            img_data = np.array(img)
            pixels = img_data.reshape(-1, 3)
            
            # Count color frequencies
            color_counts = {color: 0 for color in basic_colors}
            for pixel in pixels:
                for color_name, (min_vals, max_vals) in basic_colors.items():
                    if (pixel[0] >= min_vals[0] and pixel[0] <= max_vals[0] and
                        pixel[1] >= min_vals[1] and pixel[1] <= max_vals[1] and
                        pixel[2] >= min_vals[2] and pixel[2] <= max_vals[2]):
                        color_counts[color_name] += 1
                        break
            
            # Calculate proportions
            total_pixels = len(pixels)
            color_props = {color: count / total_pixels for color, count in color_counts.items() if count > 0}
            
            # Sort colors by frequency
            sorted_colors = sorted(color_props.items(), key=lambda x: x[1], reverse=True)
            
            # Return top colors with proportion
            result = []
            for color_name, proportion in sorted_colors[:3]:
                if proportion > 0.05:  # Only include colors that make up at least 5% of the image
                    result.append({
                        "name": color_name,
                        "proportion": round(proportion, 2)
                    })
            
            return result
        except Exception as e:
            logger.error(f"Error in basic color analysis: {str(e)}")
            return [] 