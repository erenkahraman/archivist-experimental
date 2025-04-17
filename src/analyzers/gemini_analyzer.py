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
import re  # New: for dynamic tokenization
import sklearn
from sklearn.cluster import KMeans
from scipy import ndimage

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
        if not self._api_key:
            logger.error("No API key provided for Gemini analysis")
            return self._generate_fallback_analysis(image_path)
            
        # Check if we've cached this result
        cache_key = f"gemini:{image_path}"
        cached_result = self._get_from_cache(image_path)
        if cached_result:
            logger.info(f"Using cached result for {image_path}")
            return cached_result
            
        try:
            # Check if we're currently rate limited
            current_time = time.time()
            if self._rate_limited and current_time < self._rate_limit_reset_time:
                logger.warning(f"Skipping Gemini API call due to active rate limit. Using fallback analysis. Retry after {int(self._rate_limit_reset_time - current_time)}s")
                return self._generate_fallback_analysis(image_path)
            
            # Load the image and resize for token efficiency
            image = Image.open(image_path)
            
            # Get image dimensions
            width, height = image.size
            
            # Resize to smaller dimensions for token efficiency
            max_dim = 384
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
            image.convert('RGB').save(buffer, format="JPEG", quality=85)
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
                model_name=GEMINI_CONFIG.get('model', "gemini-1.5-flash"),
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
                    
                    # Enhance with pattern database if available
                    result = self._enhance_with_pattern_database(result, image_path)
                    
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
                # Set rate limiting flag and calculate when to reset
                self._rate_limited = True
                self._rate_limit_wait_seconds = 300  # Default 5 minutes
                self._rate_limit_reset_time = time.time() + 300
                logger.warning(f"Setting rate limit cooldown for {300} seconds until {time.ctime(self._rate_limit_reset_time)}")
                
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

    def _get_cache_path(self, image_path: str) -> Path:
        """Generate a unique cache file path for an image"""
        # Create a hash of the image path to use as the cache file name
        hash_obj = hashlib.md5(image_path.encode())
        file_hash = hash_obj.hexdigest()
        cache_path = self.cache_dir / f"{file_hash}.json"
        return cache_path
    
    def _generate_fallback_analysis(self, image_path: str) -> Dict[str, Any]:
        """
        Generate a 100% dynamic fallback analysis using Gemini's potential fully.
        Instead of using static dictionaries, tokenize the filename and leverage basic color analysis.
        
        This dynamic fallback extracts descriptive tokens from the filename, filters out common stopwords,
        and builds a dynamic analysis that includes dominant colors from the image.
        
        Args:
            image_path: Path to the image file
        
        Returns:
            Fallback analysis result as a dictionary.
        """
        logger.info(f"Generating fallback analysis for {os.path.basename(image_path)}")
        try:
            image = Image.open(image_path)
            width, height = image.size
            
            # Enhanced color analysis
            try:
                from src.analyzers.color_analyzer import ColorAnalyzer  # Import locally if needed
                # Pass the API key if the analyzer needs it
                color_analyzer_instance = ColorAnalyzer(api_key=getattr(self, '_api_key', None))
                # Get image as numpy array for color analysis
                image_np = np.array(image)  # Assumes 'image' is the PIL Image object
                color_info = color_analyzer_instance.analyze_colors(image_np)
                dominant_colors = color_info.get('dominant_colors', [])
            except Exception as e:
                logger.error(f"Fallback: Error during color analysis: {str(e)}")
                # Fall back to basic color analysis
                dominant_colors = self._analyze_colors(image)
            
            file_name = os.path.basename(image_path)
            base_name = os.path.splitext(file_name)[0].lower()
            
            # Check if filename appears to be a UUID or random hash
            uuid_pattern = re.compile(r'^[a-f0-9]{8}-?[a-f0-9]{4}-?[a-f0-9]{4}-?[a-f0-9]{4}-?[a-f0-9]{12}$', re.IGNORECASE)
            hash_pattern = re.compile(r'^[a-f0-9]{10,}$', re.IGNORECASE)
            
            if uuid_pattern.match(base_name) or hash_pattern.match(base_name):
                logger.info(f"Filename appears to be a UUID or hash: {base_name}")
                main_theme = "textile"  # Default
                main_theme_confidence = 0.5
                tokens = ["textile", "fabric", "pattern"]  # Default tokens
                
                if dominant_colors:
                    # Use the name of the most dominant color
                    primary_color_name = dominant_colors[0].get('name', '').split()[0]  # Just first word
                    if primary_color_name:
                        main_theme = f"{primary_color_name.capitalize()} Pattern"  # More descriptive theme
                        main_theme_confidence = 0.60  # Slightly higher confidence
                        tokens.insert(0, primary_color_name.lower())  # Add color token
            else:
                # Dynamically tokenize the filename on non-alphanumeric characters
                tokens = re.split(r'\W+', base_name)
                stopwords = set(["a", "an", "the", "of", "and", "in", "on", "at", "to", "for", "with"])
                
                # Filter out tokens that look like UUIDs or short meaningless strings
                filtered_tokens = []
                for token in tokens:
                    # Skip empty tokens
                    if not token:
                        continue
                        
                    # Skip tokens that are just numbers
                    if token.isdigit():
                        continue
                        
                    # Skip tokens that look like part of UUIDs or random strings
                    if len(token) >= 6 and all(c.isdigit() or c.lower() in 'abcdef' for c in token):
                        continue
                        
                    # Skip very short tokens unless they're the only token
                    if len(token) <= 2 and len(tokens) > 1:
                        continue
                        
                    # Skip stopwords
                    if token.lower() in stopwords:
                        continue
                        
                    filtered_tokens.append(token)
                
                tokens = filtered_tokens
                
                # Use the longest token as the primary descriptor if available
                if tokens:
                    # Sort by length and prefer alphabetic tokens
                    main_theme = max(tokens, key=lambda t: (any(c.isalpha() for c in t), len(t)))
                    main_theme_confidence = 0.6
                else:
                    # If no meaningful tokens found
                    main_theme = "textile"
                    main_theme_confidence = 0.5
            
            # Build content details dynamically using the tokens and dominant colors
            content_details = []
            for token in tokens[:2]:
                if token and token != main_theme:  # Avoid duplicating main theme
                    content_details.append({
                        "name": token,
                        "confidence": 0.6
                    })
            
            # Add color information to content details
            if dominant_colors:
                color_name = dominant_colors[0]["name"]
                content_details.append({
                    "name": f"{color_name} background",
                    "confidence": 0.7
                })
                
                # Add additional color details if available
                if len(dominant_colors) > 1:
                    secondary_color = dominant_colors[1]["name"]
                    content_details.append({
                        "name": f"{secondary_color} accent",
                        "confidence": 0.65
                    })
            
            # Ensure we have at least one content detail
            if not content_details:
                content_details.append({
                    "name": "abstract element",
                    "confidence": 0.5
                })
            
            # Build stylistic attributes dynamically from the tokens
            stylistic_attributes = []
            for token in set(tokens):
                if token != main_theme:  # Avoid duplicating main theme
                    stylistic_attributes.append({
                        "name": token,
                        "confidence": 0.6
                    })
            
            # Add default style attributes if none found
            if not stylistic_attributes:
                if dominant_colors:
                    color_name = dominant_colors[0]["name"]
                    stylistic_attributes.append({"name": f"{color_name}-toned", "confidence": 0.6})
                stylistic_attributes.append({"name": "contemporary", "confidence": 0.55})
            
            # Construct a dynamic final prompt with more structured information
            final_prompt = f"Fallback analysis for {file_name}."  # Start simpler
            if main_theme != 'textile':
                final_prompt += f" Identified main theme as '{main_theme}'"
            if dominant_colors:
                color_desc = ", ".join([f"'{c['name']}'" for c in dominant_colors[:2]])
                final_prompt += f" Dominant colors appear to be {color_desc}."
            
            # Create distinct style keywords
            style_keywords = [main_theme]
            for token in tokens:
                if token != main_theme:
                    style_keywords.append(token)
                    
            # Add color keywords from dominant colors
            if dominant_colors:
                for color in dominant_colors[:2]:  # Top 2 colors
                    color_name = color.get('name')
                    if color_name and color_name not in style_keywords:
                        style_keywords.append(color_name)
                        
            # Add sensible default keywords if we have few
            if len(style_keywords) < 3:
                for default_kw in ["textile", "fabric", "pattern"]:
                    if default_kw not in style_keywords:
                        style_keywords.append(default_kw)
            
            # Create secondary patterns based on available tokens
            secondary_patterns = []
            for token in tokens:
                if token != main_theme and len(token) > 3:  # Only meaningful tokens
                    secondary_patterns.append({
                        "name": f"{token} pattern",
                        "confidence": 0.55
                    })
                    if len(secondary_patterns) >= 2:  # Limit to 2 secondary patterns
                        break
            
            # Add color-based secondary pattern if no token-based ones
            if not secondary_patterns and dominant_colors:
                secondary_patterns.append({
                    "name": f"{dominant_colors[0]['name']} accent pattern",
                    "confidence": 0.5
                })
            
            # Generate a better primary pattern for UUID filenames
            primary_pattern = f"{main_theme.capitalize()} pattern"
            
            # For UUID files, try to make a more descriptive primary pattern based on colors and content
            if uuid_pattern.match(base_name) or hash_pattern.match(base_name):
                if dominant_colors and len(dominant_colors) > 0:
                    color_name = dominant_colors[0]["name"].split()[0]
                    
                    # Choose a pattern type that's commonly associated with certain colors
                    if color_name.lower() in ["black", "charcoal", "dark"]:
                        pattern_types = ["abstract", "geometric", "modern"]
                    elif color_name.lower() in ["blue", "navy", "azure"]:
                        pattern_types = ["stripe", "geometric", "floral"]
                    elif color_name.lower() in ["red", "crimson", "burgundy"]:
                        pattern_types = ["floral", "damask", "oriental"]
                    elif color_name.lower() in ["green", "emerald", "olive"]:
                        pattern_types = ["tropical", "floral", "leaf"]
                    elif color_name.lower() in ["brown", "tan", "beige"]:
                        pattern_types = ["natural", "textile", "organic"]
                    else:
                        pattern_types = ["textile", "abstract", "geometric"]
                    
                    # Select a pattern type based on image path hash (deterministic but seems random)
                    hash_obj = hashlib.md5(image_path.encode())
                    hash_val = int(hash_obj.hexdigest(), 16)
                    pattern_type = pattern_types[hash_val % len(pattern_types)]
                    
                    primary_pattern = f"{color_name} {pattern_type} pattern"
            
            fallback_result = {
                "main_theme": main_theme.capitalize(),
                "main_theme_confidence": round(main_theme_confidence, 2),
                "category": main_theme.capitalize(),
                "category_confidence": 0.7,
                "primary_pattern": primary_pattern,
                "pattern_confidence": round(main_theme_confidence - 0.05, 2),
                "content_details": content_details,
                "stylistic_attributes": stylistic_attributes,
                "secondary_patterns": secondary_patterns,
                "style_keywords": style_keywords,
                "prompt": {"final_prompt": final_prompt},
                "dimensions": {"width": width, "height": height},
                "original_path": image_path
            }
            fallback_result["is_fallback"] = True
            self._cache_result(image_path, fallback_result)
            
            logger.info(f"Generated fallback analysis for {os.path.basename(image_path)}: {main_theme.capitalize()} ({fallback_result['main_theme_confidence']})")
            return fallback_result
        except Exception as e:
            logger.error(f"Error in fallback analysis: {str(e)}")
            return self._get_default_response(image_path)

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
        
        # Generate stylistic attributes from content details if missing
        if ("stylistic_attributes" not in response or 
            not isinstance(response["stylistic_attributes"], list) or
            (len(response["stylistic_attributes"]) == 1 and 
             response["stylistic_attributes"][0].get("name") == "unknown style")):
            
            response["stylistic_attributes"] = []
            
            # Try to generate from content details
            if "content_details" in response and isinstance(response["content_details"], list):
                for detail in response["content_details"]:
                    if isinstance(detail, dict) and "name" in detail:
                        # Extract adjectives or descriptive elements from the detail name
                        words = detail["name"].split()
                        if len(words) > 1:  # If multiple words, use first as potential adjective
                            response["stylistic_attributes"].append({
                                "name": f"{words[0].lower()} style",
                                "confidence": detail.get("confidence", 0.7)
                            })
            
            # If main_theme is known but stylistic_attributes is still empty, derive from it
            if (response["main_theme"] != "Unknown" and 
                not response["stylistic_attributes"] and
                response["main_theme"].lower() != "unknown"):
                
                theme_parts = response["main_theme"].split()
                if len(theme_parts) > 0:
                    response["stylistic_attributes"].append({
                        "name": f"{theme_parts[0].lower()} style",
                        "confidence": response.get("main_theme_confidence", 0.7)
                    })
                    
                    if len(theme_parts) > 1:
                        response["stylistic_attributes"].append({
                            "name": f"{theme_parts[-1].lower()} elements",
                            "confidence": response.get("main_theme_confidence", 0.7) - 0.1
                        })
                        
            # If still empty, add a default based on main_theme
            if not response["stylistic_attributes"] and response["main_theme"].lower() != "unknown":
                response["stylistic_attributes"].append({
                    "name": f"{response['main_theme'].lower()} style",
                    "confidence": response.get("main_theme_confidence", 0.7) - 0.1
                })
        
        # Update style_keywords if they're unknown or empty
        if ("style_keywords" not in response or 
            not isinstance(response["style_keywords"], list) or
            len(response["style_keywords"]) == 0 or
            (len(response["style_keywords"]) == 1 and response["style_keywords"][0].lower() == "unknown")):
            
            # Start with main theme
            keywords = []
            if response["main_theme"] and response["main_theme"].lower() != "unknown":
                keywords.append(response["main_theme"].lower())
                
                # Add parts of the theme if it's a compound term
                theme_parts = response["main_theme"].split()
                for part in theme_parts:
                    if part.lower() not in keywords and len(part) > 3:
                        keywords.append(part.lower())
            
            # Add from content details
            if "content_details" in response and isinstance(response["content_details"], list):
                for detail in response["content_details"]:
                    if isinstance(detail, dict) and "name" in detail and detail["name"]:
                        # Add significant words from details
                        for word in detail["name"].split():
                            if len(word) > 3 and word.lower() not in keywords:
                                keywords.append(word.lower())
            
            # If we have at least one keyword from above, use them
            if keywords:
                response["style_keywords"] = keywords
            # Otherwise fall back to reasonable defaults based on main theme
            elif response["main_theme"].lower() != "unknown":
                default_keywords = [
                    response["main_theme"].lower(),
                    "pattern",
                    "design",
                    "decorative"
                ]
                response["style_keywords"] = default_keywords
        
        # Ensure content_details is a list
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
        
        # Add fields expected by the gallery component
        if "main_theme" in response and response["main_theme"]:
            # Check if "paisley" is in the main_theme
            if "paisley" in response["main_theme"].lower():
                response["primary_pattern"] = "paisley pattern"
            else:
                response["primary_pattern"] = response["main_theme"]
        else:
            response["primary_pattern"] = response.get("category", "Unknown")
        
        # NEW: If primary_pattern is still Unknown but we have useful style keywords, use them
        if (response.get("primary_pattern", "Unknown").lower() == "unknown" and 
              "style_keywords" in response and isinstance(response["style_keywords"], list) and 
              len(response["style_keywords"]) > 0):
            
            # Look for pattern-indicating keywords
            pattern_keywords = ["border", "floral", "geometric", "paisley", "stripe", "check", 
                             "plaid", "polka dot", "chevron", "herringbone", "damask", "ikat", 
                             "batik", "tribal", "abstract", "argyle", "houndstooth", "toile"]
            
            for pattern_keyword in pattern_keywords:
                for keyword in response["style_keywords"]:
                    if isinstance(keyword, str) and pattern_keyword.lower() in keyword.lower():
                        response["primary_pattern"] = f"{pattern_keyword.capitalize()} pattern"
                        response["pattern_confidence"] = 0.75
                        logger.info(f"Set primary pattern to '{response['primary_pattern']}' based on style keywords")
                        break
                if response.get("primary_pattern", "Unknown").lower() != "unknown":
                    break
        
        # Ensure pattern_confidence is a valid number
        if "main_theme_confidence" in response and response["main_theme_confidence"] is not None:
            response["pattern_confidence"] = float(response["main_theme_confidence"])
        elif "category_confidence" in response and response["category_confidence"] is not None:
            response["pattern_confidence"] = float(response["category_confidence"])
        else:
            response["pattern_confidence"] = 0.8
        
        # Ensure secondary_patterns is a list
        if not isinstance(response.get("secondary_patterns"), list):
            response["secondary_patterns"] = []
        
        # Add a secondary pattern if we have none but have content details
        if (len(response["secondary_patterns"]) == 0 and 
            "content_details" in response and 
            isinstance(response["content_details"], list) and 
            len(response["content_details"]) > 0):
            
            for detail in response["content_details"]:
                if (isinstance(detail, dict) and 
                    "name" in detail and 
                    detail["name"].lower() != "unknown element"):
                    
                    response["secondary_patterns"].append({
                        "name": detail["name"],
                        "confidence": detail.get("confidence", 0.6)
                    })
                    break  # Just add one secondary pattern
                    
        # Ensure dimensions if not present
        if "dimensions" not in response:
            response["dimensions"] = {"width": 0, "height": 0}
        
        # Generate prompt if it's unknown
        if ("prompt" not in response or 
            not isinstance(response["prompt"], dict) or
            not response["prompt"].get("final_prompt") or
            response["prompt"].get("final_prompt") == "Unknown pattern"):
            
            # Build a basic prompt from available information
            prompt_parts = []
            
            if response["main_theme"] and response["main_theme"].lower() != "unknown":
                prompt_parts.append(f"This is a {response['main_theme'].lower()} pattern")
                
                if "content_details" in response and isinstance(response["content_details"], list):
                    details = []
                    for detail in response["content_details"]:
                        if isinstance(detail, dict) and "name" in detail and detail["name"].lower() != "unknown element":
                            details.append(detail["name"])
                    
                    if details:
                        prompt_parts.append(f"featuring {', '.join(details)}")
                
                if "style_keywords" in response and isinstance(response["style_keywords"], list):
                    keywords = [k for k in response["style_keywords"] if k.lower() != "unknown"]
                    if keywords:
                        prompt_parts.append(f"with {', '.join(keywords[:3])} characteristics")
                
                if prompt_parts:
                    final_prompt = ". ".join(prompt_parts) + "."
                    response["prompt"] = {"final_prompt": final_prompt}
        
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

    def _analyze_colors(self, image):
        """
        Analyze the colors in the image to extract dominant colors and their proportions
        
        Args:
            image: PIL Image object
            
        Returns:
            Dict containing color analysis results
        """
        try:
            # Convert image to numpy array
            img_array = np.array(image)
            
            # Reshape the array to be a list of pixels
            pixels = img_array.reshape(-1, 3)
            
            # Sample pixels if there are too many (for performance)
            if len(pixels) > 10000:
                indices = np.random.choice(len(pixels), 10000, replace=False)
                pixels = pixels[indices]
            
            # Use KMeans to find dominant colors
            num_colors = min(5, len(set(map(tuple, pixels))))  # Ensure we don't exceed unique colors
            if num_colors < 2:  # If image has very few colors
                num_colors = 2  # Minimum of 2 clusters
                
            kmeans = KMeans(n_clusters=num_colors, n_init=10)
            kmeans.fit(pixels)
            
            # Get the dominant colors
            colors = kmeans.cluster_centers_.astype(int)
            
            # Count pixels in each cluster
            labels = kmeans.labels_
            counts = np.bincount(labels)
            
            # Calculate percentages
            percentages = counts / counts.sum()
            
            # Create color names and hex values
            color_data = []
            for i, (color, percent) in enumerate(zip(colors, percentages)):
                r, g, b = color
                hex_color = f"#{r:02x}{g:02x}{b:02x}"
                color_name = self._get_color_name(r, g, b)
                
                color_data.append({
                    "color": color_name,
                    "hex": hex_color,
                    "percentage": round(float(percent) * 100, 1)
                })
            
            # Sort by percentage (highest first)
            color_data = sorted(color_data, key=lambda x: x["percentage"], reverse=True)
            
            # Calculate overall brightness and saturation
            brightness = np.mean(pixels) / 255.0  # 0 to 1 range
            
            # RGB to HSV for saturation
            hsv_pixels = np.zeros((len(pixels), 3))
            for i, (r, g, b) in enumerate(pixels):
                h, s, v = self._rgb_to_hsv(r, g, b)
                hsv_pixels[i] = [h, s, v]
                
            saturation = np.mean(hsv_pixels[:, 1])  # 0 to 1 range
            
            # Calculate contrast
            if len(pixels) > 1:
                std_dev = np.std(pixels)
                max_std = 255 * np.sqrt(3)  # Maximum possible std dev for RGB values
                contrast = min(1.0, std_dev / max_std)
            else:
                contrast = 0.0
            
            # Determine color temperature
            r_mean = np.mean(pixels[:, 0])
            b_mean = np.mean(pixels[:, 2])
            
            if r_mean > b_mean:
                temperature = "warm"
            else:
                temperature = "cool"
                
            # Get color harmony type
            harmony = self._determine_color_harmony(color_data)
            
            return {
                "dominant_colors": color_data,
                "brightness": round(brightness, 2),
                "saturation": round(saturation, 2),
                "contrast": round(contrast, 2),
                "temperature": temperature,
                "harmony": harmony
            }
            
        except Exception as e:
            logger.error(f"Error analyzing colors: {str(e)}")
            return {
                "dominant_colors": [{"color": "unknown", "hex": "#808080", "percentage": 100.0}],
                "brightness": 0.5,
                "saturation": 0.5,
                "contrast": 0.5,
                "temperature": "neutral",
                "harmony": "unknown"
            }
    
    def _get_color_name(self, r, g, b):
        """Convert RGB values to a color name"""
        # Simple color naming based on RGB values
        if max(r, g, b) < 30:
            return "black"
        if min(r, g, b) > 225:
            return "white"
            
        # Gray detection
        if abs(r - g) < 20 and abs(g - b) < 20 and abs(r - b) < 20:
            brightness = (r + g + b) / 3
            if brightness < 80:
                return "dark gray"
            if brightness < 150:
                return "gray"
            return "light gray"
            
        # Primary and secondary colors
        if r > max(g, b) + 50:
            if g > 150 and b > 150:
                return "pink"
            if g > 150:
                return "orange"
            if b > 150:
                return "magenta"
            return "red"
            
        if g > max(r, b) + 50:
            if r > 150:
                return "yellow-green"
            if b > 150:
                return "teal"
            return "green"
            
        if b > max(r, g) + 50:
            if r > 150:
                return "purple"
            if g > 150:
                return "cyan"
            return "blue"
            
        # Mixed colors
        if r > 150 and g > 150 and b < 100:
            return "yellow"
        if r > 150 and b > 150 and g < 100:
            return "purple"
        if g > 150 and b > 150 and r < 100:
            return "turquoise"
            
        # Fallback
        return "mixed"
    
    def _rgb_to_hsv(self, r, g, b):
        """Convert RGB to HSV color space"""
        r, g, b = r/255.0, g/255.0, b/255.0
        mx = max(r, g, b)
        mn = min(r, g, b)
        df = mx - mn
        
        if mx == mn:
            h = 0
        elif mx == r:
            h = (60 * ((g-b)/df) + 360) % 360
        elif mx == g:
            h = (60 * ((b-r)/df) + 120) % 360
        elif mx == b:
            h = (60 * ((r-g)/df) + 240) % 360
            
        s = 0 if mx == 0 else df/mx
        v = mx
        
        return h/360, s, v
    
    def _determine_color_harmony(self, color_data):
        """Determine the color harmony type based on dominant colors"""
        if not color_data or len(color_data) <= 1:
            return "monochromatic"
            
        # Extract only colors that make up at least 10% of the image
        significant_colors = [c for c in color_data if c["percentage"] >= 10.0]
        
        if len(significant_colors) == 1:
            return "monochromatic"
            
        # Check if all colors are gray/black/white
        neutral_colors = ["black", "white", "gray", "dark gray", "light gray"]
        if all(c["color"] in neutral_colors for c in significant_colors):
            return "monochromatic"
            
        # Check for complementary (opposites: red-green, blue-orange, yellow-purple)
        complementary_pairs = [
            ({"red", "pink", "orange"}, {"green", "teal", "turquoise"}),
            ({"blue", "cyan"}, {"orange", "yellow-green"}),
            ({"yellow"}, {"purple", "magenta"})
        ]
        
        color_set = {c["color"] for c in significant_colors}
        
        for pair in complementary_pairs:
            if color_set.intersection(pair[0]) and color_set.intersection(pair[1]):
                return "complementary"
                
        # Check for analogous (colors that are next to each other on the color wheel)
        analogous_groups = [
            {"red", "orange", "yellow"},
            {"yellow", "yellow-green", "green"},
            {"green", "teal", "cyan"},
            {"cyan", "blue", "turquoise"},
            {"blue", "purple", "magenta"},
            {"magenta", "red", "pink"}
        ]
        
        for group in analogous_groups:
            if len(color_set.intersection(group)) >= 2:
                return "analogous"
                
        # Check for triadic (three colors evenly spaced on the color wheel)
        triadic_groups = [
            {"red", "yellow", "blue"},
            {"orange", "green", "purple"},
            {"yellow", "blue", "magenta"},
            {"yellow-green", "purple", "pink"}
        ]
        
        for group in triadic_groups:
            if len(color_set.intersection(group)) >= 3:
                return "triadic"
                
        # Default to "mixed" if no specific harmony is detected
        return "mixed"

    def _enhance_with_pattern_database(self, analysis_result, img_path=None):
        """
        Enhance the analysis results using pattern database.
        """
        # Load pattern database for enhancement
        try:
            pattern_db = self._load_pattern_database()
            if not pattern_db:
                return analysis_result

            # Check if the main theme matches any pattern in the database
            main_theme = analysis_result.get('main_theme', '').lower()
            
            # Extract best matching pattern
            best_match = None
            best_match_score = 0
            
            for pattern, details in pattern_db.get('pattern_categories', {}).items():
                # Direct match
                if pattern.lower() == main_theme:
                    best_match = pattern
                    best_match_score = 1.0
                    break
                
                # Check if main_theme contains pattern name
                if pattern.lower() in main_theme:
                    match_score = 0.8
                    if match_score > best_match_score:
                        best_match = pattern
                        best_match_score = match_score
                        
                # Check if pattern name appears in keywords
                if 'keywords' in details:
                    for keyword in details['keywords']:
                        if keyword.lower() == main_theme or keyword.lower() in main_theme:
                            match_score = 0.7
                            if match_score > best_match_score:
                                best_match = pattern
                                best_match_score = match_score
            
            # If no match found but we have a main_theme, use it for basic enhancement
            if not best_match and main_theme and main_theme != "unknown":
                # Basic enhancement with available data
                if analysis_result.get('confidence', 0) < 0.7:
                    analysis_result['confidence'] = 0.7
                
                if not analysis_result.get('style_keywords') or "unknown" in analysis_result.get('style_keywords', []):
                    # Generate basic style keywords from main_theme
                    words = main_theme.split()
                    keywords = [main_theme]
                    if len(words) > 1:
                        keywords.extend(words)
                    
                    # Add category keywords if available
                    category = analysis_result.get('category', '')
                    if category and category != "unknown":
                        keywords.append(category)
                        
                    # Add material keywords if filename suggests them
                    if img_path:
                        filename = os.path.basename(img_path).lower()
                        materials = ["silk", "cotton", "linen", "wool", "polyester", "leather", "denim"]
                        for material in materials:
                            if material in filename:
                                keywords.append(material)
                                
                    analysis_result['style_keywords'] = list(set(keywords))
                
                # Ensure we have content details
                if not analysis_result.get('content_details') or len(analysis_result.get('content_details', [])) == 0:
                    analysis_result['content_details'] = [f"{main_theme} pattern", "repeating elements"]
                
                return analysis_result
            
            if best_match and best_match_score >= 0.7:
                pattern_details = pattern_db['pattern_categories'][best_match]
                
                # Update confidence based on match quality
                if analysis_result.get('confidence', 0) < best_match_score:
                    analysis_result['confidence'] = best_match_score
                
                # Add historical context if available
                if 'historical_context' in pattern_details and pattern_details['historical_context']:
                    analysis_result['historical_context'] = pattern_details['historical_context']
                
                # Add or enhance style keywords
                if 'keywords' in pattern_details and pattern_details['keywords']:
                    existing_keywords = analysis_result.get('style_keywords', [])
                    if not existing_keywords or "unknown" in existing_keywords:
                        analysis_result['style_keywords'] = pattern_details['keywords']
                    else:
                        # Merge keywords, remove duplicates
                        combined = existing_keywords + pattern_details['keywords']
                        analysis_result['style_keywords'] = list(set(combined))
                
                # Enhance content details
                if 'content_elements' in pattern_details and pattern_details['content_elements']:
                    existing_content = analysis_result.get('content_details', [])
                    if not existing_content or (len(existing_content) == 1 and existing_content[0] == "unknown"):
                        analysis_result['content_details'] = pattern_details['content_elements']
                    else:
                        # Merge content elements, remove duplicates
                        combined = existing_content + pattern_details['content_elements']
                        analysis_result['content_details'] = list(set(combined))
                
                # Add origin information if available
                if 'origin' in pattern_details and pattern_details['origin']:
                    analysis_result['origin'] = pattern_details['origin']
                
                # Enhance category if it's unknown
                if analysis_result.get('category') == "unknown" and 'applications' in pattern_details:
                    analysis_result['category'] = pattern_details['applications'][0] if pattern_details['applications'] else "decorative"
                
                # Add description to prompt if available
                if 'description' in pattern_details and pattern_details['description']:
                    prompt = analysis_result.get('prompt', '')
                    if prompt and "unknown" not in prompt:
                        analysis_result['prompt'] = f"{prompt} {pattern_details['description']}"
                    else:
                        analysis_result['prompt'] = f"This image shows a {best_match} pattern. {pattern_details['description']}"
            
            # Final check to ensure no unknown values remain
            self._clean_unknown_values(analysis_result)
            
            return analysis_result
                
        except Exception as e:
            logging.error(f"Error enhancing with pattern database: {e}")
            return analysis_result
            
    def _clean_unknown_values(self, analysis_result):
        """
        Remove or replace any 'unknown' values in the analysis result.
        """
        main_theme = analysis_result.get('main_theme', '')
        
        # Clean up style keywords
        if 'style_keywords' in analysis_result:
            if "unknown" in analysis_result['style_keywords']:
                # Replace unknown with defaults based on main theme
                analysis_result['style_keywords'] = [keyword for keyword in analysis_result['style_keywords'] if keyword != "unknown"]
                if main_theme and main_theme != "unknown":
                    analysis_result['style_keywords'].append(main_theme)
                    words = main_theme.split()
                    if len(words) > 1:
                        analysis_result['style_keywords'].extend(words)
                    analysis_result['style_keywords'] = list(set(analysis_result['style_keywords']))
                if not analysis_result['style_keywords']:
                    analysis_result['style_keywords'] = ["decorative", "patterned"]
        
        # Ensure content details exists and has no unknowns
        if 'content_details' not in analysis_result or not analysis_result['content_details'] or "unknown" in analysis_result['content_details']:
            content = [item for item in analysis_result.get('content_details', []) if item != "unknown"]
            if main_theme and main_theme != "unknown":
                content.append(f"{main_theme} pattern")
            if not content:
                content = ["decorative pattern", "repeating elements"]
            analysis_result['content_details'] = content
        
        # Ensure we have a prompt
        if 'prompt' not in analysis_result or not analysis_result['prompt'] or analysis_result['prompt'] == "unknown":
            pattern_name = main_theme if main_theme and main_theme != "unknown" else "decorative"
            color_info = ""
            if 'color_palette' in analysis_result and analysis_result['color_palette']:
                color_names = [color.get('color_name', '') for color in analysis_result['color_palette'] if color.get('color_name')]
                if color_names:
                    color_info = f" with {', '.join(color_names[:3])} colors"
            analysis_result['prompt'] = f"This image shows a {pattern_name} pattern{color_info}."
        
        return analysis_result
    
    def _load_pattern_database(self):
        """
        Loads the pattern database from the config file.
        
        Returns:
            Dictionary containing pattern database information or None if loading fails
        """
        try:
            db_path = os.path.join(self.cache_dir, "pattern_database.json")
            if not os.path.exists(db_path):
                logger.warning(f"Pattern database not found at {db_path}")
                return None
            
            with open(db_path, "r") as f:
                pattern_db = json.load(f)
            
            logger.info(f"Successfully loaded pattern database with {len(pattern_db.get('pattern_categories', {}))} categories")
            return pattern_db
            
        except Exception as e:
            logger.error(f"Failed to load pattern database: {str(e)}")
            return None

    def _classify_based_on_database(self, pattern_info, confidence_threshold=0.65):
        """
        Attempts to refine the classification of patterns based on the database
        when confidence is below the specified threshold.
        
        Args:
            pattern_info: Dictionary containing pattern analysis information
            confidence_threshold: Threshold below which database matching is attempted
            
        Returns:
            Updated pattern_info with potentially refined classification
        """
        if not hasattr(self, 'pattern_database') or not self.pattern_database:
            self.pattern_database = self._load_pattern_database()
        
        if not self.pattern_database or 'pattern_categories' not in self.pattern_database:
            return pattern_info
        
        # Only attempt classification if confidence is below threshold
        if pattern_info.get('confidence', 0) >= confidence_threshold:
            return pattern_info
        
        # Extract relevant content to match against database
        content_to_match = []
        
        # Get all words from the main theme
        if 'main_theme' in pattern_info and pattern_info['main_theme']:
            content_to_match.extend(pattern_info['main_theme'].lower().split())
        
        # Get all words from content details
        if 'content_details' in pattern_info and pattern_info['content_details']:
            content_to_match.extend([word.lower() for phrase in pattern_info['content_details'] 
                                    for word in phrase.split()])
        
        # Get all words from secondary patterns
        if 'secondary_patterns' in pattern_info and pattern_info['secondary_patterns']:
            content_to_match.extend([word.lower() for phrase in pattern_info['secondary_patterns'] 
                                    for word in phrase.split()])
        
        # Get all style keywords
        if 'style_keywords' in pattern_info and pattern_info['style_keywords']:
            content_to_match.extend([keyword.lower() for keyword in pattern_info['style_keywords']])
        
        # Remove common stopwords
        stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 
                    'of', 'by', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
        content_to_match = [word for word in content_to_match if word.lower() not in stopwords]
        
        if not content_to_match:
            return pattern_info
        
        # Calculate match scores for each category
        category_scores = {}
        for category_name, category_data in self.pattern_database['pattern_categories'].items():
            score = 0
            
            # Check direct category name match (highest weight)
            if category_name.lower() in content_to_match:
                score += 10
            
            # Check keywords matches
            if 'keywords' in category_data:
                for keyword in category_data['keywords']:
                    if keyword.lower() in content_to_match:
                        score += 5
                    # Partial matching for compound words
                    for content_word in content_to_match:
                        if content_word in keyword.lower() or keyword.lower() in content_word:
                            if content_word != keyword.lower():  # Avoid double counting
                                score += 2
            
            # Check content element matches
            if 'content_elements' in category_data:
                for element in category_data['content_elements']:
                    element_words = element.lower().split()
                    # Count matching words in each content element
                    element_match_count = sum(1 for word in element_words if word in content_to_match)
                    if element_match_count > 0:
                        # Score based on percentage of words matched
                        score += (element_match_count / len(element_words)) * 3
            
            category_scores[category_name] = score
        
        # Find the best match
        if category_scores:
            best_category = max(category_scores.items(), key=lambda x: x[1])
            
            # Only update if the score is meaningful (> 5)
            if best_category[1] > 5:
                category_name = best_category[0]
                category_data = self.pattern_database['pattern_categories'][category_name]
                
                # Update pattern information
                pattern_info['main_theme'] = category_name.title() + " Pattern"
                pattern_info['confidence'] = min(0.85, pattern_info.get('confidence', 0) + 0.25)  # Boost confidence but cap at 0.85
                
                # Add historical context if available
                if 'historical_context' in category_data:
                    if 'historical_context' not in pattern_info:
                        pattern_info['historical_context'] = category_data['historical_context']
                    else:
                        pattern_info['historical_context'] += ". " + category_data['historical_context']
                
                # Add origin if available
                if 'origin' in category_data:
                    if 'historical_context' not in pattern_info:
                        pattern_info['historical_context'] = f"Origin: {category_data['origin']}"
                    else:
                        pattern_info['historical_context'] += f". Origin: {category_data['origin']}"
                
                # Update style keywords by merging and removing duplicates
                existing_keywords = set(pattern_info.get('style_keywords', []))
                new_keywords = set(category_data.get('keywords', []))
                pattern_info['style_keywords'] = list(existing_keywords.union(new_keywords))
                
                # Log the classification
                logger.info(f"Pattern classified as '{category_name}' with confidence {pattern_info['confidence']}")
        
        return pattern_info

    def _generate_top_keywords(self, pattern_info: Dict[str, Any], color_info: Dict[str, Any], count: int = 5) -> List[str]:
        """Generates a list of top keywords from analysis results."""
        keywords = set()

        # 1. Main Theme
        main_theme = pattern_info.get("main_theme")
        if main_theme and main_theme != "Unknown":
            # Add theme and potentially parts of it if multi-word
            keywords.add(main_theme.lower())
            theme_parts = [part for part in main_theme.split() if len(part) > 3]
            keywords.update(tp.lower() for tp in theme_parts)


        # 2. Content Details
        details = pattern_info.get("content_details", [])
        for detail in details:
            name = detail.get("name")
            if name:
                # Add full name and significant words within it
                keywords.add(name.lower())
                name_parts = [part.strip(',.') for part in name.split() if len(part) > 3]
                keywords.update(np.lower() for np in name_parts)


        # 3. Secondary Patterns
        secondary = pattern_info.get("secondary_patterns", [])
        for sec_pattern in secondary:
            name = sec_pattern.get("name")
            if name:
                 keywords.add(name.lower())
                 sec_parts = [part for part in name.split() if len(part) > 3]
                 keywords.update(sp.lower() for sp in sec_parts)

        # 4. Top Colors
        colors = color_info.get("dominant_colors", [])
        for color_data in colors[:2]: # Top 2 colors
            color_name = color_data.get("name")
            if color_name:
                # Add precise color name and potentially the base color word
                 keywords.add(color_name.lower())
                 base_color = color_name.split()[-1] # Get last word (often the base color)
                 if base_color:
                      keywords.add(base_color.lower())


        # Clean up common/generic words (optional, customize as needed)
        stopwords = {"pattern", "design", "element", "motif", "shape", "background", "unknown", "style", "and", "with", "the"}
        final_keywords = [kw for kw in keywords if kw not in stopwords and len(kw) > 2]

        # Return top 'count' keywords, prioritizing theme/elements? (Simple list for now)
        # Convert back to list and limit
        return final_keywords[:count]

    def _generate_final_prompt(self, pattern_info: Dict[str, Any], color_info: Dict[str, Any]) -> str:
        """
        Generate a comprehensive final prompt text based on pattern and color analysis.
        
        Args:
            pattern_info: Dictionary containing pattern analysis results
            color_info: Dictionary containing color analysis results
            
        Returns:
            A formatted prompt text combining pattern and color insights
        """
        logger.info(f"Generating final prompt. Pattern Info: {pattern_info}, Color Info: {color_info}")
        
        prompt_parts = []
        
        # Start with pattern identification
        main_theme = pattern_info.get("main_theme", "")
        confidence = pattern_info.get("main_theme_confidence", 0)
        
        if main_theme and main_theme != "Unknown":
            confidence_text = ""
            if confidence > 0.8:
                confidence_text = "clear"
            elif confidence > 0.6:
                confidence_text = "likely"
            else:
                confidence_text = "possible"
            
            prompt_parts.append(f"This is a {confidence_text} {main_theme.lower()} pattern")
        else:
            prompt_parts.append("This pattern has multiple visual elements")
        
        # Add content details if available
        content_details = pattern_info.get("content_details", [])
        if content_details and isinstance(content_details, list):
            details_text = []
            for detail in content_details:
                if isinstance(detail, dict) and "name" in detail:
                    details_text.append(detail["name"])
            
            if details_text:
                prompt_parts.append(f"featuring {', '.join(details_text)}")
        
        # Add color information
        if color_info:
            # Color palette
            dominant_colors = color_info.get("dominant_colors", [])
            if dominant_colors and len(dominant_colors) > 0:
                color_names = [color.get("color", "") for color in dominant_colors[:3]]
                color_names = [c for c in color_names if c]
                if color_names:
                    prompt_parts.append(f"with a color palette of {', '.join(color_names)}")
            
            # Color temperature
            temperature = color_info.get("temperature", "")
            if temperature:
                prompt_parts.append(f"giving a {temperature} tone")
            
            # Brightness and saturation
            brightness = color_info.get("brightness", 0)
            saturation = color_info.get("saturation", 0)
            
            brightness_text = ""
            if brightness > 0.7:
                brightness_text = "bright"
            elif brightness < 0.3:
                brightness_text = "dark"
            
            saturation_text = ""
            if saturation > 0.7:
                saturation_text = "vibrant"
            elif saturation < 0.3:
                saturation_text = "subdued"
            
            if brightness_text and saturation_text:
                prompt_parts.append(f"with {brightness_text} and {saturation_text} appearance")
            elif brightness_text:
                prompt_parts.append(f"with {brightness_text} appearance")
            elif saturation_text:
                prompt_parts.append(f"with {saturation_text} appearance")
        
        # Add secondary patterns
        secondary_patterns = pattern_info.get("secondary_patterns", [])
        if secondary_patterns and isinstance(secondary_patterns, list) and len(secondary_patterns) > 0:
            secondary_names = []
            for pattern in secondary_patterns:
                if isinstance(pattern, dict) and "name" in pattern:
                    secondary_names.append(pattern["name"])
                elif isinstance(pattern, str):
                    secondary_names.append(pattern)
            
            if secondary_names:
                prompt_parts.append(f"combined with {', '.join(secondary_names)}")
        
        # Add style keywords if available
        style_keywords = pattern_info.get("style_keywords", [])
        if style_keywords and isinstance(style_keywords, list) and len(style_keywords) > 0:
            # Filter out duplicates or keywords already mentioned
            mentioned_words = " ".join(prompt_parts).lower()
            unique_keywords = []
            
            for kw in style_keywords[:5]:  # Limit to top 5
                # Skip if already mentioned in prompt or if it's just a color name or duplicate
                if (isinstance(kw, str) and 
                    kw.lower() not in mentioned_words and
                    kw not in unique_keywords and
                    len(kw) > 3):
                    unique_keywords.append(kw)
            
            if unique_keywords:
                prompt_parts.append(f"style can be described as {', '.join(unique_keywords)}")
        
        # Join all parts with proper connecting words
        final_prompt = ". ".join(prompt_parts) + "."
        logger.info(f"Generated final prompt text: {final_prompt}")
        return final_prompt
        