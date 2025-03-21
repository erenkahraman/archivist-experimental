"""Configuration settings for Google Gemini integration."""

# Gemini API configuration
GEMINI_CONFIG = {
    'model': 'gemini-1.5-flash',  # Using the fastest, most token-efficient model
    'max_tokens': 800,            # Reduced token count for efficiency
    'temperature': 0.2            # Lower temperature for more consistent results
}

# Image size for thumbnails
IMAGE_SIZE = 512 