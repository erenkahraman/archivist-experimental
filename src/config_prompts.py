"""Configuration settings for Google Gemini integration."""

# Gemini API configuration
GEMINI_CONFIG = {
    'model': 'gemini-2.0-flash',
    'max_tokens': 1000,
    'temperature': 0.2,
    'response_mime_type': 'application/json'
}

# Image size for thumbnails
IMAGE_SIZE = 512 