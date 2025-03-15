"""Configuration settings for pattern analysis and prompt generation."""

# Similarity thresholds for attribute detection
THRESHOLDS = {
    'layout': 0.25,
    'scale': 0.25,
    'texture': 0.25,
}

# Template options for different pattern types
TEMPLATE_OPTIONS = {
    'floral': {
        'adjectives': ["Elegant", "Delicate", "Lush", "Vibrant", "Intricate"],
        'arrangements': ["scattered", "clustered", "symmetrical"],
    },
    'botanical': {
        'adjectives': ["Natural", "Organic", "Verdant"],
        'arrangements': ["layered", "overlapping", "directional"],
    },
    'geometric': {
        'adjectives': ["Bold", "Precise", "Structured", "Modern"],
        'arrangements': ["repeating", "aligned", "symmetrical"],
    },
    'animal_print': {
        'adjectives': ["Wild", "Exotic", "Natural", "Textured"],
        'arrangements': ["organic", "random", "all-over"],
    },
    # Default options for unknown pattern types
    'default': {
        'adjectives': ["Interesting", "Distinctive", "Detailed"],
        'arrangements': ["balanced", "composed", "arranged"],
    }
}

# Other global parameters
MAX_EMBEDDING_CLUSTERS = 10
IMAGE_SIZE = 512
MIN_CONFIDENCE_THRESHOLD = 0.2
HIGH_CONFIDENCE_THRESHOLD = 0.3 