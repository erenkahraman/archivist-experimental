from typing import Dict, List
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import logging

logger = logging.getLogger(__name__)

class PatternAnalyzer:
    def __init__(self, model: CLIPModel, processor: CLIPProcessor, device: torch.device):
        self.model = model
        self.processor = processor
        self.device = device
        
        # Expanded pattern categories with more specific types
        self.pattern_categories = [
            "geometric", "floral", "abstract", "striped", "polka dot", 
            "chevron", "paisley", "plaid", "animal print", "tribal",
            "damask", "herringbone", "houndstooth", "ikat", "lattice",
            "medallion", "moroccan", "ogee", "quatrefoil", "trellis",
            "leopard print", "zebra print", "tiger print", "giraffe print", "snake print",
            "checkered", "tartan", "argyle", "pinstripe", "harlequin",
            "toile", "chintz", "batik", "shibori", "tie-dye"
        ]
        
        # Specific elements with more descriptive options
        self.pattern_elements = [
            # Floral elements
            "roses", "tulips", "daisies", "sunflowers", "orchids", "peonies", "lilies",
            "cherry blossoms", "lotus flowers", "hibiscus", "lavender", "poppies",
            
            # Leaf/plant elements
            "palm leaves", "fern leaves", "maple leaves", "oak leaves", "ivy", "vines",
            "tropical leaves", "bamboo", "pine needles", "succulent plants",
            
            # Geometric elements
            "circles", "squares", "triangles", "diamonds", "hexagons", "octagons",
            "stars", "crescents", "hearts", "crosses", "spirals", "zigzags",
            
            # Animal elements
            "leopard spots", "zebra stripes", "tiger stripes", "giraffe patches",
            "snake scales", "peacock feathers", "butterfly wings", "dragonflies",
            
            # Texture elements
            "dots", "dashes", "swirls", "waves", "ripples", "crosshatch", "grid",
            "honeycomb", "basketweave", "herringbone pattern", "chevron pattern"
        ]
        
        # Specific animal print types for better identification
        self.animal_prints = {
            "leopard print": ["spotted pattern", "rosettes", "brown and black spots", "beige background"],
            "zebra print": ["black and white stripes", "parallel lines", "high contrast"],
            "tiger print": ["orange and black stripes", "vertical stripes"],
            "giraffe print": ["irregular patches", "tan and brown patches", "geometric patches"],
            "snake print": ["scales", "diamond pattern", "reptile skin"]
        }
        
        # Specific floral types
        self.floral_types = {
            "roses": ["rose petals", "rose buds", "thorny stems"],
            "tulips": ["tulip blooms", "tulip stems", "bulbous flowers"],
            "daisies": ["daisy petals", "round centers", "white petals"],
            "sunflowers": ["large centers", "yellow petals", "circular flowers"],
            "tropical": ["hibiscus", "bird of paradise", "exotic flowers"]
        }

    def _analyze_patterns(self, image_features) -> Dict:
        """
        Analyze patterns in the image features with enhanced element detection.
        
        Args:
            image_features (torch.Tensor): Image features from CLIP model
            
        Returns:
            Dict: Pattern analysis results with detailed elements
        """
        try:
            logger.info("Starting pattern analysis...")
            
            # Analyze basic pattern categories
            category_scores = {}
            with torch.no_grad():
                for category in self.pattern_categories:
                    text_inputs = self.processor(
                        text=[f"this is a {category} pattern"],
                        return_tensors="pt",
                        padding=True
                    ).to(self.device)
                    
                    text_features = self.model.get_text_features(**text_inputs)
                    similarity = torch.nn.functional.cosine_similarity(
                        image_features.to(self.device),
                        text_features,
                        dim=1
                    )
                    category_scores[category] = float(similarity[0].cpu())

            # Find primary pattern category
            primary_pattern = max(category_scores.items(), key=lambda x: x[1])
            
            # Find secondary patterns (above threshold)
            threshold = 0.2
            secondary_patterns = [
                {"name": pattern, "confidence": score}
                for pattern, score in category_scores.items()
                if score > threshold and pattern != primary_pattern[0]
            ]
            
            # Get specific elements based on the primary pattern
            detected_elements = self._analyze_specific_elements(image_features, primary_pattern[0])
            
            # Analyze pattern density
            density_info = self._analyze_pattern_density(image_features)
            
            # Get specific details for animal prints
            specific_details = []
            if "animal print" in primary_pattern[0] or any("print" in p["name"] for p in secondary_patterns):
                # Check for specific animal print types
                for animal_print, descriptors in self.animal_prints.items():
                    if animal_print in primary_pattern[0] or any(animal_print in p["name"] for p in secondary_patterns):
                        specific_details.extend(descriptors)
                        break
            
            # Get specific details for floral patterns
            if "floral" in primary_pattern[0] or any("floral" in p["name"] for p in secondary_patterns):
                # Check for specific floral types
                for floral_type, descriptors in self.floral_types.items():
                    if any(floral_type in e["name"] for e in detected_elements):
                        specific_details.extend(descriptors)
                        break
            
            # Log detected elements
            if detected_elements:
                logger.info(f"Detected elements: {', '.join([e['name'] for e in detected_elements[:5]])}")
            else:
                logger.info("No specific elements detected with high confidence")
            
            return {
                "category": primary_pattern[0],
                "category_confidence": float(primary_pattern[1]),
                "secondary_patterns": sorted(secondary_patterns, 
                                          key=lambda x: x["confidence"], 
                                          reverse=True)[:3],
                "elements": detected_elements[:10],  # Top 10 elements
                "specific_details": specific_details,  # Specific details for the pattern type
                "density": density_info  # Add density information
            }

        except Exception as e:
            logger.error(f"Error in pattern analysis: {e}", exc_info=True)
            return {
                "category": "Unknown",
                "category_confidence": 0.0,
                "secondary_patterns": [],
                "elements": [],
                "specific_details": [],
                "density": {
                    "type": "regular",
                    "confidence": 0.0
                }
            } 

    def _analyze_pattern_density(self, image_features) -> Dict:
        """Analyze the density and distribution of the pattern."""
        try:
            density_types = ["dense", "scattered", "sparse", "regular", "irregular", "clustered"]
            
            scores = {}
            with torch.no_grad():
                for density in density_types:
                    text_inputs = self.processor(
                        text=[f"this is a {density} pattern"],
                        return_tensors="pt",
                        padding=True
                    ).to(self.device)
                    
                    text_features = self.model.get_text_features(**text_inputs)
                    similarity = torch.nn.functional.cosine_similarity(
                        image_features.to(self.device),
                        text_features,
                        dim=1
                    )
                    scores[density] = float(similarity[0].cpu())
            
            # Get the most likely density type
            top_density = max(scores.items(), key=lambda x: x[1])
            
            return {
                'type': top_density[0],
                'confidence': top_density[1]
            }
        
        except Exception as e:
            logger.error(f"Error analyzing pattern density: {e}")
            return {
                'type': 'regular',
                'confidence': 0.0
            } 

    def _analyze_specific_elements(self, image_features, primary_pattern):
        """Analyze for specific elements based on the primary pattern category with improved relevance."""
        try:
            # Determine which specific elements to check based on primary pattern
            pattern_category = primary_pattern.lower()
            specific_elements = []
            
            # Only check for relevant elements based on pattern category
            if any(floral in pattern_category for floral in ["floral", "flower"]):
                # Only check for floral elements if it's a floral pattern
                specific_elements.extend([
                    "roses", "tulips", "daisies", "sunflowers", "peonies", "lilies", 
                    "orchids", "cherry blossoms", "poppies", "hibiscus", "lotus flowers"
                ])
                # Add some leaf elements that might accompany flowers
                specific_elements.extend(["leaves", "vines", "stems", "foliage"])
                
            elif any(botanical in pattern_category for botanical in ["leaf", "botanical", "tropical", "plant"]):
                # Only check for botanical elements if it's a leaf/botanical pattern
                specific_elements.extend([
                    "palm leaves", "fern leaves", "maple leaves", "oak leaves", 
                    "tropical leaves", "ivy", "monstera leaves", "banana leaves",
                    "succulent plants", "bamboo"
                ])
                
            elif any(animal in pattern_category for animal in ["animal", "leopard", "zebra", "tiger", "giraffe"]):
                # Only check for animal print elements if it's an animal pattern
                specific_elements.extend([
                    "leopard spots", "zebra stripes", "tiger stripes", "giraffe patches",
                    "snake scales", "crocodile texture", "animal skin"
                ])
                
            elif any(geo in pattern_category for geo in ["geometric", "abstract"]):
                # Only check for geometric elements if it's a geometric pattern
                specific_elements.extend([
                    "circles", "squares", "triangles", "diamonds", "hexagons",
                    "stars", "spirals", "zigzags", "stripes", "dots"
                ])
            
            else:
                # For unknown patterns, check common elements
                specific_elements.extend([
                    "stripes", "dots", "checks", "swirls", "waves", "grid", "lattice",
                    "flowers", "leaves", "geometric shapes"
                ])
            
            # Remove duplicates
            specific_elements = list(set(specific_elements))
            
            # Analyze for each specific element
            element_scores = {}
            with torch.no_grad():
                for element in specific_elements:
                    # Try multiple phrasings for better detection
                    phrasings = [
                        f"a pattern with {element}",
                        f"textile with {element}",
                        f"design containing {element}",
                        f"{element} pattern"
                    ]
                    
                    max_score = 0
                    for phrase in phrasings:
                        text_inputs = self.processor(
                            text=[phrase],
                            return_tensors="pt",
                            padding=True
                        ).to(self.device)
                        
                        text_features = self.model.get_text_features(**text_inputs)
                        similarity = torch.nn.functional.cosine_similarity(
                            image_features.to(self.device),
                            text_features,
                            dim=1
                        )
                        score = float(similarity[0].cpu())
                        max_score = max(max_score, score)
                    
                    element_scores[element] = max_score
            
            # Use tiered confidence levels
            high_confidence_threshold = 0.28  # Higher threshold for primary elements
            medium_confidence_threshold = 0.24  # Medium threshold for secondary elements
            
            # Get elements with high confidence
            high_confidence_elements = [
                {"name": element, "confidence": score, "confidence_level": "high"}
                for element, score in element_scores.items()
                if score > high_confidence_threshold
            ]
            
            # Get elements with medium confidence
            medium_confidence_elements = [
                {"name": element, "confidence": score, "confidence_level": "medium"}
                for element, score in element_scores.items()
                if medium_confidence_threshold < score <= high_confidence_threshold
            ]
            
            # Combine and sort by confidence
            detected_elements = high_confidence_elements + medium_confidence_elements
            detected_elements.sort(key=lambda x: x["confidence"], reverse=True)
            
            return detected_elements
        
        except Exception as e:
            logger.error(f"Error analyzing specific elements: {e}")
            return [] 