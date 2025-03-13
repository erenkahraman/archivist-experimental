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
        
        # Enhanced pattern elements for detection
        self.pattern_elements = {
            "floral": ["roses", "tulips", "daisies", "sunflowers", "peonies", "lilies", "orchids", 
                      "wildflowers", "lotus", "cherry blossoms", "hibiscus", "poppies"],
            "botanical": ["leaves", "ferns", "palm leaves", "vines", "branches", "trees", 
                         "foliage", "tropical leaves", "monstera leaves", "ivy"],
            "geometric": ["circles", "squares", "triangles", "hexagons", "diamonds", "stars", 
                         "spirals", "chevrons", "zigzags", "stripes", "dots", "grid"],
            "animal": ["leopard spots", "zebra stripes", "tiger stripes", "peacock feathers", 
                      "butterflies", "birds", "fish", "insects", "snakeskin"],
            "celestial": ["stars", "moons", "planets", "constellations", "galaxies", "suns"],
            "cultural": ["paisley", "mandala", "arabesque", "celtic knots", "greek key", 
                        "ikat", "batik", "tribal motifs", "folk art", "oriental designs"]
        }

    def _analyze_patterns(self, image_features) -> Dict:
        """
        Analyze patterns in the image features.
        
        Args:
            image_features (torch.Tensor): Image features from CLIP model
            
        Returns:
            Dict: Pattern analysis results
        """
        try:
            logger.info("Starting pattern analysis...")
            
            pattern_categories = [
                "geometric", "floral", "abstract", "stripes", "polka dots", 
                "chevron", "paisley", "plaid", "animal print", "tribal",
                "damask", "herringbone", "houndstooth", "ikat", "lattice",
                "medallion", "moroccan", "ogee", "quatrefoil", "trellis"
            ]
            
            scores = {}
            with torch.no_grad():
                for category in pattern_categories:
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
                    scores[category] = float(similarity[0].cpu())

            primary_pattern = max(scores.items(), key=lambda x: x[1])
            
            threshold = 0.2
            secondary_patterns = [
                {"name": pattern, "confidence": score}
                for pattern, score in scores.items()
                if score > threshold and pattern != primary_pattern[0]
            ]
            
            # Analyze specific pattern elements
            pattern_elements = self._detect_pattern_elements(image_features)
            
            return {
                "category": primary_pattern[0],
                "category_confidence": float(primary_pattern[1]),
                "secondary_patterns": sorted(secondary_patterns, 
                                          key=lambda x: x["confidence"], 
                                          reverse=True)[:3],
                "elements": pattern_elements
            }

        except Exception as e:
            logger.error(f"Error in pattern analysis: {e}", exc_info=True)
            return {
                "category": "Unknown",
                "category_confidence": 0.0,
                "secondary_patterns": [],
                "elements": []
            }
            
    def _detect_pattern_elements(self, image_features) -> List[Dict]:
        """
        Detect specific elements within the pattern (flowers, leaves, stars, etc.)
        
        Args:
            image_features (torch.Tensor): Image features from CLIP model
            
        Returns:
            List[Dict]: Detected elements with confidence scores
        """
        try:
            logger.info("Detecting specific pattern elements...")
            
            # Flatten our element categories for detection
            all_elements = []
            for category, elements in self.pattern_elements.items():
                all_elements.extend(elements)
            
            # Add descriptive adjectives for common elements - precompute these
            descriptive_queries = [
                # Floral descriptors
                "a pattern with vibrant flowers", "a pattern with delicate flowers",
                "a pattern with bold floral elements", "a pattern with intricate floral details",
                
                # Leaf descriptors
                "a pattern with lush green leaves", "a pattern with tropical palm leaves",
                "a pattern with detailed leaf textures", "a pattern with overlapping foliage",
                
                # Geometric descriptors
                "a pattern with bold geometric shapes", "a pattern with intricate geometric details",
                "a pattern with precise geometric elements", "a pattern with layered geometric forms",
                
                # Texture descriptors
                "a pattern with rich textural details", "a pattern with subtle texture variations",
                "a pattern with dimensional textures", "a pattern with contrasting textures"
            ]
            
            # Process all queries in batches for better performance
            element_scores = {}
            descriptive_element_scores = {}
            
            with torch.no_grad():
                # Process regular elements in batches
                batch_size = 16  # Adjust based on your GPU memory
                for i in range(0, len(all_elements), batch_size):
                    batch_elements = all_elements[i:i+batch_size]
                    
                    # Positive prompts
                    pos_texts = [f"a pattern containing {element}" for element in batch_elements]
                    pos_inputs = self.processor(
                        text=pos_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    ).to(self.device)
                    
                    pos_features = self.model.get_text_features(**pos_inputs)
                    
                    # Negative prompts
                    neg_texts = [f"a pattern without {element}" for element in batch_elements]
                    neg_inputs = self.processor(
                        text=neg_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    ).to(self.device)
                    
                    neg_features = self.model.get_text_features(**neg_inputs)
                    
                    # Calculate similarities
                    for j, element in enumerate(batch_elements):
                        pos_similarity = torch.nn.functional.cosine_similarity(
                            image_features.to(self.device),
                            pos_features[j:j+1],
                            dim=1
                        )
                        
                        neg_similarity = torch.nn.functional.cosine_similarity(
                            image_features.to(self.device),
                            neg_features[j:j+1],
                            dim=1
                        )
                        
                        # Calculate differential score
                        differential = float(pos_similarity[0].cpu()) - float(neg_similarity[0].cpu())
                        element_scores[element] = differential
                
                # Process descriptive queries in batches
                for i in range(0, len(descriptive_queries), batch_size):
                    batch_queries = descriptive_queries[i:i+batch_size]
                    
                    inputs = self.processor(
                        text=batch_queries,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    ).to(self.device)
                    
                    features = self.model.get_text_features(**inputs)
                    
                    # Calculate similarities
                    similarities = torch.nn.functional.cosine_similarity(
                        image_features.to(self.device),
                        features,
                        dim=1
                    )
                    
                    # Store scores
                    for j, query in enumerate(batch_queries):
                        descriptor = query.replace("a pattern with ", "")
                        descriptive_element_scores[descriptor] = float(similarities[j].cpu())
            
            # Filter elements with significant positive scores
            threshold = 0.05
            detected_elements = [
                {"name": element, "confidence": score}
                for element, score in element_scores.items()
                if score > threshold
            ]
            
            # Add descriptive elements that scored well
            desc_threshold = 0.25  # Higher threshold for descriptive elements
            for descriptor, score in descriptive_element_scores.items():
                if score > desc_threshold:
                    detected_elements.append({"name": descriptor, "confidence": score})
            
            # Sort by confidence and return top elements
            return sorted(detected_elements, key=lambda x: x["confidence"], reverse=True)[:6]
            
        except Exception as e:
            logger.error(f"Error detecting pattern elements: {e}", exc_info=True)
            return [] 