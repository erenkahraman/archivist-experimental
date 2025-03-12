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
            
            return {
                "category": primary_pattern[0],
                "category_confidence": float(primary_pattern[1]),
                "secondary_patterns": sorted(secondary_patterns, 
                                          key=lambda x: x["confidence"], 
                                          reverse=True)[:3]
            }

        except Exception as e:
            logger.error(f"Error in pattern analysis: {e}", exc_info=True)
            return {
                "category": "Unknown",
                "category_confidence": 0.0,
                "secondary_patterns": []
            } 