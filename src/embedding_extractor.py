import torch
from transformers import CLIPProcessor, CLIPModel
import config
from PIL import Image
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class EmbeddingExtractor:
    """Class for extracting embeddings from images using CLIP"""
    
    def __init__(self, model, processor, device):
        """Initialize with CLIP model and processor"""
        self.model = model
        self.processor = processor
        self.device = device
    
    def extract_features(self, image_path: Path) -> torch.Tensor:
        """Extract features from an image using CLIP
        
        Args:
            image_path: Path to the image file
            
        Returns:
            torch.Tensor: Image features
        """
        try:
            # Load the image
            image = Image.open(image_path)
            
            # Convert to RGB if needed (for PNG with transparency)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Process image through CLIP
            inputs = self.processor(
                images=image,
                return_tensors="pt",
                padding=True
            )
            
            # Extract features
            with torch.no_grad():
                image_features = self.model.get_image_features(
                    pixel_values=inputs['pixel_values'].to(self.device)
                )
            
            return image_features
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            raise

    def extract_image_features(self, image: Image.Image) -> torch.Tensor:
        """Extract features from a PIL Image object using CLIP
        
        Args:
            image: PIL Image object
            
        Returns:
            torch.Tensor: Image features
        """
        try:
            # Ensure image is in RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Process image through CLIP
            inputs = self.processor(
                images=image,
                return_tensors="pt",
                padding=True
            )
            
            # Extract features
            with torch.no_grad():
                image_features = self.model.get_image_features(
                    pixel_values=inputs['pixel_values'].to(self.device)
                )
            
            return image_features
            
        except Exception as e:
            logger.error(f"Error extracting features from image: {str(e)}")
            raise

    @torch.no_grad()
    def extract_image_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from preprocessed images."""
        images = images.to(self.device)
        image_features = self.model.get_image_features(images)
        return image_features.cpu().numpy()

    @torch.no_grad()
    def extract_text_embedding(self, text: str) -> torch.Tensor:
        """Extract embedding from text query."""
        inputs = self.processor(
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        text_features = self.model.get_text_features(**inputs)
        return text_features.cpu().numpy() 