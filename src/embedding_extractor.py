import torch
from transformers import CLIPProcessor, CLIPModel
import config
import logging

logger = logging.getLogger(__name__)

class EmbeddingExtractor:
    def __init__(self):
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            
            self.model = CLIPModel.from_pretrained(config.CLIP_MODEL_NAME).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL_NAME)
            
            # Set model to evaluation mode for better performance
            self.model.eval()
            
            logger.info(f"Loaded CLIP model: {config.CLIP_MODEL_NAME}")
        except Exception as e:
            logger.error(f"Error initializing EmbeddingExtractor: {e}")
            raise

    @torch.no_grad()
    def extract_image_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from preprocessed images."""
        try:
            if images.size(0) == 0:
                logger.warning("Empty image batch provided")
                return torch.zeros((0, config.EMBEDDING_DIM))
                
            images = images.to(self.device)
            image_features = self.model.get_image_features(images)
            
            # Normalize features for cosine similarity
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            
            return image_features.cpu().numpy()
        except Exception as e:
            logger.error(f"Error extracting image embeddings: {e}")
            return torch.zeros((images.size(0), config.EMBEDDING_DIM)).numpy()

    @torch.no_grad()
    def extract_text_embedding(self, text: str) -> torch.Tensor:
        """Extract embedding from text query."""
        try:
            inputs = self.processor(
                text=text,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            text_features = self.model.get_text_features(**inputs)
            
            # Normalize features for cosine similarity
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            
            return text_features.cpu().numpy()
        except Exception as e:
            logger.error(f"Error extracting text embedding for '{text}': {e}")
            return torch.zeros((1, config.EMBEDDING_DIM)).numpy() 