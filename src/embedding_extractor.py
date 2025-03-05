import torch
from transformers import CLIPProcessor, CLIPModel
import config

class EmbeddingExtractor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(config.CLIP_MODEL_NAME).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL_NAME)

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