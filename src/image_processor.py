from PIL import Image, ImageOps
import torch
from torchvision import transforms
from pathlib import Path
import io
import config
import logging
import os

logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                              (0.26862954, 0.26130258, 0.27577711))
        ])
        
        # Create thumbnail directory if it doesn't exist
        os.makedirs(config.THUMBNAIL_DIR, exist_ok=True)

    def create_thumbnail(self, image_path: Path) -> Path:
        """Create and save optimized thumbnail."""
        thumb_path = config.THUMBNAIL_DIR / f"{image_path.stem}_thumb.jpg"
        
        if thumb_path.exists():
            return thumb_path

        try:
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                img.thumbnail((config.IMAGE_SIZE, config.IMAGE_SIZE))
                img.save(thumb_path, 'JPEG', quality=85, optimize=True)
            
            return thumb_path
        except Exception as e:
            logger.error(f"Error creating thumbnail for {image_path}: {e}")
            # Return original path if thumbnail creation fails
            return image_path

    def preprocess_image(self, image_path: Path) -> torch.Tensor:
        """Preprocess image for CLIP model."""
        try:
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                return self.transform(img).unsqueeze(0)
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            # Return a blank tensor if preprocessing fails
            return torch.zeros((1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE))

    def process_batch(self, image_paths: list[Path]) -> tuple[list[Path], torch.Tensor]:
        """Process a batch of images."""
        thumbnails = []
        tensors = []
        
        for path in image_paths:
            try:
                thumb_path = self.create_thumbnail(path)
                thumbnails.append(thumb_path)
                tensors.append(self.preprocess_image(thumb_path))
            except Exception as e:
                logger.error(f"Error processing image {path}: {e}")
                # Skip failed images
                continue
        
        if not tensors:
            return [], torch.zeros((0, 3, config.IMAGE_SIZE, config.IMAGE_SIZE))
            
        return thumbnails, torch.cat(tensors)

    def validate_image(self, image: Image.Image) -> bool:
        """
        Validate image before processing.
        
        Args:
            image (PIL.Image): Image to validate
            
        Returns:
            bool: True if image is valid
        """
        try:
            # Check if image is None
            if image is None:
                logger.error("Image is None")
                return False
            
            # Check image mode
            if image.mode not in ['RGB', 'RGBA']:
                logger.warning(f"Unusual image mode: {image.mode}, attempting to convert to RGB")
                try:
                    image = image.convert('RGB')
                except Exception as e:
                    logger.error(f"Failed to convert image to RGB: {e}")
                    return False
                
            # Check image size
            if image.width < 10 or image.height < 10:
                logger.error(f"Image too small: {image.width}x{image.height}")
                return False
            
            # Check if image is corrupt
            try:
                image.verify()
                return True
            except Exception as e:
                logger.error(f"Image verification failed: {e}")
                return False
            
        except Exception as e:
            logger.error(f"Image validation error: {e}")
            return False 