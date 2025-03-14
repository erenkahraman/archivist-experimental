from PIL import Image, ImageOps, ImageEnhance
import torch
from torchvision import transforms
from pathlib import Path
import io
import config
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                              (0.26862954, 0.26130258, 0.27577711))
        ])

    def create_thumbnail(self, image_path: Path) -> Path:
        """Create and save optimized thumbnail."""
        thumb_path = config.THUMBNAIL_DIR / f"{image_path.stem}_thumb.jpg"
        
        if thumb_path.exists():
            return thumb_path

        with Image.open(image_path) as img:
            img = img.convert('RGB')
            
            # Enhance image for better pattern recognition
            img = self._enhance_image(img)
            
            img.thumbnail((config.IMAGE_SIZE, config.IMAGE_SIZE))
            img.save(thumb_path, 'JPEG', quality=85, optimize=True)
        
        return thumb_path

    def _enhance_image(self, img: Image.Image) -> Image.Image:
        """Enhance image for better pattern and element detection."""
        try:
            # Slightly increase contrast
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.2)
            
            # Slightly increase sharpness
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.3)
            
            return img
        except Exception as e:
            logger.warning(f"Image enhancement failed: {e}")
            return img

    def preprocess_image(self, image_path: Path) -> torch.Tensor:
        """Preprocess image for CLIP model."""
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            # Apply enhancements for better pattern detection
            img = self._enhance_image(img)
            return self.transform(img).unsqueeze(0)

    def process_batch(self, image_paths: list[Path]) -> tuple[list[Path], torch.Tensor]:
        """Process a batch of images."""
        thumbnails = []
        tensors = []
        
        for path in image_paths:
            thumb_path = self.create_thumbnail(path)
            thumbnails.append(thumb_path)
            tensors.append(self.preprocess_image(thumb_path))
        
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
            image.verify()
            return True
        except Exception as e:
            logger.error(f"Invalid image: {e}")
            return False 