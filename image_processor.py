import cv2
import io
import base64
from PIL import Image
import yaml
from typing import Tuple, Dict, Any
import numpy as np
from pathlib import Path

class ImageProcessor:
    def __init__(self, config_path: str = "config/config.yaml", max_size: int = None):
        """
        Initialize image processor
        Args:
            config_path: Path to configuration file
            max_size: Optional override for max image size
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Get image settings from config
        self.max_size = max_size or self.config["image"]["max_size"]
        self.quality = self.config["image"]["quality"]
        self.format = self.config["image"]["format"]
        self.target_size = (
            self.config["image"]["width"],
            self.config["image"]["height"]
        )

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from yaml file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config from {config_path}: {e}")
            # Return default configuration
            return {
                "image": {
                    "width": 640,
                    "height": 480,
                    "max_size": 1568,
                    "format": "jpeg",
                    "quality": 95
                }
            }

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target size while maintaining aspect ratio
        Args:
            image: Input image
        Returns:
            Resized image
        """
        h, w = image.shape[:2]
        target_w, target_h = self.target_size
        
        # Calculate scaling factor maintaining aspect ratio
        scale = min(target_w/w, target_h/h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def process_frame(self, frame: np.ndarray) -> Tuple[str, Dict[str, Any], np.ndarray]:
        """
        Process a frame: resize, convert color, encode to base64
        Args:
            frame: numpy array (BGR format)
        Returns:
            tuple: (base64_data, metadata, processed_frame)
            - base64_data: base64 encoded image
            - metadata: dict with image info
            - processed_frame: processed numpy array
        """
        # Get original size for metadata
        orig_h, orig_w = frame.shape[:2]
        
        # Resize to target size
        frame = self.resize_image(frame)
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get processed dimensions for metadata
        height, width = frame_rgb.shape[:2]
        metadata = {
            "width": width,
            "height": height,
            "megapixels": (width * height) / 1_000_000,
            "original_size": (orig_w, orig_h),
            "target_size": self.target_size
        }
        
        # Encode to base64
        pil_image = Image.fromarray(frame_rgb)
        buffer = io.BytesIO()
        pil_image.save(buffer, format=self.format.upper(), quality=self.quality)
        image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Save debug image if configured
        if self.config.get("debug", {}).get("save_images", False):
            save_dir = Path(self.config["debug"]["save_dir"])
            save_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(save_dir / f"debug_{len(list(save_dir.glob('*.jpg')))}.jpg"), frame)
        
        return image_data, metadata, frame

    def create_mock_image(self, text: str = "Mock Camera") -> np.ndarray:
        """Create a mock image with text for testing"""
        img = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)
        
        # Scale text size based on image dimensions
        font_scale = min(self.target_size) / 640.0
        cv2.putText(img, text,
                   (int(self.target_size[0]*0.1), int(self.target_size[1]*0.5)),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale, (255, 255, 255), 2)
                   
        return img