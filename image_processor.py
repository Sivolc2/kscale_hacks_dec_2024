import cv2
import io
import base64
from PIL import Image

class ImageProcessor:
    def __init__(self, max_size=1568):
        """
        Initialize image processor
        Args:
            max_size (int): Maximum size for image's longest edge
        """
        self.max_size = max_size

    def downsample_image(self, image):
        """
        Downsample image if it exceeds max size while maintaining aspect ratio
        Args:
            image: numpy array (BGR format)
        Returns:
            downsampled image
        """
        height, width = image.shape[:2]
        
        # Calculate scaling factor if image exceeds max size
        if max(height, width) > self.max_size:
            scale = self.max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Resize image
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return image

    def process_frame(self, frame):
        """
        Process a frame: downsample, convert to RGB, and prepare for Claude
        Args:
            frame: numpy array (BGR format)
        Returns:
            tuple: (base64_data, image_info)
            - base64_data: base64 encoded JPEG
            - image_info: tuple of (width, height, megapixels)
        """
        # Downsample if needed
        frame = self.downsample_image(frame)
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get image dimensions for logging
        height, width = frame.shape[:2]
        megapixels = (width * height) / 1_000_000
        
        # Convert to base64 for Claude
        pil_image = Image.fromarray(frame_rgb)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=95)
        image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return image_data, (width, height, megapixels), frame 