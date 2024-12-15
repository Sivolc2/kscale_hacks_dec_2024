import time
import cv2
import anthropic
from datetime import datetime
import os
import argparse
from image_processor import ImageProcessor

class RobotMonitor:
    def __init__(self, capture_interval=5, save_dir="pictures/webcam", max_size=1568):
        """
        Initialize the robot monitoring system
        Args:
            capture_interval (int): Seconds between captures
            save_dir (str): Directory to save captured images
            max_size (int): Maximum size for image's longest edge
        """
        self.client = anthropic.Anthropic()
        self.capture_interval = capture_interval
        self.cap = None
        self.save_dir = save_dir
        self.frame_count = 0
        self.processor = ImageProcessor(max_size=max_size)
        
        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)

    def initialize_camera(self, camera_id=0):
        """Initialize webcam capture"""
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open webcam")

    def capture_frame(self):
        """Capture a frame, save it, and convert to base64"""
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")
        
        # Process the frame
        image_data, image_info, processed_frame = self.processor.process_frame(frame)
        
        # Save the processed image
        self.frame_count += 1
        save_path = os.path.join(self.save_dir, f"pic_{self.frame_count}.jpg")
        cv2.imwrite(save_path, processed_frame)
        
        return image_data, save_path, image_info

    def process_frame(self, image_data):
        """Send frame to Claude and get response"""
        message = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": "Analyze this robot state image. What are the key observations and what should be the next control action? Be concise."
                        }
                    ],
                }
            ],
        )
        return message.content

    def run_monitoring(self):
        """Main monitoring loop"""
        try:
            self.initialize_camera()
            print(f"Starting robot monitoring...")
            print(f"Saving images to: {self.save_dir}")
            print(f"Capture interval: {self.capture_interval} seconds")
            print(f"Max image dimension: {self.processor.max_size}px")
            
            while True:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n[{timestamp}] Capturing frame...")
                
                image_data, save_path, (width, height, mp) = self.capture_frame()
                print(f"Saved image to: {save_path}")
                print(f"Image size: {width}x{height}px ({mp:.2f} MP)")
                
                response = self.process_frame(image_data)
                print(f"Claude's analysis:\n{response}")
                
                time.sleep(self.capture_interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        finally:
            if self.cap is not None:
                self.cap.release()

def parse_args():
    parser = argparse.ArgumentParser(description='Robot Monitor with Webcam')
    parser.add_argument('-i', '--interval', 
                       type=float, 
                       default=5.0,
                       help='Capture interval in seconds (default: 5.0)')
    parser.add_argument('-d', '--directory', 
                       type=str, 
                       default="pictures/webcam",
                       help='Directory to save captured images (default: pictures/webcam)')
    parser.add_argument('-c', '--camera', 
                       type=int, 
                       default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('-s', '--max-size', 
                       type=int, 
                       default=1568,
                       help='Maximum image dimension in pixels (default: 1568)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Initialize and run monitor with specified parameters
    monitor = RobotMonitor(
        capture_interval=args.interval,
        save_dir=args.directory,
        max_size=args.max_size
    )
    monitor.initialize_camera(camera_id=args.camera)
    monitor.run_monitoring()

