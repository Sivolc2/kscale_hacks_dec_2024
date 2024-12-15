import time
import cv2
from datetime import datetime
import os
import argparse
from image_processor import ImageProcessor
from skill_library import SkillLibrary, RobotPlatform
from typing import List, Tuple, Optional, Dict
from robot import Robot
import threading
from model_handler import ModelHandler

class RobotMonitor:
    def __init__(self, 
                 capture_interval=5, 
                 save_dir="pictures/webcam", 
                 max_size=1568, 
                 platform=RobotPlatform.ZEROTH,
                 voice_mode=False,
                 voice_processor=None):
        """Initialize the robot monitoring system"""
        self.model_handler = ModelHandler()
        self.capture_interval = capture_interval
        self.cap = None
        self.save_dir = save_dir
        self.frame_count = 0
        self.processor = ImageProcessor(max_size=max_size)
        self.skill_library = SkillLibrary()
        self.platform = platform
        self.robot: Optional[Robot] = None
        self.walk_stop_event = None
        self.walk_thread = None
        
        # Voice mode settings
        self.voice_mode = voice_mode
        self.voice_processor = voice_processor
        
        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)

    def initialize_robot(self) -> None:
        """Initialize the robot hardware"""
        try:
            self.robot = Robot()
            self.robot.initialize()
            print("Robot initialized successfully")
        except Exception as e:
            print(f"Failed to initialize robot: {e}")
            raise

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

    def process_frame_with_voice(self, image_data: str, voice_command: Optional[str] = None) -> str:
        """Process frame with optional voice command input"""
        # Get model's analysis
        response = self.model_handler.analyze_scene(
            image_data, 
            self.skill_library.get_skill_descriptions(),
            voice_command
        )
        
        # Validate response format
        valid_steps = self.model_handler.validate_response(response)
        if not valid_steps:
            response += "\n\nNo valid steps were found in the response."
            return response
            
        # Extract and execute skills
        skills = self.model_handler.extract_skills(response, self.skill_library.skills)
        if skills:
            print("\nExecuting planned skills:")
            results = self.run_skills(skills)
            
            # Add execution results to response
            response += "\n\nExecution results:"
            for skill, success in results.items():
                status = "✓ Success" if success else "✗ Failed"
                response += f"\n- [{skill}]: {status}"
        
        return response

    def run_skills(self, skills: List[str]) -> Dict[str, bool]:
        """
        Run a sequence of skills and return their success status
        Returns dict mapping skill names to success boolean
        """
        results = {}
        for skill_name in skills:
            try:
                print(f"\nExecuting skill: {skill_name}")
                
                if skill_name == "walk_forward":
                    # Handle walking specially with threading
                    self.walk_stop_event = threading.Event()
                    self.walk_thread = threading.Thread(
                        target=lambda: self.execute_skill(
                            "walk_forward",
                            stop_event=self.walk_stop_event
                        )
                    )
                    self.walk_thread.start()
                    # Let it walk for 5 seconds
                    time.sleep(5)
                    self.walk_stop_event.set()
                    self.walk_thread.join(timeout=2)
                    self.walk_stop_event = None
                    self.walk_thread = None
                else:
                    # Execute other skills normally
                    self.execute_skill(skill_name)
                    
                results[skill_name] = True
                print(f"Successfully executed: {skill_name}")
            except Exception as e:
                print(f"Failed to execute {skill_name}: {e}")
                results[skill_name] = False
        return results

    def execute_skill(self, skill_name: str, **kwargs):
        """Execute a skill on the current platform"""
        if self.robot is None:
            raise RuntimeError("Robot not initialized. Call initialize_robot() first.")
        return self.skill_library.execute_skill(skill_name, self.platform, self.robot, **kwargs)

    def get_voice_command(self) -> Optional[str]:
        """Get transcribed voice command from the voice processor"""
        if not self.voice_mode or not self.voice_processor:
            return None
        
        try:
            # Get command if available
            command = self.voice_processor.get_command()
            if command:
                print(f"Voice command received: {command}")
                
                # Get recent context
                context = self.voice_processor.get_recent_context()
                if len(context) > 1:
                    print("Recent conversation context:")
                    for text in context[-3:]:  # Show last 3 entries
                        print(f"  - {text}")
                    
                return command
        except Exception as e:
            print(f"Error processing voice command: {e}")
        return None

    def run_monitoring(self):
        """Main monitoring loop"""
        try:
            self.initialize_robot()
            self.initialize_camera()
            
            print(f"Starting robot monitoring...")
            print(f"Saving images to: {self.save_dir}")
            print(f"Capture interval: {self.capture_interval} seconds")
            print(f"Max image dimension: {self.processor.max_size}px")
            print(f"Voice mode: {'enabled' if self.voice_mode else 'disabled'}")
            
            # Start voice processing if enabled
            if self.voice_mode and self.voice_processor:
                self.voice_processor.start_listening()
            
            while True:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n[{timestamp}] Capturing frame...")
                
                voice_command = self.get_voice_command() if self.voice_mode else None
                image_data, save_path, (width, height, mp) = self.capture_frame()
                
                print(f"Saved image to: {save_path}")
                print(f"Image size: {width}x{height}px ({mp:.2f} MP)")
                
                response = self.process_frame_with_voice(image_data, voice_command)
                print(f"Analysis and execution results:\n{response}")
                
                time.sleep(self.capture_interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        finally:
            # Stop voice processing
            if self.voice_mode and self.voice_processor:
                self.voice_processor.stop_listening()
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        # Stop any ongoing walking
        if self.walk_stop_event is not None:
            self.walk_stop_event.set()
            if self.walk_thread is not None:
                self.walk_thread.join(timeout=2)
                    
        if self.cap is not None:
            self.cap.release()
        if self.robot is not None:
            try:
                self.robot.disable_motors()
                print("Robot motors disabled")
            except Exception as e:
                print(f"Error disabling robot motors: {e}")

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
    parser.add_argument('-p', '--platform', 
                       type=str, 
                       choices=['zeroth', 'generic'],
                       default="zeroth",
                       help='Robot platform (default: zeroth)')
    parser.add_argument('-v', '--voice-mode',
                       action='store_true',
                       help='Enable voice command mode')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Initialize and run monitor with specified parameters
    monitor = RobotMonitor(
        capture_interval=args.interval,
        save_dir=args.directory,
        max_size=args.max_size,
        platform=RobotPlatform[args.platform.upper()],
        voice_mode=args.voice_mode
    )
    
    try:
        monitor.run_monitoring()
    except Exception as e:
        print(f"Error during monitoring: {e}")
        # Ensure robot is disabled even if an error occurs
        if monitor.robot is not None:
            try:
                monitor.robot.disable_motors()
                print("Robot motors disabled after error")
            except Exception as disable_error:
                print(f"Error disabling robot motors: {disable_error}")

