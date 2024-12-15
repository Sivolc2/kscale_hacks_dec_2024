import time
import cv2
import anthropic
from datetime import datetime
import os
import argparse
from image_processor import ImageProcessor
from skill_library import SkillLibrary, RobotPlatform
from typing import List, Tuple, Optional, Dict
from robot import Robot
import re
import threading

class RobotMonitor:
    def __init__(self, capture_interval=5, save_dir="pictures/webcam", max_size=1568, platform=RobotPlatform.ZEROTH):
        """
        Initialize the robot monitoring system
        Args:
            capture_interval (int): Seconds between captures
            save_dir (str): Directory to save captured images
            max_size (int): Maximum size for image's longest edge
            platform (RobotPlatform): Robot platform
        """
        self.client = anthropic.Anthropic()
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

    def plan_task(self, task_description: str) -> List[Tuple[str, float]]:
        """
        Break down high-level task into sequence of skills using SayCan approach
        Returns list of (skill_name, confidence) tuples
        """
        # Construct prompt for task planning
        prompt = f"""You are a helpful robot assistant that can execute the following skills:

{self.skill_library.get_skill_descriptions()}

For the task of moving to another spot: "{task_description}"

Break this down into a sequence of skills from the above list. Format your response as:
1. [skill_name]: Brief justification
2. [skill_name]: Brief justification
...

Only use skills from the provided list. Be concise but clear in your justifications.
If there is nothing to do, just send the wave command
"""

        # Get plan from Claude
        message = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            messages=[{
                "role": "user", 
                "content": prompt
            }]
        )
        
        # Parse response into skill sequence
        plan = []
        for line in message.content.split('\n'):
            if not line.strip() or not line[0].isdigit():
                continue
            # Extract skill name from between [ ]
            if '[' in line and ']' in line:
                skill_name = line[line.find('[')+1:line.find(']')]
                if skill_name in self.skill_library.skills:
                    # Get affordance score for this skill
                    affordance = self.skill_library.skills[skill_name].get_affordance()
                    plan.append((skill_name, affordance))
        
        return plan

    def extract_skills_from_response(self, response: str) -> List[str]:
        """
        Extract skill names from Claude's response
        Returns list of skill names in order
        """
        skills = []
        # Look for numbered list items with [skill_name]
        for line in response.split('\n'):
            if re.match(r'^\d+\.\s*\[([^\]]+)\]', line):
                skill_name = re.findall(r'\[([^\]]+)\]', line)[0]
                if skill_name in self.skill_library.skills:
                    skills.append(skill_name)
        return skills

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
                            "text": f"""Given an external view of a robot. What task needs to be done?
                            After describing the scene, break down the needed task into specific steps using available skills:
                            
                            Available skills:
                            {self.skill_library.get_skill_descriptions()}
                            
                            Format your response as:
                            Scene description: [brief description]
                            Required task: [task description]
                            Planned steps:
                            1. [skill_name]: Brief justification
                            2. [skill_name]: Brief justification
                            ..."""
                        }
                    ],
                }
            ],
        )
        
        response = message.content
        
        # Extract and execute skills if task identified
        if "Required task:" in response:
            skills = self.extract_skills_from_response(response)
            if skills:
                print("\nExecuting planned skills:")
                results = self.run_skills(skills)
                
                # Add execution results to response
                response += "\n\nExecution results:"
                for skill, success in results.items():
                    status = "✓ Success" if success else "✗ Failed"
                    response += f"\n- {skill}: {status}"
            
        return response

    def execute_skill(self, skill_name: str, **kwargs):
        """Execute a skill on the current platform"""
        if self.robot is None:
            raise RuntimeError("Robot not initialized. Call initialize_robot() first.")
        return self.skill_library.execute_skill(skill_name, self.platform, self.robot, **kwargs)

    def run_monitoring(self):
        """Main monitoring loop"""
        try:
            # Initialize robot first
            self.initialize_robot()
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
                print(f"Claude's analysis and execution results:\n{response}")
                
                time.sleep(self.capture_interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        finally:
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
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Initialize and run monitor with specified parameters
    monitor = RobotMonitor(
        capture_interval=args.interval,
        save_dir=args.directory,
        max_size=args.max_size,
        platform=RobotPlatform[args.platform.upper()]
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

