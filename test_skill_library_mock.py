import time
import threading
import numpy as np
import cv2
import base64
import argparse
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional
from skill_library import SkillLibrary, RobotPlatform, RESET_POSITIONS, Skill, RobotCommand
from model_handler import ModelHandler
from utils.image_processor import ImageProcessor

def load_config(config_path: str = "config/test_config.yaml") -> Dict:
    """Load configuration from yaml file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        print("Using default configuration")
        return {
            "image": {
                "width": 640,
                "height": 480,
                "max_size": 1568,
                "format": "jpg",
                "quality": 95
            },
            "test": {
                "max_attempts": 3,
                "check_interval": 2.0,
                "timeout": 30.0
            }
        }

@dataclass
class MockRobot:
    """Mock robot class for testing without hardware"""
    def __init__(self):
        self.joints = []
        self.joint_dict = {
            "left_shoulder_pitch": 0.0,
            "left_elbow_yaw": 0.0,
            "right_shoulder_pitch": 0.0,
            "right_elbow_yaw": 0.0,
        }
        self.camera_image = None
        self.motors_enabled = False
        self.image_processor = ImageProcessor()
        
    def initialize(self):
        print("Mock robot initialized")
        self.motors_enabled = True
        
    def disable_motors(self):
        print("Mock robot motors disabled")
        self.motors_enabled = False
        
    def set_desired_positions(self, positions: Dict[str, float]):
        print(f"Mock robot setting positions: {positions}")
        for joint, pos in positions.items():
            if joint in self.joint_dict:
                self.joint_dict[joint] = pos
        
    def get_camera_image(self) -> Optional[np.ndarray]:
        if self.camera_image is None:
            return self.image_processor.create_mock_image()
        return self.camera_image

def create_debug_skills(skill_lib: SkillLibrary):
    """Add debug/test skills to the library"""
    
    print("Adding debug skills to library...")
    
    # Rock Paper Scissors skill
    def play_rps(robot, **kwargs):
        print("Robot is choosing...")
        print("Robot plays: PAPER")
        # Set robot joints to "paper" position
        positions = {
            "left_shoulder_pitch": 90.0,  # Raise arm
            "left_elbow_yaw": 0.0,       # Straighten elbow
            "right_shoulder_pitch": 90.0, # Raise arm
            "right_elbow_yaw": 0.0,      # Straighten elbow
        }
        robot.set_desired_positions(positions)
        time.sleep(1.0)  # Give time for pose
        return True

    skill_lib.add_skill(Skill(
        name="play_rock_paper_scissors",
        description="Play a game of rock paper scissors with the robot - robot always plays paper",
        platform_commands={
            RobotPlatform.GENERIC: RobotCommand(
                platform=RobotPlatform.GENERIC,
                command_fn=play_rps
            )
        },
        objective="Robot should play paper and wait for human player to show their choice",
        timeout_seconds=10.0,
        check_interval=2.0,
        requires_validation=True
    ))
    
    # Test pose skill
    skill_lib.add_skill(Skill(
        name="strike_pose",
        description="Make the robot strike a specific pose for testing",
        platform_commands={
            RobotPlatform.GENERIC: RobotCommand(
                platform=RobotPlatform.GENERIC,
                command_fn=lambda robot, **kwargs: print("Robot striking pose...")
            )
        },
        objective="Robot should be in the requested pose position",
        timeout_seconds=5.0,
        check_interval=1.0,
        requires_validation=True
    ))
    
    print(f"Added debug skills. Available skills: {skill_lib.get_skill_names()}")

def test_skill_library_mock(use_webcam=False):
    """Test the skill library functionality with mock robot"""
    print("\n=== Starting Mock Skill Library Test ===")
    
    skill_lib = SkillLibrary()
    robot = MockRobot()
    model_handler = ModelHandler()
    image_processor = ImageProcessor()
    cap = None
    
    # Add debug skills before testing
    create_debug_skills(skill_lib)
    
    print("\nAvailable Skills:")
    print(skill_lib.get_skill_descriptions())
    
    if use_webcam:
        try:
            cap = cv2.VideoCapture(0)  # Use default camera
            if not cap.isOpened():
                raise RuntimeError("Failed to open webcam")
                
            def get_camera_image():
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError("Failed to capture frame")
                return frame
                
            robot.get_camera_image = get_camera_image
        except Exception as e:
            print(f"Error setting up camera: {e}")
            print("Falling back to mock images")
            use_webcam = False
    
    try:
        print("\nInitializing mock robot...")
        robot.initialize()
        
        # Test sequence of skills including debug skills
        test_sequence = [
            # ("strike_pose", "Robot should be in a T-pose position"),
            ("play_rock_paper_scissors", "Robot should play rock paper scissors and determine the winner"),
        ]
        
        for skill_name, objective in test_sequence:
            print(f"\nExecuting {skill_name} with objective: {objective}")
            
            attempts = 0
            while attempts < 3:  # Maximum 3 attempts per skill
                attempts += 1
                print(f"\nAttempt {attempts}")
                
                # Execute skill
                success = skill_lib.execute_skill(skill_name, RobotPlatform.GENERIC, robot)
                if not success:
                    print(f"Failed to execute {skill_name}")
                    continue
                
                # Get image for verification
                image = robot.get_camera_image()
                
                # Convert image to base64 for model
                base64_data, metadata = image_processor.process_image(image)
                
                if metadata["width"] != image_processor.output_size[0]:
                    print(f"Warning: Image width {metadata['width']} differs from configured {image_processor.output_size[0]}")
                
                # Check if objective achieved
                decision = model_handler.action_agent(
                    skill_name,
                    objective,
                    base64_data,
                    attempts
                )
                
                print(f"Action agent decision:")
                print(f"Analysis: {decision['analysis']}")
                print(f"Continue: {'Yes' if decision['continue_execution'] else 'No'}")
                print(f"Reason: {decision['reason']}")
                
                if not decision["continue_execution"]:
                    print(f"Successfully completed {skill_name}")
                    break
                    
                time.sleep(2)
        
        print("\n=== Test completed successfully! ===")
        
    except Exception as e:
        print(f"\nError during test: {str(e)}")
        raise
    finally:
        print("\nCleaning up...")
        robot.disable_motors()
        if cap is not None:
            cap.release()
    
    # Save debug images if configured
    if config["debug"]["save_images"]:
        save_dir = Path(config["debug"]["save_dir"])
        save_dir.mkdir(exist_ok=True)
        
def test_interactive_mode():
    """Interactive test mode for trying individual skills"""
    skill_lib = SkillLibrary()
    robot = MockRobot()
    create_debug_skills(skill_lib)
    
    print("\nAvailable skills:")
    for i, name in enumerate(skill_lib.get_skill_names(), 1):
        print(f"{i}. {name}")
    
    while True:
        try:
            choice = input("\nEnter skill number to test (or 'q' to quit): ")
            if choice.lower() == 'q':
                break
                
            idx = int(choice) - 1
            skill_names = skill_lib.get_skill_names()
            if 0 <= idx < len(skill_names):
                skill_name = skill_names[idx]
                objective = input("Enter objective (or press Enter for default): ").strip()
                if not objective:
                    objective = f"Test execution of {skill_name}"
                    
                print(f"\nExecuting {skill_name} with objective: {objective}")
                skill_lib.execute_skill(skill_name, RobotPlatform.GENERIC, robot)
            else:
                print("Invalid skill number")
                
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description='Test Skill Library with Mock Robot')
    parser.add_argument('--webcam', 
                       action='store_true',
                       help='Use real webcam instead of mock images')
    parser.add_argument('--interactive',
                       action='store_true',
                       help='Run in interactive mode to test individual skills')
    parser.add_argument('--config',
                       type=str,
                       default="config/test_config.yaml",
                       help='Path to configuration file')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    print("=== Mock Skill Library Test Suite ===")
    print("This test suite verifies the functionality without real robot hardware")
    
    try:
        if args.interactive:
            test_interactive_mode()
        else:
            test_skill_library_mock(use_webcam=args.webcam)
            
        print("\n=== All tests completed successfully! ===")
        
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
    except Exception as e:
        print(f"\nTests failed with error: {str(e)}")
        raise 