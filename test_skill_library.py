import time
import threading
import re
import numpy as np
import cv2
import base64
import argparse
from skill_library import SkillLibrary, RobotPlatform, RESET_POSITIONS
from robot import Robot
from model_handler import ModelHandler

def test_skill_library(dummy_mode=False):
    """
    Test the skill library functionality with real robot
    
    Args:
        dummy_mode: If True, use dummy black images instead of webcam
    """
    print("\n=== Starting Skill Library Test ===")
    print(f"Mode: {'Dummy (black images)' if dummy_mode else 'Webcam'}")
    
    skill_lib = SkillLibrary()
    robot = Robot()
    model_handler = ModelHandler()
    cap = None
    
    # Print available skills
    print("\nAvailable Skills:")
    print(skill_lib.get_skill_descriptions())
    
    if dummy_mode:
        # Mock camera image for testing
        def get_camera_image():
            # Create a simple test image (black square)
            return np.zeros((480, 640, 3), dtype=np.uint8)
    else:
        # Initialize real webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Failed to open webcam")
            
        def get_camera_image():
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Failed to capture frame from webcam")
            return frame
    
    # Add camera function to robot
    robot.get_camera_image = get_camera_image
    
    try:
        print("\nInitializing robot...")
        robot.initialize()
        
        print("\n1. Testing 'reset_positions' skill...")
        # Simply reset to default positions
        robot.set_desired_positions(RESET_POSITIONS)
        time.sleep(2)
        
        # Test sequence of skills
        test_sequence = [
            ("stand", "Robot is standing upright with both feet flat on the ground"),
            ("walk_forward", "Robot has moved forward towards wall")
        ]
        
        for skill_name, objective in test_sequence:
            print(f"\nExecuting {skill_name} with objective: {objective}")
            
            attempts = 0
            while attempts < 3:  # Maximum 3 attempts per skill
                attempts += 1
                print(f"\nAttempt {attempts}")
                
                if skill_name in ["walk_forward", "walk_backward"]:
                    # Handle walking skills with stop event
                    stop_event = threading.Event()
                    walk_thread = threading.Thread(
                        target=lambda: skill_lib.execute_skill(
                            skill_name,
                            RobotPlatform.ZEROTH,
                            robot,
                            stop_event=stop_event
                        )
                    )
                    walk_thread.start()
                    print(f"{skill_name} started... will stop in 5 seconds")
                    time.sleep(5)
                    print("Stopping walk...")
                    stop_event.set()
                    walk_thread.join(timeout=2)
                else:
                    # Execute other skills normally
                    success = skill_lib.execute_skill(skill_name, RobotPlatform.ZEROTH, robot)
                    if not success:
                        print(f"Failed to execute {skill_name}")
                        continue
                
                # Get image for verification
                image = robot.get_camera_image()
                # Convert image to base64 for model
                success, buffer = cv2.imencode('.jpg', image)
                if not success:
                    print("Error encoding image")
                    continue
                image_data = base64.b64encode(buffer).decode('utf-8')
                
                # Check if objective achieved
                decision = model_handler.action_agent(
                    skill_name,
                    objective,
                    image_data,
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
        
        print("\nFinal position reset...")
        robot.set_desired_positions(RESET_POSITIONS)
        
        print("\n=== Test completed successfully! ===")
        
    except Exception as e:
        print(f"\nError during test: {str(e)}")
        raise
    finally:
        print("\nDisabling motors...")
        robot.disable_motors()
        if cap is not None:
            cap.release()

def test_error_cases():
    """Test error handling in the skill library"""
    print("\n=== Starting Error Case Tests ===")
    skill_lib = SkillLibrary()
    robot = Robot()
    
    try:
        robot.initialize()
        
        # Test invalid skill name
        print("\n1. Testing invalid skill name...")
        try:
            skill_lib.execute_skill("nonexistent_skill", RobotPlatform.ZEROTH, robot)
        except ValueError as e:
            print(f"Successfully caught error: {e}")
            
        # Test execution with no objective
        print("\n2. Testing execution with no objective...")
        try:
            skill_lib.execute_skill("stand", RobotPlatform.ZEROTH, robot)
            print("Note: Skills can execute with default behavior when no objective specified")
        except Exception as e:
            print(f"Got expected warning about missing objective")
            
        print("\n=== Error case tests completed! ===")
        
    except Exception as e:
        print(f"\nUnexpected error during test: {str(e)}")
        raise
    finally:
        robot.disable_motors()

def parse_args():
    parser = argparse.ArgumentParser(description='Test Skill Library')
    parser.add_argument('--dummy', 
                       action='store_true',
                       help='Use dummy black images instead of webcam')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    print("=== Skill Library Test Suite ===")
    print("This test suite verifies the functionality of the robot skill library")
    print("It uses the actual robot hardware and LLM-based state verification")
    
    try:
        # Run main functionality tests
        test_skill_library(dummy_mode=args.dummy)
        
        # Run error case tests
        test_error_cases()
        
        print("\n=== All tests completed successfully! ===")
        
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
    except Exception as e:
        print(f"\nTests failed with error: {str(e)}")
        raise

