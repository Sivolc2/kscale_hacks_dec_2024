from enum import Enum
from dataclasses import dataclass
from typing import Dict, Callable, Optional, List, Any, Union
import numpy as np
import threading
from robot import Robot, RobotConfig
from model_controller import ModelController
from model_handler import ModelHandler
import os
import time
import base64
import cv2
import yaml

class RobotPlatform(Enum):
    ZEROTH = "zeroth"
    GENERIC = "generic"
    # Add other platforms as needed

@dataclass
class RobotCommand:
    """Represents a specific robot command implementation"""
    platform: RobotPlatform
    command_fn: Callable
    params: Dict[str, Any] = None

# Define default joint positions with correct mapping and values
RESET_POSITIONS = {
    # Left Leg
    "left_hip_pitch": 28.4765625,      # ID 10
    "left_hip_yaw": 42.71484375,       # ID 9
    "left_hip_roll": 0.087890625,      # ID 8
    "left_knee_pitch": -41.66015625,   # ID 7
    "left_ankle_pitch": -17.578125,    # ID 6
    
    # Right Leg
    "right_hip_pitch": -28.65234375,   # ID 5
    "right_hip_yaw": -42.451171875,    # ID 4
    "right_hip_roll": 0.0,             # ID 3
    "right_knee_pitch": 41.572265625,  # ID 2
    "right_ankle_pitch": 17.9296875,   # ID 1
    
    # Arms
    "right_elbow_yaw": 0.0,            # ID 11
    "right_shoulder_yaw": 0.0,         # ID 12
    "right_shoulder_pitch": 0.17578125, # ID 13
    "left_shoulder_pitch": -0.17578125, # ID 14
    "left_shoulder_yaw": 0.0,          # ID 15
    "left_elbow_yaw": 0.0              # ID 16
}

@dataclass 
class Skill:
    name: str
    description: str
    platform_commands: Dict[RobotPlatform, RobotCommand]
    objective: Optional[str] = None
    timeout_seconds: float = 30.0
    check_interval: float = 2.0
    requires_validation: bool = True
    affordance_fn: Optional[Callable[[], float]] = None
    
    def __post_init__(self):
        """Initialize after dataclass initialization"""
        self.model_handler = ModelHandler()
    
    def check_completion(self, robot: Robot, image: Optional[np.ndarray] = None, initial_image: Optional[np.ndarray] = None) -> bool:
        """Check if skill execution is complete using LLM verification
        
        Args:
            robot: Robot instance
            image: Current camera image of robot
            initial_image: Optional initial state image for comparison
            
        Returns:
            bool: True if target state is reached
        """
        if not self.objective:
            print(f"Warning: No objective defined for skill {self.name}")
            return True
            
        if image is None:
            print("Warning: No image provided for state verification")
            return False
            
        # Convert current image to base64
        success, buffer = cv2.imencode('.jpg', image)
        if not success:
            print("Error encoding current image")
            return False
        current_image_b64 = base64.b64encode(buffer).decode('utf-8')
        
        try:
            print(f"\nChecking completion for skill '{self.name}'...")
            print(f"Objective: {self.objective}")
            
            if initial_image is not None:
                # Convert initial image to base64
                success, buffer = cv2.imencode('.jpg', initial_image)
                if not success:
                    print("Error encoding initial image")
                    return False
                initial_image_b64 = base64.b64encode(buffer).decode('utf-8')
                
                # Use action agent with both images
                decision = self.model_handler.action_agent_compare(
                    self.name,
                    self.objective,
                    initial_image_b64,
                    current_image_b64,
                    previous_attempts=1
                )
            else:
                # Use original single image check
                decision = self.model_handler.action_agent(
                    self.name,
                    self.objective,
                    current_image_b64,
                    previous_attempts=1
                )
            
            # Print decision for visibility
            print("\nAction agent decision:")
            print(f"Continue: {'Yes' if decision['continue_execution'] else 'No'}")
            print(f"Reason: {decision['reason']}")
            print("-" * 50)
            
            # If agent says don't continue, we've achieved the objective
            return not decision["continue_execution"]
            
        except Exception as e:
            print(f"Error during state verification: {e}")
            return False
    
    def execute(self, platform: RobotPlatform, robot: Any, **kwargs):
        """Execute the skill on the specified platform with LLM completion monitoring"""
        if platform not in self.platform_commands:
            raise ValueError(f"Platform {platform} not supported for skill {self.name}")
        
        cmd = self.platform_commands[platform]
        if cmd.params:
            kwargs.update(cmd.params)
            
        # Capture initial state image if robot has camera
        initial_image = robot.get_camera_image() if hasattr(robot, 'get_camera_image') else None
        
        # Create stop event for interruptible skills
        stop_event = threading.Event()
        
        # Start skill execution in separate thread
        execution_thread = threading.Thread(
            target=cmd.command_fn,
            args=(robot,),
            kwargs={"stop_event": stop_event, **kwargs}
        )
        execution_thread.start()
        
        start_time = time.time()
        success = False
        check_count = 0
        
        try:
            # For skills that don't need validation, wait one cycle then return
            if not self.requires_validation:
                time.sleep(1.0)  # Wait for one cycle
                success = True
                print(f"Skill '{self.name}' completed (no validation required)")
            else:
                # Give the skill time to complete one cycle before first check
                time.sleep(self.check_interval)
                
                # Monitor completion with reduced frequency
                while time.time() - start_time < self.timeout_seconds:
                    check_count += 1
                    print(f"\nCheck #{check_count} at {time.time() - start_time:.1f}s:")
                    
                    current_image = robot.get_camera_image() if hasattr(robot, 'get_camera_image') else None
                    
                    if self.check_completion(robot, current_image, initial_image):
                        success = True
                        break
                        
                    time.sleep(self.check_interval)
                
            if not success:
                print(f"\n✗ Skill '{self.name}' timed out after {self.timeout_seconds} seconds")
                
        except Exception as e:
            print(f"\n✗ Error monitoring skill '{self.name}': {str(e)}")
            success = False
            
        finally:
            # Signal thread to stop and wait for it
            stop_event.set()
            execution_thread.join(timeout=2.0)
            
        return success

class SkillLibrary:
    def __init__(self, config_path: str = "param.yaml"):
        self.skills: Dict[str, Skill] = {}
        self.config_path = config_path
        # Initialize model controller with path to ONNX model
        self.model_controller = None
        try:
            # Use the same model path as run.py
            model_path = "./models/walking_micro.onnx"
            if not os.path.isfile(model_path):
                print(f"ERROR: Model file not found at absolute path: {os.path.abspath(model_path)}")
            else:
                print(f"Found model at: {os.path.abspath(model_path)}")
                self.model_controller = ModelController(model_path, config_path)
                print("Successfully loaded model controller")
        except Exception as e:
            print(f"ERROR: Could not load walking model: {str(e)}")
            print(f"Exception type: {type(e)}")
        self._initialize_basic_skills()
    
    def _initialize_basic_skills(self):
        """Initialize generic robot skills"""
        
        # Stand skill
        self.add_skill(Skill(
            name="stand",
            description="Make the robot stand in a stable position",
            platform_commands={
                RobotPlatform.ZEROTH: RobotCommand(
                    platform=RobotPlatform.ZEROTH,
                    command_fn=lambda robot, **kwargs: robot.set_desired_positions(RESET_POSITIONS)
                )
            },
            objective="Robot is standing upright with both feet flat on the ground",
            timeout_seconds=5.0,
            check_interval=2.0,
            requires_validation=True
        ))

        # Walking skills
        self.add_skill(Skill(
            name="walk_forward",
            description="Make the robot walk forward",
            platform_commands={
                RobotPlatform.ZEROTH: RobotCommand(
                    platform=RobotPlatform.ZEROTH,
                    command_fn=lambda robot, **kwargs: self._zeroth_walk(
                        robot,
                        kwargs.get('stop_event')
                    )
                )
            },
            objective="Robot has moved forward while maintaining stable walking gait",
            timeout_seconds=20.0,
            check_interval=5.0,
            requires_validation=True
        ))

        # Reset positions skill
        self.add_skill(Skill(
            name="reset_positions",
            description="Reset all joints to their default positions",
            platform_commands={
                RobotPlatform.ZEROTH: RobotCommand(
                    platform=RobotPlatform.ZEROTH,
                    command_fn=lambda robot, **kwargs: robot.set_desired_positions(RESET_POSITIONS)
                )
            },
            requires_validation=False  # No validation needed for reset
        ))

    def _zeroth_walk(self, robot: Robot, stop_event: threading.Event, cmd_vx=0.4):
        """Implementation of walking for Zeroth robot using ONNX model"""
        print(f"Walking with velocity: {cmd_vx}")
        print("Restoring original offsets")
        
        if self.model_controller is None:
            print("ERROR: Model controller is not initialized")
            print("Please ensure the ONNX model exists at ./models/walking_micro.onnx")
            print("Current working directory:", os.getcwd())
            raise RuntimeError("Walking model not initialized - missing ONNX file")
        
        # Get torque settings from params
        with open(self.config_path, 'r') as f:
            params = yaml.safe_load(f)
        
        # Get walking torque value
        walking_torque = params['robot']['servos'].get('walking_torque', 32.0)
        print(f"Setting walking torque to {walking_torque:.1f}")
        
        # Set walking torque
        servo_ids = [joint.servo_id for joint in robot.joints]
        robot.hal.servo.set_torque(
            [(servo_id, walking_torque) for servo_id in servo_ids]
        )
        
        # Restore walking offsets like in run.py
        robot.joint_dict["left_hip_pitch"].offset_deg = 0.23 * 180/3.14159
        robot.joint_dict["left_hip_yaw"].offset_deg = 45.0
        robot.joint_dict["left_hip_roll"].offset_deg = 0.0
        robot.joint_dict["left_knee_pitch"].offset_deg = -0.741 * 180/3.14159
        robot.joint_dict["left_ankle_pitch"].offset_deg = -0.5 * 180/3.14159
        robot.joint_dict["right_hip_pitch"].offset_deg = -0.23 * 180/3.14159
        robot.joint_dict["right_hip_yaw"].offset_deg = -45.0
        robot.joint_dict["right_hip_roll"].offset_deg = 0.0
        robot.joint_dict["right_knee_pitch"].offset_deg = 0.741 * 180/3.14159
        robot.joint_dict["right_ankle_pitch"].offset_deg = 0.5 * 180/3.14159
        robot.joint_dict["right_elbow_yaw"].offset_deg = 0.0
        robot.joint_dict["right_shoulder_yaw"].offset_deg = 0.0
        robot.joint_dict["right_shoulder_pitch"].offset_deg = 0.0
        robot.joint_dict["left_shoulder_pitch"].offset_deg = 0.0
        robot.joint_dict["left_shoulder_yaw"].offset_deg = 0.0
        robot.joint_dict["left_elbow_yaw"].offset_deg = 0.0

        if self.model_controller is None:
            raise RuntimeError("Walking model not initialized")
            
        try:
            # Run model inference for walking
            print(f"Starting model-based walking with velocity {cmd_vx}...")
            self.model_controller.run_inference(
                robot=robot,
                stop_event=stop_event,
                cmd_vx=cmd_vx,  # Use passed velocity
                cmd_vy=0.0,  # Lateral velocity 
                cmd_dyaw=0.0  # Yaw rate
            )
            print("Walking completed")
            
        except Exception as e:
            print(f"Error during model-based walking: {e}")
            raise

    def _zeroth_wave(self, robot: Robot):
        """Implementation of waving for Zeroth robot"""
        import time
        
        initial_positions = {
            "left_shoulder_yaw": 0.0,
            "left_shoulder_pitch": 0.0,
            "left_elbow_yaw": 0.0,
        }
        robot.set_desired_positions(initial_positions)
        time.sleep(0.5)

        wave_up_positions = {
            "left_shoulder_pitch": 0.0,
            "left_shoulder_yaw": 150.0,
        }
        robot.set_desired_positions(wave_up_positions)
        time.sleep(0.5)

        for _ in range(6):
            robot.set_desired_positions({"left_elbow_yaw": -90.0})
            time.sleep(0.3)
            robot.set_desired_positions({"left_elbow_yaw": -45.0})
            time.sleep(0.3)
            
        robot.set_desired_positions(initial_positions)
        time.sleep(0.5)

    def _zeroth_forward_recovery(self, robot: Robot):
        """Implementation of forward recovery for Zeroth robot"""
        
        # Reset joint offsets
        for joint in robot.joints:
            joint.offset_deg = 0.0
        robot.joint_dict["right_hip_yaw"].offset_deg = -45.0
        robot.joint_dict["left_hip_yaw"].offset_deg = 45.0
        
        # Initialize all joints to 0
        initial_positions = {joint.name: 0.0 for joint in robot.joints}
        robot.set_desired_positions(initial_positions)

        # Getting feet on the ground
        robot.set_desired_positions({
            "left_hip_pitch": 30.0,
            "right_hip_pitch": -30.0,
            "left_knee_pitch": 50.0,
            "right_knee_pitch": -50.0,
            "left_ankle_pitch": -30.0,
            "right_ankle_pitch": 30.0,
        })

        # ... (rest of the recovery sequence from run.py)
        # Copy the full sequence of positions from run.py's state_forward_recovery

    def add_skill(self, skill: Skill):
        """Add a new skill to the library"""
        self.skills[skill.name] = skill
    
    def get_skill_names(self) -> List[str]:
        """Get list of all skill names"""
        return list(self.skills.keys())
    
    def get_skill_descriptions(self) -> str:
        """Get formatted string of all skills for LLM prompt"""
        return "\n".join([
            f"- {skill.name}: {skill.description}"
            for skill in self.skills.values()
        ])

    def execute_skill(self, skill_name: str, platform: RobotPlatform, robot: Any, **kwargs):
        """Execute a skill on the specified platform"""
        if skill_name not in self.skills:
            raise ValueError(f"Skill {skill_name} not found")
        return self.skills[skill_name].execute(platform, robot, **kwargs)
