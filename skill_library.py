from enum import Enum
from dataclasses import dataclass
from typing import Dict, Callable, Optional, List, Any
import numpy as np
import threading
from robot import Robot, RobotConfig
from model_controller import ModelController
import os

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

@dataclass 
class Skill:
    name: str
    description: str
    platform_commands: Dict[RobotPlatform, RobotCommand]
    affordance_fn: Optional[Callable[[], float]] = None
    
    def get_affordance(self) -> float:
        """Returns probability of successful execution from current state"""
        if self.affordance_fn is None:
            return 0.5
        return self.affordance_fn()
    
    def execute(self, platform: RobotPlatform, robot: Any, **kwargs):
        """Execute the skill on the specified platform"""
        if platform not in self.platform_commands:
            raise ValueError(f"Platform {platform} not supported for skill {self.name}")
        
        cmd = self.platform_commands[platform]
        if cmd.params:
            kwargs.update(cmd.params)
        return cmd.command_fn(robot, **kwargs)

class SkillLibrary:
    def __init__(self):
        self.skills: Dict[str, Skill] = {}
        # Initialize model controller with path to ONNX model
        self.model_controller = None
        try:
            # Use the same model path as run.py
            model_path = "./models/walking_micro.onnx"
            if not os.path.isfile(model_path):
                print(f"ERROR: Model file not found at absolute path: {os.path.abspath(model_path)}")
            else:
                print(f"Found model at: {os.path.abspath(model_path)}")
                self.model_controller = ModelController(model_path)
                print("Successfully loaded model controller")
        except Exception as e:
            print(f"ERROR: Could not load walking model: {str(e)}")
            print(f"Exception type: {type(e)}")
        self._initialize_basic_skills()
    
    def _initialize_basic_skills(self):
        """Initialize generic robot skills"""
        # Basic movement skills
        self.add_skill(Skill(
            name="stand",
            description="Make the robot stand in a stable position",
            platform_commands={
                RobotPlatform.GENERIC: RobotCommand(
                    platform=RobotPlatform.GENERIC,
                    command_fn=lambda robot: None
                ),
                RobotPlatform.ZEROTH: RobotCommand(
                    platform=RobotPlatform.ZEROTH,
                    command_fn=lambda robot: robot.set_desired_positions({
                        "left_hip_pitch": 0.23 * 180/3.14159,
                        "left_knee_pitch": -0.741 * 180/3.14159,
                        "left_ankle_pitch": -0.5 * 180/3.14159,
                        "right_hip_pitch": -0.23 * 180/3.14159,
                        "right_knee_pitch": 0.741 * 180/3.14159,
                        "right_ankle_pitch": 0.5 * 180/3.14159,
                        "left_hip_yaw": 0.0,
                        "right_hip_yaw": 0.0,
                        "left_hip_roll": 0.0,
                        "right_hip_roll": 0.0,
                        "left_shoulder_pitch": 0.0,
                        "right_shoulder_pitch": 0.0,
                        "left_shoulder_yaw": 0.0,
                        "right_shoulder_yaw": 0.0,
                        "left_elbow_yaw": 0.0,
                        "right_elbow_yaw": 0.0
                    })
                )
            }
        ))

        self.add_skill(Skill(
            name="walk_forward",
            description="Make the robot walk forward",
            platform_commands={
                RobotPlatform.GENERIC: RobotCommand(
                    platform=RobotPlatform.GENERIC,
                    command_fn=lambda robot: None  # Generic placeholder
                ),
                RobotPlatform.ZEROTH: RobotCommand(
                    platform=RobotPlatform.ZEROTH,
                    command_fn=lambda robot, stop_event: self._zeroth_walk(robot, stop_event)
                )
            }
        ))

        self.add_skill(Skill(
            name="wave",
            description="Make the robot wave its arm",
            platform_commands={
                RobotPlatform.GENERIC: RobotCommand(
                    platform=RobotPlatform.GENERIC,
                    command_fn=lambda robot: None
                ),
                RobotPlatform.ZEROTH: RobotCommand(
                    platform=RobotPlatform.ZEROTH,
                    command_fn=lambda robot: self._zeroth_wave(robot)
                )
            }
        ))

        self.add_skill(Skill(
            name="recover_forward",
            description="Recover from a forward fall",
            platform_commands={
                RobotPlatform.GENERIC: RobotCommand(
                    platform=RobotPlatform.GENERIC,
                    command_fn=lambda robot: None
                ),
                RobotPlatform.ZEROTH: RobotCommand(
                    platform=RobotPlatform.ZEROTH,
                    command_fn=lambda robot: self._zeroth_forward_recovery(robot)
                )
            }
        ))

        # Add reset positions skill
        reset_positions = {
            "right_hip_yaw": 28.65234375,
            "right_hip_pitch": 42.451171875,
            "right_hip_roll": 0.0,
            "right_knee_pitch": -41.572265625,
            "right_ankle_pitch": -17.9296875,
            "left_hip_yaw": -42.71484375,
            "left_hip_pitch": -0.087890625,
            "left_hip_roll": 41.66015625,
            "left_knee_pitch": 17.578125,
            "left_ankle_pitch": 0.0,
            "right_shoulder_pitch": -0.17578125,
            "right_shoulder_yaw": 0.0,
            "left_shoulder_pitch": 0.17578125,
            "left_shoulder_yaw": 0.0,
            "right_elbow_yaw": 0.0,
            "left_elbow_yaw": 0.0
        }

        self.add_skill(Skill(
            name="reset_positions",
            description="Reset all joints to their default positions",
            platform_commands={
                RobotPlatform.GENERIC: RobotCommand(
                    platform=RobotPlatform.GENERIC,
                    command_fn=lambda robot: None  # Generic placeholder
                ),
                RobotPlatform.ZEROTH: RobotCommand(
                    platform=RobotPlatform.ZEROTH,
                    command_fn=lambda robot: robot.set_desired_positions(reset_positions)
                )
            }
        ))

    def _zeroth_walk(self, robot: Robot, stop_event: threading.Event):
        """Implementation of walking for Zeroth robot using ONNX model"""
        print("Walking")
        print("Restoring original offsets")
        
        if self.model_controller is None:
            print("ERROR: Model controller is not initialized")
            print("Please ensure the ONNX model exists at ./models/walking_micro.onnx")
            print("Current working directory:", os.getcwd())
            raise RuntimeError("Walking model not initialized - missing ONNX file")
        
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
            print("Starting model-based walking...")
            self.model_controller.run_inference(
                robot=robot,
                stop_event=stop_event,
                cmd_vx=0.4,  # Forward velocity
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
        import time
        
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

