"""
Module: robot.py
Simplified Robot class with computations in degrees and direct position/velocity methods.
"""

import subprocess
import logging
from typing import List, Dict, Tuple, Optional, Any
from openlch import HAL
import math
import numpy as np
import yaml
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Joint:
    """Represents a single joint of the robot, using degrees for positions and velocities."""

    def __init__(
        self, 
        name: str, 
        servo_id: int, 
        offset_deg: float = 0.0
    ):
        self.name: str = name
        self.servo_id: int = servo_id
        self.offset_deg: float = offset_deg  # Offset in degrees
        self.feedback_position: float = 0.0  # Feedback position in degrees
        self.feedback_velocity: float = 0.0  # Feedback velocity in degrees/s

class RobotConfig:
    """Configuration parameters for the Robot."""

    def __init__(self):
        self.joint_configs = [
            # LEGS
            {'name': "left_hip_pitch", 'servo_id': 10, 'offset_deg': 0.23 * 180/math.pi + 5.0},
            {'name': "left_hip_yaw", 'servo_id': 9, 'offset_deg': 45.0},
            {'name': "left_hip_roll", 'servo_id': 8, 'offset_deg': 0.0},
            {'name': "left_knee_pitch", 'servo_id': 7, 'offset_deg': -0.741 * 180/math.pi},
            {'name': "left_ankle_pitch", 'servo_id': 6, 'offset_deg': -0.5 * 180/math.pi},
            {'name': "right_hip_pitch", 'servo_id': 5, 'offset_deg': -0.23 * 180/math.pi - 5.0},
            {'name': "right_hip_yaw", 'servo_id': 4, 'offset_deg': -45.0},
            {'name': "right_hip_roll", 'servo_id': 3, 'offset_deg': 0.0},
            {'name': "right_knee_pitch", 'servo_id': 2, 'offset_deg': 0.741 * 180/math.pi},
            {'name': "right_ankle_pitch", 'servo_id': 1, 'offset_deg': 0.5 * 180/math.pi},
            # ARMS
            {'name': "right_elbow_yaw", 'servo_id': 11, 'offset_deg': 0.0},
            {'name': "right_shoulder_yaw", 'servo_id': 12, 'offset_deg': 0.0},
            {'name': "right_shoulder_pitch", 'servo_id': 13, 'offset_deg': 0.0},
            {'name': "left_shoulder_pitch", 'servo_id': 14, 'offset_deg': 0.0},
            {'name': "left_shoulder_yaw", 'servo_id': 15, 'offset_deg': 0.0},
            {'name': "left_elbow_yaw", 'servo_id': 16, 'offset_deg': 0.0},
        ]
        self.torque_enable = True
        self.torque_value = 20.0

class Robot:
    """Controls the robot's hardware and joint movements."""

    def __init__(self, config_path: str = "param.yaml"):
        self.hal = HAL()
        self.config = RobotConfig()
        self.joints: List[Joint] = [
            Joint(**joint_config) for joint_config in self.config.joint_configs
        ]
        self.joint_dict: Dict[str, Joint] = {joint.name: joint for joint in self.joints}
        
        # Load parameters
        self.params = self._load_params(config_path)
        
        # Initialize IMU if enabled
        if self.params['robot']['imu']['enabled']:
            self.imu = self.hal.imu
            logger.info("IMU enabled - using hardware IMU")
        else:
            self.imu = None
            logger.info("IMU disabled - using mock values")

    def _load_params(self, config_path: str) -> Dict:
        """Load parameters from yaml file."""
        try:
            with open(config_path, 'r') as f:
                params = yaml.safe_load(f)
            return params
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            # Return default configuration
            return {
                'robot': {
                    'imu': {
                        'enabled': True,
                        'mock_values': {
                            'gyro': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                            'accel': {'x': 0.0, 'y': 0.0, 'z': 0.0}
                        }
                    }
                }
            }

    def initialize(self) -> None:
        """Initializes the robot's hardware and joints."""
        logger.info("Robot initializing...")
        self.check_connection()
        self.setup_servos()
        self.set_initial_positions()
        logger.info("Robot initialized.")

    def check_connection(self) -> None:
        """Checks the connection to the robot."""
        logger.info("Checking connection to robot...")
        try:
            subprocess.run(
                ["ping", "-c", "1", "192.168.42.1"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
            logger.info("Successfully pinged robot.")
        except subprocess.CalledProcessError:
            logger.error("Could not ping robot at 192.168.42.1")
            raise ConnectionError("Robot connection failed.")

    def setup_servos(self) -> None:
        """Sets up servo parameters."""
        logger.info("Scanning servos...")
        servo_ids = [joint.servo_id for joint in self.joints]
        available_servos = self.hal.servo.scan()
        logger.debug(f"Available servos: {available_servos}")

        # Get torque settings from params
        torque_enable = self.params['robot']['servos'].get('torque_enable', True)
        torque_scale = self.params['robot']['servos'].get('torque_scale', 1.0)
        base_torque = self.params['robot']['servos'].get('default_torque', self.config.torque_value)
        
        # Calculate scaled torque
        scaled_torque = base_torque * torque_scale
        logger.info(f"Setting torque to {scaled_torque:.1f} ({torque_scale*100:.0f}% of {base_torque})")

        self.hal.servo.set_torque_enable(
            [(servo_id, torque_enable) for servo_id in servo_ids]
        )
        self.hal.servo.set_torque(
            [(servo_id, scaled_torque) for servo_id in servo_ids]
        )

        self.hal.servo.enable_movement()

    def set_initial_positions(self) -> None:
        """Sets initial positions for all joints to their default standing pose.
        
        Uses pre-calibrated reset positions that have been tested to provide
        a stable standing position for the robot.
        """
        logger.info("Setting initial standing positions...")
        
        initial_positions = {
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
        
        self.set_desired_positions(initial_positions)
        logger.info("Initial standing position set.")

    def get_feedback_positions(self) -> Dict[str, float]:
        """Gets feedback positions from the servos.

        Returns:
            A dictionary mapping joint names to their feedback positions in degrees.
        """
        servo_states = self.hal.servo.get_positions()  # [(id_, position_deg, velocity_deg_s), ...]
        # Build a mapping from servo_id to position
        servo_position_dict = {servo_id: position_deg for servo_id, position_deg, _ in servo_states}
        feedback_positions = {}
        for joint in self.joints:
            if joint.servo_id in servo_position_dict:
                position_deg = servo_position_dict[joint.servo_id] - joint.offset_deg
                joint.feedback_position = position_deg
                feedback_positions[joint.name] = position_deg
            else:
                logger.warning(f"Servo ID {joint.servo_id} not found in servo states.")
                feedback_positions[joint.name] = None  # Or handle as appropriate
        return feedback_positions

    def get_feedback_velocities(self) -> Dict[str, float]:
        """Gets feedback velocities from the servos.

        Returns:
            A dictionary mapping joint names to their feedback velocities in degrees/s.
        """
        servo_states = self.hal.servo.get_positions()  # [(id_, position_deg, velocity_deg_s), ...]
        # Build a mapping from servo_id to velocity
        servo_velocity_dict = {servo_id: velocity_deg_s for servo_id, _, velocity_deg_s in servo_states}
        feedback_velocities = {}
        for joint in self.joints:
            if joint.servo_id in servo_velocity_dict:
                velocity_deg_s = servo_velocity_dict[joint.servo_id]
                joint.feedback_velocity = velocity_deg_s
                feedback_velocities[joint.name] = velocity_deg_s
            else:
                logger.warning(f"Servo ID {joint.servo_id} not found in servo states.")
                feedback_velocities[joint.name] = None  # Or handle as appropriate
        return feedback_velocities

    def set_desired_positions(self, positions: Dict[str, float]) -> None:
        """Sets desired positions for specified joints directly to the servos.

        Args:
            positions: A dictionary mapping joint names to desired positions in degrees.
        """
        position_commands: List[Tuple[int, float]] = []
        for name, position in positions.items():
            if name in self.joint_dict:
                joint = self.joint_dict[name]
                desired_position = position + joint.offset_deg
                position_commands.append((joint.servo_id, desired_position))
            else:
                logger.error(f"Joint name '{name}' not found.")
                raise ValueError(f"Joint name '{name}' not found.")
        # Send positions to servos
        self.hal.servo.set_positions(position_commands)

    def disable_motors(self) -> None:
        logger.info("Disabling all motors.")
        self.hal.servo.disable_movement()
        servo_ids = [joint.servo_id for joint in self.joints]
        self.hal.servo.set_torque_enable(
            [(servo_id, False) for servo_id in servo_ids]
        )

    def get_camera_image(self) -> Optional[np.ndarray]:
        """Get current camera image if available"""
        if hasattr(self, 'monitor') and hasattr(self.monitor, 'cap'):
            ret, frame = self.monitor.cap.read()
            if ret:
                return frame
        return None

    def get_imu_data(self) -> Dict[str, Any]:
        """Get current IMU data with fallback to mock values.
        
        Returns:
            Dictionary containing gyroscope and accelerometer data
        """
        if not self.params['robot']['imu']['enabled']:
            return self.params['robot']['imu']['mock_values']
            
        try:
            if self.imu is not None:
                return self.imu.get_data()
            else:
                return self.params['robot']['imu']['mock_values']
        except Exception as e:
            logger.warning(f"IMU read failed: {e}. Using mock values.")
            return self.params['robot']['imu']['mock_values']

    def get_imu_orientation(self) -> Tuple[float, float, float]:
        """Get current IMU orientation (roll, pitch, yaw).
        
        Returns:
            Tuple of (roll, pitch, yaw) in degrees
        """
        data = self.get_imu_data()
        accel = data['accel']
        
        # Simple euler angle estimation from accelerometer
        # Note: This is a basic implementation - consider using Madgwick filter
        # for more accurate results
        roll = math.atan2(accel['y'], accel['z']) * 180.0 / math.pi
        pitch = math.atan2(-accel['x'], 
                          math.sqrt(accel['y']**2 + accel['z']**2)) * 180.0 / math.pi
        yaw = 0.0  # Can't determine yaw from accelerometer alone
        
        return roll, pitch, yaw
