"""Model controller for running ONNX models on the robot."""

import argparse
import math
import time
from collections import deque
import numpy as np
import onnxruntime as ort
import threading
from typing import Dict, Optional
import yaml

class ModelController:
    def __init__(self, model_path: str, config_path: str = "param.yaml"):
        """Initialize the model controller.
        
        Args:
            model_path: Path to the ONNX model file
            config_path: Path to parameter configuration file
        """
        self.policy = ort.InferenceSession(model_path)
        self.cfg = self._init_config()
        
        # Load parameters
        with open(config_path, 'r') as f:
            self.params = yaml.safe_load(f)
            
        self._print_config()
        
    def _init_config(self):
        """Initialize configuration similar to Sim2simCfg."""
        class Config:
            def __init__(self):
                self.num_actions = 10
                self.frame_stack = 15
                self.c_frame_stack = 3
                self.sim_duration = 60.0
                self.dt = 0.001
                self.decimation = 10
                self.cycle_time = 0.4
                self.tau_factor = 3
                self.lin_vel = 2.0
                self.ang_vel = 1.0
                self.dof_pos = 1.0
                self.dof_vel = 0.05
                self.clip_observations = 18.0
                self.clip_actions = 18.0
                self.action_scale = 0.25
                
                # Computed attributes
                self.num_single_obs = 11 + self.num_actions * self.c_frame_stack
                self.num_observations = int(self.frame_stack * self.num_single_obs)
        
        return Config()

    def _print_config(self):
        """Print model configuration parameters."""
        print("\nModel inference configuration parameters:\n")
        print("{:<25} {:<15}".format("Parameter", "Value"))
        print("-" * 40)
        for attr, value in vars(self.cfg).items():
            if isinstance(value, np.ndarray):
                value = value.tolist()
            print("{:<25} {:<15}".format(attr, str(value)))
        print()

    def run_inference(self, robot, stop_event: threading.Event, cmd_vx=0.4, cmd_vy=0.0, cmd_dyaw=0.0):
        """Run model inference loop.
        
        Args:
            robot: Robot instance
            stop_event: Threading event to stop inference
            cmd_vx: Forward velocity command
            cmd_vy: Lateral velocity command 
            cmd_dyaw: Yaw rate command
        """
        action = np.zeros((self.cfg.num_actions), dtype=np.double)
        
        # Load hip adjustment from params
        with open("param.yaml", 'r') as f:
            params = yaml.safe_load(f)
        hip_adjust = params['robot']['servos'].get('hip_adjust', 20.0)  # Default to 20.0 if not found
        
        print(f"Using hip adjustment of {hip_adjust} degrees for walking stability")

        hist_obs = deque()
        for _ in range(self.cfg.frame_stack):
            hist_obs.append(np.zeros([1, self.cfg.num_single_obs], dtype=np.double))

        target_frequency = 1 / (self.cfg.dt * self.cfg.decimation)
        target_loop_time = 1.0 / target_frequency

        last_time = time.time()
        t_start = time.time()
        t = 0.0

        joint_names = [joint.name for joint in robot.joints]

        while not stop_event.is_set():
            loop_start_time = time.time()
            t = time.time() - t_start

            current_time = time.time()
            cycle_time = current_time - last_time
            last_time = current_time

            # Get robot state
            feedback_positions = robot.get_feedback_positions()
            feedback_velocities = robot.get_feedback_velocities()

            # Add hip adjustment for stability
            feedback_positions["left_hip_pitch"] += hip_adjust
            feedback_positions["right_hip_pitch"] -= hip_adjust

            # Convert to radians
            current_positions_np = np.radians(
                np.array([feedback_positions[name] for name in joint_names], dtype=np.float32)
            )
            current_velocities_np = np.radians(
                np.array([feedback_velocities[name] for name in joint_names], dtype=np.float32)
            )

            # Use leg joints for policy input
            positions_leg = current_positions_np[: self.cfg.num_actions]
            velocities_leg = current_velocities_np[: self.cfg.num_actions]

            # Get IMU data (will return mock values if IMU is disabled)
            imu_data = robot.get_imu_data()
            
            # Convert gyro from deg/s to rad/s
            omega = np.array([
                imu_data['gyro']['x'],
                imu_data['gyro']['y'],
                imu_data['gyro']['z']
            ], dtype=np.float32) * np.pi / 180.0

            # Convert accelerometer from mg to g
            accel = np.array([
                imu_data['accel']['x'],
                imu_data['accel']['y'],
                imu_data['accel']['z']
            ], dtype=np.float32) / 1000.0  # mg to g
            
            # Calculate euler angles
            eu_ang = np.array([
                np.arctan2(accel[1], accel[2]),  # roll
                np.arctan2(-accel[0], np.sqrt(accel[1]**2 + accel[2]**2)),  # pitch
                0.0  # yaw (cannot be determined from accelerometer alone)
            ], dtype=np.float32)

            # Prepare observation
            obs = np.zeros([1, self.cfg.num_single_obs], dtype=np.float32)
            obs[0, 0] = math.sin(2 * math.pi * t / self.cfg.cycle_time)
            obs[0, 1] = math.cos(2 * math.pi * t / self.cfg.cycle_time)
            obs[0, 2] = cmd_vx * self.cfg.lin_vel
            obs[0, 3] = cmd_vy * self.cfg.lin_vel
            obs[0, 4] = cmd_dyaw * self.cfg.ang_vel
            obs[0, 5 : self.cfg.num_actions + 5] = positions_leg * self.cfg.dof_pos
            obs[0, self.cfg.num_actions + 5 : 2 * self.cfg.num_actions + 5] = velocities_leg * self.cfg.dof_vel
            obs[0, 2 * self.cfg.num_actions + 5 : 3 * self.cfg.num_actions + 5] = action
            obs[0, 3 * self.cfg.num_actions + 5 : 3 * self.cfg.num_actions + 5 + 3] = omega
            obs[0, 3 * self.cfg.num_actions + 5 + 3 : 3 * self.cfg.num_actions + 5 + 6] = eu_ang
            obs = np.clip(obs, -self.cfg.clip_observations, self.cfg.clip_observations)

            hist_obs.append(obs)
            hist_obs.popleft()

            # Prepare policy input
            policy_input = np.zeros([1, self.cfg.num_observations], dtype=np.float32)
            for i in range(self.cfg.frame_stack):
                start = i * self.cfg.num_single_obs
                end = (i + 1) * self.cfg.num_single_obs
                policy_input[0, start:end] = hist_obs[i][0, :]

            # Run inference
            ort_inputs = {self.policy.get_inputs()[0].name: policy_input}
            action[:] = self.policy.run(None, ort_inputs)[0][0]

            action = np.clip(action, -self.cfg.clip_actions, self.cfg.clip_actions)
            scaled_action = action * self.cfg.action_scale

            # Prepare full action
            full_action = np.zeros(len(robot.joints), dtype=np.float32)
            full_action[: self.cfg.num_actions] = scaled_action

            # Convert to degrees and send to robot
            full_action_deg = np.degrees(full_action)
            desired_positions_dict = {
                name: position for name, position in zip(joint_names, full_action_deg)
            }
            robot.set_desired_positions(desired_positions_dict)

            # Timing control
            loop_end_time = time.time()
            loop_duration = loop_end_time - loop_start_time
            sleep_time = max(0, target_loop_time - loop_duration)
            time.sleep(sleep_time)

        print("[INFO]: Model inference stopped.")
