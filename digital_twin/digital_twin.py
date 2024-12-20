import mujoco
import numpy as np
import time
from mujoco import viewer
from openlch.hal import HAL
from pathlib import Path
import select
import sys
import math

class RobotDigitalTwin:
    def __init__(self, model_path, mode="robot_to_sim"):
        # Initialize with different models based on mode
        if mode == "sim_to_robot":
            self.model = mujoco.MjModel.from_xml_path("robot_upper.xml")
        else:
            self.model = mujoco.MjModel.from_xml_path(model_path)
        
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)
        
        # Initialize physical robot connection
        self.robot = HAL()
        
        # Initialize visualization
        self.viewer = viewer.launch_passive(model=self.model, data=self.data)
        
        # Store the mapping between robot servo IDs and MuJoCo joint indices
        self.servo_to_joint_mapping = self._create_servo_mapping()
        
        # Control mode flag -- sim to robot moves robot utilizing movement in mujoco viewer
        self.mode = mode  # "sim_to_robot" or "robot_to_sim" or "playback"
        self.recording = False
        self.recording_type = None  # "continuous" or "keyframe"
        self.current_episode = []
        self.episodes_dir = Path("episodes")
        self.keyframe_episodes_dir = Path("keyframe_episodes")
        self.episodes_dir.mkdir(exist_ok=True)
        self.keyframe_episodes_dir.mkdir(exist_ok=True)

        print(f"Control mode: {self.mode}")

        if self.mode == "sim_to_robot":
            self.torque_enable = True
            self.torque_value = 80
        else:
            self.torque_enable = False
            self.torque_value = 6
          
        self._setup_servos()
        
        self.joint_offsets = {
            # Left Leg
            "left_hip_pitch": 0.23 * 180/math.pi + 5.0,
            "left_hip_yaw": 45.0,
            "left_hip_roll": 0.0,
            "left_knee_pitch": -0.741 * 180/math.pi,
            "left_ankle_pitch": -0.5 * 180/math.pi,
            # Right Leg
            "right_hip_pitch": -0.23 * 180/math.pi - 5.0,
            "right_hip_yaw": -45.0,
            "right_hip_roll": 0.0,
            "right_knee_pitch": 0.741 * 180/math.pi,
            "right_ankle_pitch": 0.5 * 180/math.pi,
            # Arms
            "right_elbow_yaw": 0.0,
            "right_shoulder_yaw": 0.0,
            "right_shoulder_pitch": 0.0,
            "left_shoulder_pitch": 0.0,
            "left_shoulder_yaw": 0.0,
            "left_elbow_yaw": 0.0
        }
        
        
    def _create_servo_mapping(self):
        # Create a mapping between robot servo IDs and MuJoCo joint indices
        joint_mapping = {
            10: "left_hip_pitch",
            9: "left_hip_yaw",
            8: "left_hip_roll",
            7: "left_knee_pitch",
            6: "left_ankle_pitch",
            5: "right_hip_pitch",
            4: "right_hip_yaw",
            3: "right_hip_roll",
            2: "right_knee_pitch",
            1: "right_ankle_pitch",
            11: "right_elbow_yaw",
            12: "right_shoulder_yaw",
            13: "right_shoulder_pitch",
            14: "left_shoulder_pitch",
            15: "left_shoulder_yaw",
            16: "left_elbow_yaw"
        }
        
        mapping = {}
        for servo_id, joint_name in joint_mapping.items():
            try:
                joint_id = self.model.joint(joint_name).id
                mapping[servo_id] = joint_id
            except KeyError:
                print(f"Warning: Joint {joint_name} not found in MuJoCo model")
        
        return mapping
    

    def _setup_servos(self) -> None:
        """Sets up servo parameters for all mapped servos."""
        # Get servo IDs from our mapping
        servo_ids = list(self.servo_to_joint_mapping.keys())
        
        # Scan for available servos
        available_servos = self.robot.servo.scan()
        print(f"Available servos: {available_servos}")
        
        # Enable torque for all mapped servos
        self.robot.servo.set_torque_enable(
            [(servo_id, self.torque_enable) for servo_id in servo_ids]
        )
        
        # Set torque value for all mapped 6
        # servos
        self.robot.servo.set_torque(
            [(servo_id, self.torque_value) for servo_id in servo_ids]
        )
        
        # Enable movement
        self.robot.servo.enable_movement()

    def real_to_sim_position(self, real_position, joint_name):
        offsets = self.joint_offsets  # Use the class attribute
        return real_position - offsets[joint_name]

    def sim_to_real_position(self, sim_position, joint_name):
        if joint_name not in self.joint_offsets:
            raise ValueError(f"Unknown joint name: {joint_name}")
        
        return sim_position + self.joint_offsets[joint_name]
    
    def sync_robot_to_sim(self):
        positions = self.robot.servo.get_positions()
        
        for servo_id, pos, _ in positions:
            if servo_id in self.servo_to_joint_mapping:
                joint_id = self.servo_to_joint_mapping[servo_id]
                joint_name = self.model.joint(joint_id).name
                sim_pos = self.real_to_sim_position(pos, joint_name)
                self.data.qpos[joint_id] = np.radians(sim_pos)

    def calibrate_joint_offsets(self):
        """Calibrate joint offsets by comparing sim and real positions"""
        print("Starting joint calibration...")
        new_offsets = {}
        
        # Get current robot positions
        positions = self.robot.servo.get_positions()
        
        for servo_id, real_pos, _ in positions:
            if servo_id in self.servo_to_joint_mapping:
                joint_id = self.servo_to_joint_mapping[servo_id]
                joint_name = self.model.joint(joint_id).name
                sim_pos = self.data.qpos[joint_id]
                new_offsets[joint_name] = real_pos - sim_pos
        
        self.joint_offsets = new_offsets
        print("Calibration complete")

    def _degrees_to_radians(self, degrees):
        return degrees * math.pi / 180.0

    def _radians_to_degrees(self, radians):
        return radians * 180.0 / math.pi
    
    def sync_sim_to_robot(self):
        """Update physical robot state based on MuJoCo simulation, only for upper body"""
        # Define upper body servo IDs
        upper_body_servos = {
            11: "right_elbow_yaw",
            12: "right_shoulder_yaw",
            13: "right_shoulder_pitch",
            14: "left_shoulder_pitch",
            15: "left_shoulder_yaw",
            16: "left_elbow_yaw"
        }
        
        # Get positions from MuJoCo only for upper body
        positions_to_set = []
        for servo_id, joint_name in upper_body_servos.items():
            if servo_id in self.servo_to_joint_mapping:
                joint_id = self.servo_to_joint_mapping[servo_id]
                # Convert radians to degrees and apply offset
                sim_pos = np.degrees(self.data.qpos[joint_id])
                pos = self.sim_to_real_position(sim_pos, joint_name)
                positions_to_set.append((servo_id, pos))
        
        # Update robot positions for upper body only
        if positions_to_set:
            self.robot.servo.set_positions(positions_to_set)
    
    def start_recording(self):
        """Start recording an episode"""
        episode_name = input("Enter episode name to record: ")
        rec_type = input("Enter recording type (1 for continuous, 2 for keyframe): ")
        
        self.recording = True
        self.recording_type = "continuous" if rec_type == "1" else "keyframe"
        self.current_episode = []
        
        print(f"Recording episode: {episode_name} ({self.recording_type} mode)")
        if self.recording_type == "keyframe":
            print("Press Enter to save keyframes, R to stop recording")
        
        return episode_name

    def stop_recording(self, episode_name):
        """Stop recording and save the episode"""
        self.recording = False
        
        # Choose directory based on recording type
        save_dir = self.keyframe_episodes_dir if self.recording_type == "keyframe" else self.episodes_dir
        episode_path = save_dir / f"{episode_name}.txt"
        
        with open(episode_path, "w") as f:
            for positions in self.current_episode:
                position_str = ";".join([f"{id},{pos}" for id, pos in positions])
                f.write(f"{position_str}\n")
        
        print(f"Episode saved to {episode_path}")
        self.recording_type = None

    def record_frame(self):
        """Record current joint positions"""
        positions = self.robot.servo.get_positions()
        frame_positions = [(servo_id, pos) for servo_id, pos, _ in positions 
                          if servo_id in self.servo_to_joint_mapping]
        self.current_episode.append(frame_positions)

    def get_available_episodes(self, recording_type="continuous"):
        """Get list of recorded episodes"""
        directory = self.keyframe_episodes_dir if recording_type == "keyframe" else self.episodes_dir
        return [f.stem for f in directory.glob("*.txt")]

    def load_episode(self, episode_name, recording_type="continuous"):
        """Load an episode from file"""
        directory = self.keyframe_episodes_dir if recording_type == "keyframe" else self.episodes_dir
        episode_path = directory / f"{episode_name}.txt"
        frames = []
        with open(episode_path, "r") as f:
            for line in f:
                frame_positions = []
                position_pairs = line.strip().split(";")
                for pair in position_pairs:
                    servo_id, pos = pair.split(",")
                    frame_positions.append((int(servo_id), float(pos)))
                frames.append(frame_positions)
        return frames

    def move_to_default_position(self):
        """Move robot to default position using original offsets"""
        default_positions = []
        for servo_id, joint_id in self.servo_to_joint_mapping.items():
            joint_name = self.model.joint(joint_id).name
            default_pos = self.joint_offsets[joint_name]
            default_positions.append((servo_id, default_pos))
        
        self.robot.servo.set_positions(default_positions)
        time.sleep(1)  # Give time for robot to reach position

    def set_mode(self, mode):
        """Set the control mode and adjust torques accordingly"""
        if mode not in ["sim_to_robot", "robot_to_sim", "robot_to_sim_upper_free", "robot_to_sim_all_free", "playback"]:
            raise ValueError("Invalid mode. Choose from available modes.")
        
        # Handle model switching for sim_to_robot mode
        if mode == "sim_to_robot" and self.mode != "sim_to_robot":
            self.viewer.close()
            time.sleep(0.5)
            self.model = mujoco.MjModel.from_xml_path("robot_upper.xml")
            self.data = mujoco.MjData(self.model)
            self.viewer = viewer.launch_passive(model=self.model, data=self.data)
        elif mode != "sim_to_robot" and self.mode == "sim_to_robot":
            self.viewer.close()
            time.sleep(0.5)
            self.model = mujoco.MjModel.from_xml_path("robot.xml")
            self.data = mujoco.MjData(self.model)
            self.viewer = viewer.launch_passive(model=self.model, data=self.data)
        
        self.mode = mode
        print(f"Switched to {mode} mode")
        
        # Define upper body servo IDs
        upper_body_servos = {11, 12, 13, 14, 15, 16}  # arm servo IDs
        
        # Set torques based on mode
        if mode == "sim_to_robot":
            self.torque_value = 80
            torque_settings = [(servo_id, self.torque_value) for servo_id in self.servo_to_joint_mapping.keys()]
            print("Sim-to-robot mode: Only upper body (arms) will follow simulation movements")
        elif mode == "robot_to_sim":
            self.torque_value = 6
            torque_settings = [(servo_id, self.torque_value) for servo_id in self.servo_to_joint_mapping.keys()]
        elif mode == "robot_to_sim_upper_free":
            # Set upper body torques to 0 and lower body to 6
            torque_settings = []
            for servo_id in self.servo_to_joint_mapping.keys():
                if servo_id in upper_body_servos:
                    torque_settings.append((servo_id, 0))
                else:
                    torque_settings.append((servo_id, 6))
            print("Robot-to-sim mode: Upper body free for teleoperation, lower body stabilized")
        elif mode == "robot_to_sim_all_free":
            # Set all torques to 0 for full teleoperation
            torque_settings = [(servo_id, 0) for servo_id in self.servo_to_joint_mapping.keys()]
            print("Robot-to-sim mode: All joints free for teleoperation")
        elif mode == "playback":
            # Set high torques for playback
            self.torque_value = 10
            torque_settings = [(servo_id, self.torque_value) for servo_id in self.servo_to_joint_mapping.keys()]
            self.robot.servo.set_torque(torque_settings)
            self.robot.servo.enable_movement()
            
            episodes = self.get_available_episodes()
            if not episodes:
                print("No episodes available")
                return
            
            print("Available episodes:")
            for i, episode in enumerate(episodes):
                print(f"{i+1}. {episode}")
            
            choice = input("Select episode number to play: ")
            try:
                episode_name = episodes[int(choice)-1]
                self.current_episode = self.load_episode(episode_name)
                print(f"Loaded episode: {episode_name}")
                self.move_to_default_position()
            except (ValueError, IndexError):
                print("Invalid selection")
                return
        
        self.robot.servo.set_torque(torque_settings)
        self.robot.servo.enable_movement()

    def set_zero_position(self):
        """Move all servos to zero position"""

        # set torques to 50 to zero
        self.robot.servo.set_torque([(servo_id, 50) for servo_id in self.servo_to_joint_mapping.keys()])

        zero_positions = []
        for servo_id in self.servo_to_joint_mapping.keys():
            zero_positions.append((servo_id, 0.0 + self.joint_offsets[self.model.joint(self.servo_to_joint_mapping[servo_id]).name]))
        
        print("Moving to zero position...")
        self.robot.servo.set_positions(zero_positions)
        
        # Also set simulation joints to zero
        for joint_id in range(self.model.nq):
            self.data.qpos[joint_id] = 0.0
        
        time.sleep(1)  # Give time for robot to reach position

    def run_digital_twin(self):
        """Main loop for running the digital twin"""
        try:
            print("Starting digital twin...")
            print("Press 1 for sim-to-robot mode")
            print("Press 2 for robot-to-sim mode (high torques)")
            print("Press 3 for robot-to-sim mode (upper body free)")
            print("Press 4 for robot-to-sim mode (all joints free)")
            print("Press 5 for playback mode")
            print("Press 6 to reset to zero position")
            print("Press R to start/stop recording (in robot-to-sim modes)")
            print("Press Enter during keyframe recording to save a keyframe")
            print("Press Ctrl+C to exit")
            
            episode_name = None
            frame_index = 0
            
            while True:
                if select.select([sys.stdin], [], [], 0)[0]:
                    command = sys.stdin.readline().strip().lower()
                    if command == '1':
                        self.set_mode("sim_to_robot")
                    elif command == '2':
                        self.set_mode("robot_to_sim")
                    elif command == '3':
                        self.set_mode("robot_to_sim_upper_free")
                    elif command == '4':
                        self.set_mode("robot_to_sim_all_free")
                    elif command == '5':
                        # Modified playback mode selection
                        print("\nSelect playback type:")
                        print("1. Continuous recording")
                        print("2. Keyframe recording")
                        playback_type = input("Enter choice (1/2): ")
                        
                        recording_type = "continuous" if playback_type == "1" else "keyframe"
                        episodes = self.get_available_episodes(recording_type)
                        
                        if not episodes:
                            print(f"No {recording_type} episodes available")
                            continue
                        
                        print("\nAvailable episodes:")
                        for i, episode in enumerate(episodes):
                            print(f"{i+1}. {episode}")
                        
                        choice = input("Select episode number to play: ")
                        try:
                            episode_name = episodes[int(choice)-1]
                            self.current_episode = self.load_episode(episode_name, recording_type)
                            print(f"Loaded episode: {episode_name}")
                            
                            # Set torques once at the start of playback
                            self.robot.servo.set_torque([(servo_id, 10) for servo_id in self.servo_to_joint_mapping.keys()])
                            self.robot.servo.set_positions([(13, 15), (14, 15)])
                            
                            self.mode = "playback"
                            self.recording_type = recording_type
                            frame_index = 0
                            self.move_to_default_position()
                        except (ValueError, IndexError):
                            print("Invalid selection")
                            continue
                    elif command == '6':
                        self.set_zero_position()
                    elif command == 'r' and self.mode in ["robot_to_sim", "robot_to_sim_upper_free", "robot_to_sim_all_free"]:
                        if not self.recording:
                            episode_name = self.start_recording()
                        else:
                            self.stop_recording(episode_name)
                    elif self.recording and self.recording_type == "keyframe":
                        # Save keyframe when Enter is pressed during keyframe recording
                        positions = self.robot.servo.get_positions()
                        frame_positions = [(servo_id, pos) for servo_id, pos, _ in positions 
                                        if servo_id in self.servo_to_joint_mapping]
                        self.current_episode.append(frame_positions)
                        print("Keyframe saved!")
                    elif command == 'q':
                        break
                
                if self.mode == "sim_to_robot":
                    mujoco.mj_step(self.model, self.data)
                    self.sync_sim_to_robot()
                elif self.mode in ["robot_to_sim", "robot_to_sim_upper_free", "robot_to_sim_all_free"]:
                    self.sync_robot_to_sim()
                    mujoco.mj_step(self.model, self.data)
                    if self.recording and self.recording_type == "continuous":
                        self.record_frame()
                elif self.mode == "playback":
                    if frame_index < len(self.current_episode):
                        positions = self.current_episode[frame_index]
                        self.robot.servo.set_positions(positions)
                        frame_index += 1
                        # Add delay between keyframes for smoother movement
                        if self.recording_type == "keyframe":
                            time.sleep(2.0)  # Adjust this value as needed
                    else:
                        print("Episode playback complete")
                        self.mode = "robot_to_sim"
                
                self.viewer.sync()
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("Shutting down digital twin...")
        finally:
            self.viewer.close()
            self.robot.servo.disable_movement()

def main():
    # Path to your MuJoCo XML model file
    model_path = "robot.xml"
    
    # Create and run digital twin
    digital_twin = RobotDigitalTwin(model_path)
    digital_twin.run_digital_twin()

if __name__ == "__main__":
    main()