import time
import cv2
from datetime import datetime
import os
import argparse
import yaml
from pathlib import Path
from colorama import init, Fore, Back, Style
from image_processor import ImageProcessor
from skill_library import SkillLibrary, RobotPlatform
from typing import List, Tuple, Optional, Dict
from robot import Robot
import threading
from model_handler import ModelHandler
from voice_processor import Command, CommandState, WhisperVoiceProcessor

# Initialize colorama
init(autoreset=True)

def print_status(msg: str):
    """Print status message in cyan"""
    print(f"{Fore.CYAN}{msg}{Style.RESET_ALL}")

def print_success(msg: str):
    """Print success message in green"""
    print(f"{Fore.GREEN}✓ {msg}{Style.RESET_ALL}")

def print_error(msg: str):
    """Print error message in red"""
    print(f"{Fore.RED}✗ {msg}{Style.RESET_ALL}")

def print_warning(msg: str):
    """Print warning message in yellow"""
    print(f"{Fore.YELLOW}⚠ {msg}{Style.RESET_ALL}")

def print_thinking(msg: str):
    """Print AI thinking message in magenta"""
    print(f"{Fore.MAGENTA}🤔 {msg}{Style.RESET_ALL}")

def print_action(msg: str):
    """Print robot action message in blue"""
    print(f"{Fore.BLUE}🤖 {msg}{Style.RESET_ALL}")

def print_header(msg: str):
    """Print section header in white on blue"""
    print(f"\n{Back.BLUE}{Fore.WHITE} {msg} {Style.RESET_ALL}")

def load_config(config_path: str = "config/config.yaml") -> Dict:
    """Load configuration from yaml file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Add monitor-specific defaults if not present
        if "monitor" not in config:
            config["monitor"] = {
                "capture_interval": 5.0,
                "save_dir": "pictures/webcam",
                "platform": "zeroth",
                "voice_mode": False,
                "whisper_model": "small"
            }
        return config
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        raise

class RobotMonitor:
    def __init__(self, config_path: str = "config/config.yaml", **kwargs):
        """Initialize the robot monitoring system"""
        # Load base configuration
        self.config = load_config(config_path)
        
        # Override with any provided kwargs
        monitor_config = self.config["monitor"]
        for key, value in kwargs.items():
            if value is not None:  # Only override if explicitly provided
                monitor_config[key] = value
        
        # Initialize components
        print_status("Initializing components...")
        self.model_handler = ModelHandler()
        self.capture_interval = monitor_config["capture_interval"]
        self.save_dir = monitor_config["save_dir"]
        self.frame_count = 0
        self.processor = ImageProcessor(config_path=config_path)
        self.skill_library = SkillLibrary()
        self.platform = RobotPlatform[monitor_config["platform"].upper()]
        self.robot: Optional[Robot] = None
        self.walk_stop_event = None
        self.walk_thread = None
        self.cap = None
        
        # Voice mode settings
        self.voice_mode = monitor_config["voice_mode"]
        self.voice_processor = None
        if self.voice_mode:
            try:
                self.voice_processor = WhisperVoiceProcessor(
                    model_name=monitor_config["whisper_model"]
                )
                print_success(f"Initialized Whisper voice processor with {monitor_config['whisper_model']} model")
            except Exception as e:
                print_error(f"Failed to initialize voice processor: {e}")
                self.voice_mode = False
        
        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        print_success("Monitor initialization complete")

    def initialize_robot(self) -> None:
        """Initialize the robot hardware"""
        try:
            self.robot = Robot()
            self.robot.monitor = self  # Give robot access to monitor for camera
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
        image_data, metadata, processed_frame = self.processor.process_frame(frame)
        
        # Save the processed image
        self.frame_count += 1
        save_path = os.path.join(self.save_dir, f"pic_{self.frame_count}.jpg")
        cv2.imwrite(save_path, processed_frame)
        
        # Extract width, height, and megapixels from metadata
        width = metadata["width"]
        height = metadata["height"]
        mp = metadata["megapixels"]
        
        return image_data, save_path, (width, height, mp)

    def process_frame_with_voice(self, image_data: str, voice_command: Optional[str] = None) -> str:
        """Process frame with optional voice command input"""
        try:
            print_thinking("Planning task sequence...")
            task_pairs = self.model_handler.plan_tasks(
                self.skill_library.get_skill_descriptions(),
                image_data, 
                voice_command
            )
            
            print_header("Planned Task Sequence")
            for skill_name, objective in task_pairs:
                print_status(f"- [{skill_name}]: {objective}")
            
            # Execute each skill in sequence
            for skill_name, objective in task_pairs:
                print_header(f"Executing {skill_name}")
                print_status(f"Objective: {objective}")
                
                attempts = 0
                while True:
                    attempts += 1
                    print_action(f"Attempt {attempts}")
                    
                    success = self.execute_skill(skill_name)
                    if not success:
                        print_error(f"Failed to execute skill: {skill_name}")
                        return f"Failed to execute skill: {skill_name}"
                    
                    # Get action agent's decision
                    print_thinking("Evaluating execution...")
                    decision = self.model_handler.action_agent(
                        skill_name,
                        objective,
                        image_data,
                        attempts
                    )
                    
                    print_header("Action Agent Decision")
                    print_status(f"Analysis: {decision['analysis']}")
                    print_status(f"Continue: {'Yes' if decision['continue_execution'] else 'No'}")
                    print_status(f"Reason: {decision['reason']}")
                    
                    if not decision["continue_execution"]:
                        print_success(f"Successfully completed {skill_name}")
                        break
            
            return "Completed all planned tasks successfully"
                
        except Exception as e:
            print_error(f"Error during execution: {e}")
            return f"Error: {str(e)}"

    def run_skills(self, skills: List[str]) -> Dict[str, bool]:
        """
        Run a sequence of skills and return their success status
        Returns dict mapping skill names to success boolean
        """
        results = {}
        for skill_name in skills:
            try:
                print_action(f"Executing skill: {skill_name}")
                
                # Handle walking skills with threading
                if skill_name in ["speed_walk", "slow_walk"]:
                    self.walk_stop_event = threading.Event()
                    self.walk_thread = threading.Thread(
                        target=lambda: self.skill_library.execute_skill(
                            skill_name,
                            RobotPlatform.ZEROTH,
                            self.robot,
                            stop_event=self.walk_stop_event
                        )
                    )
                    self.walk_thread.start()
                    
                    # Let it walk for configured duration
                    walk_duration = self.config.get('skills', {}).get('walk_duration', 5.0)
                    print_status(f"Walking for {walk_duration} seconds...")
                    time.sleep(walk_duration)
                    
                    print_status("Stopping walk...")
                    self.walk_stop_event.set()
                    self.walk_thread.join(timeout=2)
                    self.walk_stop_event = None
                    self.walk_thread = None
                else:
                    # Execute other skills normally
                    self.skill_library.execute_skill(
                        skill_name,
                        RobotPlatform.ZEROTH,
                        self.robot
                    )
                    
                results[skill_name] = True
                print_success(f"Successfully executed: {skill_name}")
            except Exception as e:
                print_error(f"Failed to execute {skill_name}: {e}")
                results[skill_name] = False
        return results

    def execute_skill(self, skill_name: str, **kwargs):
        """Execute a skill on the current platform"""
        if self.robot is None:
            raise RuntimeError("Robot not initialized. Call initialize_robot() first.")
        
        # Get current camera frame for verification
        ret, frame = self.cap.read()
        if ret:
            # Process the frame
            image_data, _, processed_frame = self.processor.process_frame(frame)
            kwargs['image'] = processed_frame  # Pass the processed frame to skill execution
        else:
            print("Warning: Failed to capture frame for skill verification")
        
        return self.skill_library.execute_skill(skill_name, self.platform, self.robot, **kwargs)

    def get_voice_command(self) -> Optional[Command]:
        """Get transcribed voice command from the voice processor"""
        if not self.voice_mode or not self.voice_processor:
            return None
        
        try:
            # Get next command if available
            command = self.voice_processor.get_next_command()
            if command:
                print(f"\nVoice command received: {command.text}")
                
                # Show recent context
                if command.context:
                    print("Recent conversation context:")
                    for text in command.context[-3:]:
                        print(f"  - {text}")
                    
                return command
        except Exception as e:
            print(f"Error processing voice command: {e}")
        return None

    def process_voice_command(self, command: Command) -> None:
        """Process and execute a voice command"""
        try:
            # Get planned tasks from model with retries
            max_retries = 3
            retry_delay = 1
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    task_pairs = self.model_handler.plan_tasks(
                        self.skill_library.get_skill_descriptions(),
                        voice_command=command.text
                    )
                    break
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        print(f"Attempt {attempt + 1} failed, retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        raise last_error
            
            if task_pairs:
                print("\nExecuting planned tasks:")
                for skill_name, objective in task_pairs:
                    print(f"- [{skill_name}]: {objective}")
                
                # Execute the skills
                results = self.run_skills([skill for skill, _ in task_pairs])
                
                # Update command state based on results
                if all(results.values()):
                    self.voice_processor.update_command_state(
                        command,
                        CommandState.COMPLETED,
                        f"Successfully executed all tasks"
                    )
                else:
                    failed_skills = [skill for skill, success in results.items() if not success]
                    self.voice_processor.update_command_state(
                        command,
                        CommandState.FAILED,
                        f"Failed to execute: {', '.join(failed_skills)}"
                    )
            else:
                self.voice_processor.update_command_state(
                    command,
                    CommandState.FAILED,
                    "No valid tasks planned from command"
                )
                
        except Exception as e:
            error_msg = f"Error processing voice command: {str(e)}"
            print(error_msg)
            self.voice_processor.update_command_state(
                command,
                CommandState.FAILED,
                error_msg
            )

    def run_monitoring(self):
        """Main monitoring loop"""
        try:
            print_header("Starting Robot Monitor")
            self.initialize_robot()
            self.initialize_camera()
            
            print_status(f"Saving images to: {self.save_dir}")
            print_status(f"Capture interval: {self.capture_interval} seconds")
            print_status(f"Max image dimension: {self.processor.max_size}px")
            print_status(f"Voice mode: {'enabled' if self.voice_mode else 'disabled'}")
            
            if self.voice_mode and self.voice_processor:
                self.voice_processor.start_listening()
                print_status("Listening for voice commands starting with 'zero' or 'hey zero'...")
            
            while True:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                if self.voice_mode:
                    command = self.get_voice_command()
                    if command:
                        print_header(f"[{timestamp}] Voice Command Received")
                        image_data, save_path, (width, height, mp) = self.capture_frame()
                        print_success(f"Saved image to: {save_path}")
                        print_status(f"Image size: {width}x{height}px ({mp:.2f} MP)")
                        self.process_voice_command(command)
                else:
                    print_header(f"[{timestamp}] Processing Frame")
                    image_data, save_path, (width, height, mp) = self.capture_frame()
                    print_success(f"Saved image to: {save_path}")
                    print_status(f"Image size: {width}x{height}px ({mp:.2f} MP)")
                    
                    response = self.process_frame_with_voice(image_data, None)
                    print_status(f"Analysis and execution results:\n{response}")
                
                time.sleep(self.capture_interval)
                
        except KeyboardInterrupt:
            print_warning("\nMonitoring stopped by user")
        finally:
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
                       help='Capture interval in seconds (overrides config)')
    parser.add_argument('-d', '--directory', 
                       type=str,
                       help='Directory to save captured images (overrides config)')
    parser.add_argument('-c', '--camera', 
                       type=int,
                       help='Camera device ID (overrides config)')
    parser.add_argument('-p', '--platform', 
                       type=str,
                       choices=['zeroth', 'generic'],
                       help='Robot platform (overrides config)')
    parser.add_argument('-v', '--voice-mode',
                       action='store_true',
                       help='Enable voice command mode (overrides config)')
    parser.add_argument('--whisper-model',
                       type=str,
                       choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help='Whisper model to use (overrides config)')
    parser.add_argument('--config',
                       type=str,
                       default="config/config.yaml",
                       help='Path to configuration file')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Create kwargs dict only for explicitly set arguments
    override_kwargs = {
        "capture_interval": args.interval,
        "save_dir": args.directory,
        "platform": args.platform,
        "voice_mode": args.voice_mode if args.voice_mode else None,
        "whisper_model": args.whisper_model
    }
    # Remove None values (unset arguments)
    override_kwargs = {k: v for k, v in override_kwargs.items() if v is not None}
    
    # Initialize and run monitor
    monitor = RobotMonitor(
        config_path=args.config,
        **override_kwargs
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

