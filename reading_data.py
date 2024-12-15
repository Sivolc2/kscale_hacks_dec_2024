from openlch.hal import HAL
import math
import time

# Initialize robot
robot = HAL()

# Enable movement and set torque
robot.servo.enable_movement()
robot.servo.set_torque([(servo_id, 60) for servo_id in range(1, 17)])

# Define the offset positions
positions_to_set = [
    # Left Leg
    (10, 0.23 * 180/math.pi + 5.0),    # left_hip_pitch
    (9, 45.0),                         # left_hip_yaw
    (8, 0.0),                          # left_hip_roll
    (7, -0.741 * 180/math.pi),         # left_knee_pitch
    (6, -0.5 * 180/math.pi),           # left_ankle_pitch
    
    # Right Leg
    (5, -0.23 * 180/math.pi - 5.0),    # right_hip_pitch
    (4, -45.0),                        # right_hip_yaw
    (3, 0.0),                          # right_hip_roll
    (2, 0.741 * 180/math.pi),          # right_knee_pitch
    (1, 0.5 * 180/math.pi),            # right_ankle_pitch
    
    # Arms
    (11, 0.0),                         # right_elbow_yaw
    (12, 0.0),                         # right_shoulder_yaw
    (13, 0.0),                         # right_shoulder_pitch
    (14, 0.0),                         # left_shoulder_pitch
    (15, 0.0),                         # left_shoulder_yaw
    (16, 0.0)                          # left_elbow_yaw
]

# Set the initial positions
robot.servo.set_positions(positions_to_set)

robot.servo.set_torque([(servo_id, 5) for servo_id in range(1, 17)])

# Create a dictionary to map servo IDs to their names for better readability
servo_names = {
    1: "right_ankle_pitch", 2: "right_knee_pitch", 3: "right_hip_roll",
    4: "right_hip_yaw", 5: "right_hip_pitch", 6: "left_ankle_pitch",
    7: "left_knee_pitch", 8: "left_hip_roll", 9: "left_hip_yaw",
    10: "left_hip_pitch", 11: "right_elbow_yaw", 12: "right_shoulder_yaw",
    13: "right_shoulder_pitch", 14: "left_shoulder_pitch",
    15: "left_shoulder_yaw", 16: "left_elbow_yaw"
}

try:
    while True:
        # Get current positions
        current_positions = robot.servo.get_positions()
        
        # Print timestamp
        print(f"\nTimestamp: {time.strftime('%H:%M:%S')}")
        print("Current positions:")
        
        # breakpoint()
        # Print each servo's position with its name
        for servo_id, position, velocity in current_positions:
            servo_name = servo_names.get(servo_id, f"Servo {servo_id}")
            print(f"{servo_name}: {position:.2f} degrees, {velocity:.2f} degrees/s")
        
        # Wait for 0.5 seconds
        time.sleep(0.5)

except KeyboardInterrupt:
    print("\nStopping position monitoring...")
    # You might want to add any cleanup code here if necessary
