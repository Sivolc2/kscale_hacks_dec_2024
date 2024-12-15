import time
from robot import Robot

def wave(robot: Robot):
    print("Waving")
    
    # Initial position
    initial_positions = {
        "left_shoulder_yaw": 0.0,
        "left_shoulder_pitch": 0.0,
        "left_elbow_yaw": 0.0,
    }
    robot.set_desired_positions(initial_positions)
    time.sleep(0.5)

    # Raise arm
    wave_up_positions = {
        "left_shoulder_pitch": 0.0,
        "left_shoulder_yaw": 150.0,
    }
    robot.set_desired_positions(wave_up_positions)
    time.sleep(0.5)

    # Wave motion
    for _ in range(6):
        # Wave out
        robot.set_desired_positions({"left_elbow_yaw": -90.0})
        time.sleep(0.3)
        
        # Wave in
        robot.set_desired_positions({"left_elbow_yaw": -45.0})
        time.sleep(0.3)

    # Return to initial position
    robot.set_desired_positions(initial_positions)
    time.sleep(0.5)

def main():
    robot = Robot()
    try:
        robot.initialize()
        wave(robot)
    except Exception as e:
        print(f"Error during robot operation: {e}")
    finally:
        robot.disable_motors()
        print("Motors disabled")

if __name__ == "__main__":
    main()

