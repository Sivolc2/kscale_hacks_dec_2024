import cv2
from ultralytics import YOLO
import numpy as np
import argparse
import os
from openlch.hal import HAL
import math

os.environ['CURL_CA_BUNDLE'] = ''

def calculate_joint_angles(keypoints):
    """Convert YOLO keypoints to robot joint angles"""
    # YOLO keypoint indices:
    # 5,6: shoulders | 7,8: elbows | 9,10: wrists
    joint_angles = {}
    
    # Calculate angles for both arms if keypoints are detected
    if keypoints is not None:
        kps = keypoints[0].data[0]  # Get first person's keypoints
        
        # Left arm (shoulder pitch and elbow)
        if kps[5][2] > 0.5 and kps[7][2] > 0.5:  # If confidence is high enough
            # Calculate shoulder pitch (vertical angle)
            shoulder_y_diff = kps[7][1] - kps[5][1]
            shoulder_x_diff = kps[7][0] - kps[5][0]
            left_shoulder_pitch = -math.degrees(math.atan2(shoulder_y_diff, shoulder_x_diff)) + 90
            joint_angles["left_shoulder_pitch"] = left_shoulder_pitch

        if kps[7][2] > 0.5 and kps[9][2] > 0.5:  # Elbow and wrist visible
            # Calculate elbow angle
            elbow_y_diff = kps[9][1] - kps[7][1]
            elbow_x_diff = kps[9][0] - kps[7][0]
            left_elbow = math.degrees(math.atan2(elbow_y_diff, elbow_x_diff))
            joint_angles["left_elbow_yaw"] = left_elbow

        # Right arm (mirror of left arm calculations)
        if kps[6][2] > 0.5 and kps[8][2] > 0.5:
            shoulder_y_diff = kps[8][1] - kps[6][1]
            shoulder_x_diff = kps[8][0] - kps[6][0]
            right_shoulder_pitch = -math.degrees(math.atan2(shoulder_y_diff, shoulder_x_diff)) + 90
            joint_angles["right_shoulder_pitch"] = right_shoulder_pitch

        if kps[8][2] > 0.5 and kps[10][2] > 0.5:
            elbow_y_diff = kps[10][1] - kps[8][1]
            elbow_x_diff = kps[10][0] - kps[8][0]
            right_elbow = math.degrees(math.atan2(elbow_y_diff, elbow_x_diff))
            joint_angles["right_elbow_yaw"] = right_elbow

    return joint_angles

def run_pose_control(source='0', conf=0.5):
    # Initialize robot
    robot = HAL()
    
    # Servo ID mapping
    servo_mapping = {
        "right_elbow_yaw": 11,
        "right_shoulder_pitch": 13,
        "left_shoulder_pitch": 14,
        "left_elbow_yaw": 16
    }
    
    # Joint offsets (from digital_twin.py)
    joint_offsets = {
        "right_elbow_yaw": 0.0,
        "right_shoulder_pitch": 0.0,
        "left_shoulder_pitch": 0.0,
        "left_elbow_yaw": 0.0
    }

    # Load YOLO model
    try:
        model = YOLO('yolov8n-pose.pt')
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    # Setup video capture
    if source.isnumeric():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error: Couldn't open video source")
        return

    # Enable servos with moderate torque
    servo_ids = list(servo_mapping.values())
    robot.servo.set_torque_enable([(servo_id, True) for servo_id in servo_ids])
    robot.servo.set_torque([(servo_id, 50) for servo_id in servo_ids])
    robot.servo.enable_movement()

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Run pose detection
            results = model(frame, conf=conf)

            # Process predictions and calculate joint angles
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    joint_angles = calculate_joint_angles(result.keypoints)
                    
                    # Convert joint angles to servo positions
                    positions = []
                    for joint_name, angle in joint_angles.items():
                        if joint_name in servo_mapping:
                            servo_id = servo_mapping[joint_name]
                            # Apply offset and convert to servo position
                            servo_pos = angle + joint_offsets[joint_name]
                            positions.append((servo_id, servo_pos))
                    
                    # Send positions to robot if any were calculated
                    if positions:
                        robot.servo.set_positions(positions)

            # Display the frame
            cv2.putText(frame, 'Press Q to quit', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Pose Control', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        robot.servo.disable_movement()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0',
                       help='Source (0 for webcam, or video file path)')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold')
    args = parser.parse_args()

    run_pose_control(args.source, args.conf)