import openlch
import time
from typing import Dict
import numpy as np


def print_imu_data(imu_data: Dict) -> None:
    """Pretty print IMU data."""
    print("\nIMU Data:")
    print("-" * 40)
    
    # Gyroscope data (degrees/s)
    print("Gyroscope (deg/s):")
    print(f"  X: {imu_data['gyro']['x']:8.3f}")
    print(f"  Y: {imu_data['gyro']['y']:8.3f}")
    print(f"  Z: {imu_data['gyro']['z']:8.3f}")
    
    # Accelerometer data (mg)
    print("\nAccelerometer (mg):")
    print(f"  X: {imu_data['accel']['x']:8.3f}")
    print(f"  Y: {imu_data['accel']['y']:8.3f}")
    print(f"  Z: {imu_data['accel']['z']:8.3f}")


def main():
    # Initialize HAL
    print("Initializing HAL...")
    hal = openlch.HAL()
    imu = hal.imu
    
    try:
        # Read IMU data in a loop
        print("\nReading IMU data (Press Ctrl+C to stop)...")
        while True:
            imu_data = imu.get_data()
            print_imu_data(imu_data)
            time.sleep(0.1)  # Read at 10Hz
            
    except KeyboardInterrupt:
        print("\nStopping IMU readings...")
    
    print("Test complete!")


if __name__ == "__main__":
    main()