import time
import threading
from skill_library import SkillLibrary, RobotPlatform
from robot import Robot

def test_skill_library():
    """Test the skill library functionality with real robot"""
    print("\n=== Starting Skill Library Test ===")
    skill_lib = SkillLibrary()
    robot = Robot()
    
    # Print available skills
    print("\nAvailable Skills:")
    print(skill_lib.get_skill_descriptions())
    
    # Test each skill
    try:
        print("\nInitializing robot...")
        robot.initialize()
        
        print("\n1. Testing 'reset_positions' skill (initial reset)...")
        skill_lib.execute_skill("reset_positions", RobotPlatform.ZEROTH, robot)
        time.sleep(2)  # Give time for movement to complete
        
        print("\n2. Testing 'stand' skill...")
        skill_lib.execute_skill("stand", RobotPlatform.ZEROTH, robot)
        time.sleep(2)
        
        print("\n3. Testing 'wave' skill...")
        skill_lib.execute_skill("wave", RobotPlatform.ZEROTH, robot)
        time.sleep(1)
        
        # print("\n4. Testing 'recover_forward' skill...")
        # skill_lib.execute_skill("recover_forward", RobotPlatform.ZEROTH, robot)
        # time.sleep(3)
        
        print("\n5. Testing 'walk_forward' skill (brief test)...")
        # Create a stop event for the walking test
        stop_event = threading.Event()
        
        # Start walking in a separate thread so we can stop it
        walk_thread = threading.Thread(
            target=lambda: skill_lib.execute_skill(
                "walk_forward", 
                RobotPlatform.ZEROTH,
                robot,
                stop_event=stop_event
            )
        )
        
        walk_thread.start()
        print("Walking started... will stop in 5 seconds")
        time.sleep(5)
        
        print("Stopping walk...")
        stop_event.set()
        walk_thread.join(timeout=2)

        print("\n5. Testing 'walk_backward' skill (brief test)...")
        # Create a stop event for the walking test
        stop_event = threading.Event()
        
        # Start walking in a separate thread so we can stop it
        walk_thread = threading.Thread(
            target=lambda: skill_lib.execute_skill(
                "walk_backward", 
                RobotPlatform.ZEROTH,
                robot,
                stop_event=stop_event
            )
        )
        
        walk_thread.start()
        print("Walking started... will stop in 5 seconds")
        time.sleep(5)
        
        print("Stopping walk...")
        stop_event.set()
        walk_thread.join(timeout=2)
        
        print("\nFinal position reset...")
        skill_lib.execute_skill("reset_positions", RobotPlatform.ZEROTH, robot)
        
        print("\n=== Test completed successfully! ===")
        
    except Exception as e:
        print(f"\nError during test: {str(e)}")
        raise
    finally:
        print("\nDisabling motors...")
        robot.disable_motors()

def test_error_cases():
    """Test error handling in the skill library"""
    print("\n=== Starting Error Case Tests ===")
    skill_lib = SkillLibrary()
    robot = Robot()
    
    try:
        robot.initialize()
        
        # Test invalid skill name
        print("\n1. Testing invalid skill name...")
        try:
            skill_lib.execute_skill("nonexistent_skill", RobotPlatform.ZEROTH, robot)
        except ValueError as e:
            print(f"Successfully caught error: {e}")
            
        # Test invalid platform
        print("\n2. Testing invalid platform...")
        try:
            skill_lib.execute_skill("stand", RobotPlatform.GENERIC, robot)
            print("Note: Generic platform uses placeholder implementations")
        except Exception as e:
            print(f"Got expected behavior for generic platform")
            
        print("\n=== Error case tests completed! ===")
        
    except Exception as e:
        print(f"\nUnexpected error during test: {str(e)}")
        raise
    finally:
        robot.disable_motors()

if __name__ == "__main__":
    print("=== Skill Library Test Suite ===")
    print("This test suite verifies the functionality of the robot skill library")
    print("It uses the actual robot hardware")
    
    try:
        # Run main functionality tests
        test_skill_library()
        
        # Run error case tests
        test_error_cases()
        
        print("\n=== All tests completed successfully! ===")
        
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
    except Exception as e:
        print(f"\nTests failed with error: {str(e)}")
        raise