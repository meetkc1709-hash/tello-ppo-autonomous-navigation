import time
from djitellopy import Tello

tello = Tello()
takeoff_attempted = False

try:
    print("Connecting to Tello...")
    tello.connect()

    battery = tello.get_battery()
    print(f"Battery: {battery}%")

    if battery < 50:
        raise RuntimeError("Battery below 50%. Flight cancelled.")

    print("\nPreflight checks:")
    print("- Flight area is clear")
    print("- Drone is on a level surface")
    print("- No temperature warnings")
    print("- Operator is ready to intervene")

    confirmation = input(
        "\nType TAKEOFF to begin the hover test: "
    )

    if confirmation != "TAKEOFF":
        print("Flight cancelled.")
    else:
        print("\nTaking off...")
        takeoff_attempted = True
        tello.takeoff()

        print("Hovering for 3 seconds...")
        time.sleep(3)

        print("Landing...")
        tello.land()
        takeoff_attempted = False

        print("\nHover test completed successfully.")

except KeyboardInterrupt:
    print("\nFlight interrupted. Attempting landing.")

except Exception as e:
    print(f"\nFlight error: {e}")

finally:
    if takeoff_attempted:
        try:
            print("Sending landing command...")
            tello.land()
        except Exception as e:
            print(f"Landing command failed: {e}")
            print("Use the available manual emergency procedure.")

    tello.end()
    print("Tello connection closed.")
