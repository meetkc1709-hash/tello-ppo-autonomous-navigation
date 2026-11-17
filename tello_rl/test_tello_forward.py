import time
from djitellopy import Tello

tello = Tello()
airborne = False

try:
    print("Connecting...")
    tello.connect()

    battery = tello.get_battery()
    temperature = tello.get_temperature()

    print(f"Battery: {battery}%")
    print(f"Temperature: {temperature} C")

    if battery < 50:
        raise RuntimeError("Battery too low.")

    print("\nTest: 20 cm forward movement")
    print("Confirm that the area is clear.")
    print("Ensure independent manual landing is available.")

    confirmation = input("Type TAKEOFF to begin: ")

    if confirmation != "TAKEOFF":
        raise SystemExit("Test cancelled.")

    airborne = True

    print("\nTaking off...")
    tello.takeoff()

    time.sleep(3)

    print("\nInitial telemetry:")
    print("Height:", tello.get_height(), "cm")
    print("Speed X:", tello.get_speed_x(), "cm/s")
    print("Speed Y:", tello.get_speed_y(), "cm/s")

    print("\nMoving forward 20 cm...")
    start = time.monotonic()

    tello.move_forward(20)

    duration = time.monotonic() - start

    print(f"Command duration: {duration:.2f} seconds")

    print("\nHovering after movement...")
    time.sleep(3)

    print("\nFinal telemetry:")
    print("Height:", tello.get_height(), "cm")
    print("Speed X:", tello.get_speed_x(), "cm/s")
    print("Speed Y:", tello.get_speed_y(), "cm/s")

    print("\nLanding...")
    tello.land()
    airborne = False

    print("\nMovement test completed.")

except KeyboardInterrupt:
    print("\nInterrupted. Attempting landing.")

except Exception as e:
    print("\nError:", e)

finally:
    if airborne:
        try:
            tello.land()
        except Exception as e:
            print("Landing failed:", e)
            print("Use independent manual landing control.")

    tello.end()
    print("Connection closed.")
