import time
import cv2
from djitellopy import Tello

tello = Tello()

try:
    print("Connecting...")
    tello.connect()

    print("Battery:", tello.get_battery(), "%")
    print("Temperature:", tello.get_temperature(), "C")

    tello.streamoff()
    tello.streamon()

    frame_reader = tello.get_frame_read()

    print("\nReceiving camera frames...")
    print("Press Q to quit.")

    start = time.monotonic()
    count = 0

    while time.monotonic() - start < 15:
        frame = frame_reader.frame

        if frame is None:
            continue

        count += 1

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Tello Camera Test", frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        time.sleep(0.03)

    duration = time.monotonic() - start

    print(f"\nFrames displayed: {count}")
    print(f"Elapsed time: {duration:.2f} seconds")
    print(f"Approximate display FPS: {count / duration:.2f}")

finally:
    cv2.destroyAllWindows()

    try:
        tello.streamoff()
    except Exception:
        pass

    tello.end()

    print("Camera test completed.")
