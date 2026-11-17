import cv2
import time
import numpy as np
from djitellopy import Tello

tello = Tello()

try:
    tello.connect()

    print("Battery:", tello.get_battery(), "%")
    print("Temperature:", tello.get_temperature(), "C")

    tello.streamon()
    reader = tello.get_frame_read()

    print("\nComparing color formats...")
    print("Press Q to quit.")

    start = time.monotonic()

    while time.monotonic() - start < 10:
        frame = reader.frame

        if frame is None:
            time.sleep(0.05)
            continue

        original = frame.copy()
        swapped = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2BGR
        )

        cv2.putText(
            original,
            "ORIGINAL",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.putText(
            swapped,
            "RGB-BGR SWAPPED",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        comparison = np.hstack((original, swapped))

        cv2.imshow("Tello Color Comparison", comparison)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cv2.destroyAllWindows()

    try:
        tello.streamoff()
    except Exception:
        pass

    tello.end()

    print("Color comparison completed.")
