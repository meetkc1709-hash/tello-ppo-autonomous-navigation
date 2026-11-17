import time
import cv2
from pathlib import Path
from djitellopy import Tello

OUTPUT = Path("tello_lab_frame.png")

tello = Tello()

try:
    print("Connecting...")
    tello.connect()

    battery = tello.get_battery()
    temperature = tello.get_temperature()

    print(f"Battery: {battery}%")
    print(f"Temperature: {temperature} C")

    if battery < 50:
        raise RuntimeError("Battery too low.")

    if temperature >= 75:
        raise RuntimeError("Temperature elevated. Stop testing.")

    print("Starting camera...")
    tello.streamon()

    reader = tello.get_frame_read()
    time.sleep(2)

    frame_rgb = reader.frame

    if frame_rgb is None:
        raise RuntimeError("No camera frame received.")

    # Convert RGB to BGR for OpenCV image saving.
    frame_bgr = cv2.cvtColor(
        frame_rgb,
        cv2.COLOR_RGB2BGR
    )

    if not cv2.imwrite(str(OUTPUT), frame_bgr):
        raise RuntimeError("Failed to save image.")

    print("Image saved:", OUTPUT)
    print("Resolution:", frame_bgr.shape)

finally:
    try:
        tello.streamoff()
    except Exception:
        pass

    tello.end()
    print("Connection closed.")
