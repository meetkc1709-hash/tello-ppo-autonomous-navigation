"""
STEP 1 - Can the laptop talk to the Tello and fly it?

Connect the laptop's Wi-Fi to the Tello network (TELLO-xxxxxx) first.
    python 1_check_tello.py            # connect + battery only (no flight)
    python 1_check_tello.py --fly      # also take off, hover 3 s, land
"""
import argparse, time
from djitellopy import Tello

ap = argparse.ArgumentParser()
ap.add_argument("--fly", action="store_true")
args = ap.parse_args()

tello = Tello()
tello.connect()
print("Connected!")
print("Battery     :", tello.get_battery(), "%")
print("Temperature :", tello.get_temperature(), "C")

if args.fly:
    assert tello.get_battery() > 30, "Battery below 30% - charge it first"
    try:
        tello.takeoff()
        print("Height after takeoff:", tello.get_height(), "cm")
        time.sleep(3)
    finally:
        tello.land()
        print("Landed.")
tello.end()
