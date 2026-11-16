"""
Why isn't the mission pad detected? Run with the drone on and held/placed over the pad.

    python pad_diagnose.py

Checks: SDK version (pads need SDK 2.0+, i.e. a Tello EDU), whether the drone accepts the
pad commands, and the raw 'mid' value at different heights. Follow the prompts.
"""
import time
from djitellopy import Tello

def close_tello(t):
    """Close the drone connection once. djitellopy closes it again when Python exits, which
    fails with 'WinError 6 handle is invalid' - so disable that second close."""
    try:
        if getattr(t, "stream_on", False):
            t.streamoff()
    except Exception:
        pass
    try:
        t.end()
    except Exception:
        pass
    t.end = lambda *a, **k: None

tello = Tello()
tello.connect()
print("Battery:", tello.get_battery(), "%")

def ask(cmd):
    try:
        return tello.send_read_command(cmd)
    except Exception as e:
        return f"ERROR ({e})"

print("\n--- Drone info ---")
print("SDK version :", ask("sdk?"), "  (20 or 30 = Tello EDU / pads supported; error or 13 = regular Tello)")
print("Serial no.  :", ask("sn?"))
try:
    r = tello.send_command_with_return("mon")
    print("'mon' (pads on) reply  :", r)
    r = tello.send_command_with_return("mdirection 0")
    print("'mdirection 0' reply   :", r, "  (0 = look down)")
except Exception as e:
    print("Pad commands failed:", e)

state = tello.get_current_state()
print("\nState fields the drone sends:", ", ".join(state.keys()))
if "mid" not in state:
    print(">>> No 'mid' field: this drone/firmware is not sending mission-pad data at all.")

for h in ["~30 cm", "~50 cm", "~80 cm"]:
    input(f"\nHold the drone {h} above the CENTRE of the pad, camera facing forward, then press Enter...")
    seen = []
    for _ in range(25):
        s = tello.get_current_state()
        seen.append(s.get("mid", "n/a"))
        time.sleep(0.2)
    hits = [m for m in seen if m not in (-1, -2, "n/a")]
    print(f"  mid values over 5 s: {sorted(set(seen), key=str)}   -> detected {len(hits)}/25")

print("\nNote: the pad camera is on the BOTTOM of the drone - make sure nothing covers it and the")
print("pad is well lit (no strong reflections / shadows).")
close_tello(tello)
