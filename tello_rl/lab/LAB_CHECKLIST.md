# Lab day checklist – Tello test

Goal for this visit: **collect measurements**, not a full demo. (No PPO flight yet – that needs position tracking, which we set up next.)

## Bring
- Tello + all batteries **fully charged** (each lasts ~10 min – plan ~4 flights per battery)
- Tape measure, masking tape, phone to film every flight
- Laptop with this `tello_rl` folder, including `models/stage1/best_model.zip`

## Before going (at home, with internet)
```
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m pip install djitellopy
```
(Use whatever Python has stable-baselines3 installed. In the lab, the laptop's Wi-Fi connects
to the drone, so there will be **no internet** – install everything first.)

## Safety
- Empty space, nobody standing near the flight path, start low
- Every script lands the drone if it crashes or you press **Ctrl+C** in the terminal
- Battery < 30% → swap it

## Steps (run from inside the `lab` folder)
| # | Command | Write down |
|---|---------|------------|
| 1 | `python 1_check_tello.py` then `python 1_check_tello.py --fly` | Did it connect? Take off / land OK? |
| 2 | ~~2_check_position.py~~ **skip – we have no mission pads** | – |
| 3 | `python 3_calibrate.py --rc 40` and `--rc 70` (2× each if battery allows) | Distance moved for each run (cm, by tape) |
| 4 | Measure the flight area and the **ceiling height** | Width × length (m), ceiling height (m), is there a spot to mount a webcam looking down? |

Replace `python` with the full path to Python 3.12 if needed:
`& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"`

## Send back to Bowen
- All `calib_rc*.csv` files
- The tape distances and the notes above
- Videos of the flights

These numbers go into `tello_env.py` (max_speed, tau, latency) and `run_on_tello.py`
(RC_PER_MPS); the room/ceiling info decides the webcam position-tracking setup before the first real PPO flight.
