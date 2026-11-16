# Lab checklist – Tello PPO with mission pads

Use Python 3.12 for everything (it has djitellopy, OpenCV and PyTorch):
`& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" <script> ...`
Allow it through Windows Firewall once: Windows key → "Allow an app through Windows Firewall" →
Change settings → Allow another app → `...\Python312\python.exe` → tick Private + Public.

## Done ✔
- Connection, takeoff/land
- Speed: rc 40 → 0.31 m/s, rc 60 → 0.51, rc 80 → 0.82; delay 0.51 s; tau 0.27 s
- Drone is a Tello EDU (SDK 30). Colour pad 5 detected: 30 cm 25/25, 50 cm 13/25, 80 cm 0/25
  -> fly at ~35 cm (run_on_tello.py now descends and holds 35 cm automatically in pads mode)

## 1. Pads ✔ (colour print works at low height)
- More pads = better: print other pad numbers in colour too if you have them (1-8).
- Hold the drone ~30 cm over a pad for all hand tests.

## 2. Sideways speed sign (1 flight, ~2 m clear to the drone's RIGHT)
`3_calibrate.py --rc 60 --side` → note the vy_sign it prints.

## 3. Lay out the pads
Several pads:  `5_pad_layout.py --pads <your pad numbers> --area 2.0 2.0 --vy_sign <from step 2>`
One pad only:  `5_pad_layout.py --pads 5 --area 2.0 2.0 --at 0.5 1.0 --vy_sign <from step 2>`
  (pad 5 at x=0.5, y=1.0 = the takeoff spot; the drone reads its exact start from it,
   then uses its speed readings - keep the first goals short, ~1-1.5 m)
- All pads the SAME WAY ROUND.
- Origin = back-right floor corner of the area, x = forward, y = to the LEFT.
- Find the pad axes with `2_check_position.py` (instructions at the top of that file), then
  re-run `5_pad_layout.py` adding `--pad_x_is .. --pad_y_is ..`.
- Check: hold the drone over each pad → printed room position ≈ tape-measured position.

## 4. Train for this area (on the laptop, a few minutes)
From the tello_rl folder: `train.py --stage 1 --room 2.0 2.0` (same numbers as --area)

## 5. First PPO flight
From the tello_rl folder, drone ON a pad, FACING FORWARD (+x). It takes off, drops to ~35 cm, reads the pad, then flies:

    run_on_tello.py --model models/stage1/best_model.zip --room 2.0 2.0 --goal 1.6 1.0 --max_rc 60

- If the takeoff spot isn't on a pad, add `--start <x> <y>`.
- Lands by itself if no pad is seen for 8 s, it leaves the area, or Ctrl+C.
- Film every flight; send `flight_log_<time>.csv` + video to Bowen.
