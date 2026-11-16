"""Lets lab scripts find the tello_rl folder whether lab/ is inside tello_rl/ or next to it."""
import os, sys
here = os.path.dirname(os.path.abspath(__file__))
for cand in [os.path.join(here, ".."), os.path.join(here, "..", "tello_rl")]:
    if os.path.exists(os.path.join(cand, "tello_env.py")):
        ROOT = os.path.abspath(cand)
        sys.path.insert(0, ROOT)
        break
else:
    raise SystemExit("Can't find the tello_rl folder - put lab/ inside (or next to) tello_rl.")
