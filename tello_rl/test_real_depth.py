import time
import cv2
import numpy as np
import torch

from pathlib import Path
from depth_sensor import DepthRays

DISTANCES = [50, 100, 150]

print("\n=== REAL TELLO DEPTH EVALUATION ===")

# Start with strip mode so incorrect camera-height
# assumptions do not affect this initial test.
sensor = DepthRays(
    n_rays=9,
    max_range=2.0,
    mode="strip",
    input_height=196
)

print("\nDepth model loaded successfully.")

results = []

for distance_cm in DISTANCES:

    path = Path(f"tello_obstacle_{distance_cm}cm.png")

    if not path.exists():
        print("Missing image:", path)
        continue

    frame_bgr = cv2.imread(str(path))

    if frame_bgr is None:
        print("Cannot read image:", path)
        continue

    frame_rgb = cv2.cvtColor(
        frame_bgr,
        cv2.COLOR_BGR2RGB
    )

    print(f"\nProcessing {distance_cm} cm image...")

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()

    depth = sensor.depth_map(frame_rgb)
    rays = sensor.rays_from_depth(depth)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    if not np.all(np.isfinite(depth)):
        raise RuntimeError("Invalid values in depth map.")

    center_ray = float(rays[len(rays) // 2])
    actual_distance = distance_cm / 100.0

    error = center_ray - actual_distance

    print("Actual distance:", actual_distance, "m")
    print("Estimated center ray:", round(center_ray, 3), "m")
    print("Error:", round(error, 3), "m")
    print("Inference time:", round(elapsed, 3), "seconds")

    print("All 9 rays:")
    print(np.round(rays, 3))

    results.append({
        "actual_cm": distance_cm,
        "predicted_cm": round(center_ray * 100, 2),
        "error_cm": round(error * 100, 2),
        "inference_ms": round(elapsed * 1000, 2)
    })

print("\n=== SUMMARY ===")

for result in results:
    print(result)

print("\nEvaluation completed.")
