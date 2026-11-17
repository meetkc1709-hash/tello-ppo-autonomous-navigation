import cv2
import numpy as np
from pathlib import Path

DISTANCES = [50, 100, 150]

cy_frac = 351.2 / 720.0
strip = 0.10

panels = []

for distance in DISTANCES:

    path = Path(f"tello_obstacle_{distance}cm.png")
    image = cv2.imread(str(path))

    if image is None:
        print("Could not load:", path)
        continue

    height, width = image.shape[:2]

    y1 = int((cy_frac - strip) * height)
    y2 = int((cy_frac + strip) * height)

    overlay = image.copy()

    # Highlight the region used by strip mode.
    cv2.rectangle(
        overlay,
        (0, y1),
        (width - 1, y2),
        (0, 255, 0),
        -1
    )

    image = cv2.addWeighted(
        overlay, 0.25,
        image, 0.75,
        0
    )

    # Draw boundaries.
    cv2.line(
        image, (0, y1),
        (width - 1, y1),
        (0, 255, 0), 2
    )

    cv2.line(
        image, (0, y2),
        (width - 1, y2),
        (0, 255, 0), 2
    )

    # Show the center ray.
    cv2.line(
        image,
        (width // 2, y1),
        (width // 2, y2),
        (0, 0, 255),
        3
    )

    cv2.putText(
        image,
        f"Obstacle: {distance} cm",
        (25, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        3
    )

    panels.append(image)

if not panels:
    raise RuntimeError("No images were loaded.")

comparison = np.hstack(panels)

output = "tello_depth_region_comparison.png"

cv2.imwrite(output, comparison)

print("Visualization saved:", output)
print("Green region = depth-sensing strip")
print("Red line = center ray")
