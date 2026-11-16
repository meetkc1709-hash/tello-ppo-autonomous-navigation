"""
Show the ID (and family) of any ArUco marker.

    python read_marker_ids.py                 # live, using the laptop webcam - hold markers up to it
    python read_marker_ids.py photo1.jpg ...  # or read IDs from photos

Webcam window: the ID is written on each marker. Press q to quit.
All IDs seen are listed at the end.
"""
import sys
import cv2

FAMILIES = ["DICT_4X4_50", "DICT_5X5_50", "DICT_6X6_50", "DICT_7X7_50", "DICT_ARUCO_ORIGINAL",
            "DICT_4X4_1000", "DICT_5X5_1000", "DICT_6X6_1000", "DICT_7X7_1000"]
detectors = {f: cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, f)),
                                        cv2.aruco.DetectorParameters()) for f in FAMILIES}
seen = {}

def read(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    found = []
    for fam, det in detectors.items():
        corners, ids, _ = det.detectMarkers(gray)
        if ids is None:
            continue
        for c, i in zip(corners, ids.ravel()):
            short = fam.replace("_1000", "_50") if int(i) < 50 else fam
            found.append((short, int(i), c))
    # same marker can match a family and its bigger version - keep one
    uniq = {}
    for fam, i, c in found:
        key = (round(float(c[0][:, 0].mean()) / 10), round(float(c[0][:, 1].mean()) / 10))
        uniq.setdefault(key, (fam, i, c))
    return list(uniq.values())

def annotate(bgr, found):
    for fam, i, c in found:
        pts = c.reshape(4, 2).astype(int)
        cv2.polylines(bgr, [pts], True, (0, 255, 0), 3)
        x, y = pts[:, 0].min(), pts[:, 1].min() - 10
        cv2.putText(bgr, f"ID {i} ({fam})", (int(x), int(max(y, 30))),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    return bgr

if len(sys.argv) > 1:
    for path in sys.argv[1:]:
        img = cv2.imread(path)
        if img is None:
            print(f"{path}: can't open"); continue
        found = read(img)
        print(f"{path}: " + (", ".join(f"ID {i} ({fam})" for fam, i, _ in found) or "no marker found"))
else:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Can't open the webcam.")
    print("Hold markers up to the webcam. Press q in the window to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        found = read(frame)
        for fam, i, _ in found:
            if (fam, i) not in seen:
                seen[(fam, i)] = True
                print(f"found: ID {i}  ({fam})")
        cv2.imshow("Marker IDs (q to quit)", annotate(frame, found))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    print("\nAll markers seen:", ", ".join(f"ID {i} ({fam})" for fam, i in sorted(seen, key=lambda k: k[1])) or "none")
