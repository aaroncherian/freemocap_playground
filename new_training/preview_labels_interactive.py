# preview_labels_interactive.py
from pathlib import Path
import json
import cv2

frames_dir = Path("new_training/data/frames")
labels_path = Path("new_training/data/heel_labels_all.json")

# Load labels
labels = json.loads(labels_path.read_text())

# If keys are strings, normalize to int
norm_labels = {}
for k, v in labels.items():
    if v is not None:
        norm_labels[int(k)] = v

frame_paths = sorted(frames_dir.glob("frame_*.png"))

for i, frame_path in enumerate(frame_paths):
    frame = cv2.imread(str(frame_path))

    # Get label if available
    lab = norm_labels.get(i, None)
    if lab is not None:
        x, y = int(lab["x"]), int(lab["y"])
        # Draw a red dot at the heel
        cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)
    
    # Put frame index on the image
    cv2.putText(
        frame,
        f"frame {i}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow("Heel labels preview (q to quit, any key = next)", frame)
    key = cv2.waitKey(0)
    if key & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
