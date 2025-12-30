# propagate_heel.py
import cv2
import json
import numpy as np
from pathlib import Path

frames_dir = Path("new_training/data/frames")
keyframe_labels = json.loads(Path("new_training/data/heel_labels_keyframes.json").read_text())


start_frame = 0
end_frame = 1400  # we'll only label in [0, 1400)
max_steps = 40    # max frames to propagate from each seed


keyframe_labels = {int(k): v for k, v in keyframe_labels.items()}

# Load frames in the ROI into memory
frame_paths = [frames_dir / f"frame_{i:05d}.png"
               for i in range(start_frame, end_frame)]
frames = [cv2.imread(str(p)) for p in frame_paths]
num_frames = len(frames)  # should be end_frame - start_frame

# Initialize label dict for ROI
all_labels = {i: None for i in range(start_frame, end_frame)}

# Seed with keyframe labels
for i, coord in keyframe_labels.items():
    if start_frame <= i < end_frame:
        all_labels[i] = coord

lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

def track_from_seed(seed_idx, seed_xy, direction, max_steps=40):
    p = np.array([[seed_xy]], dtype=np.float32)  # shape (1,1,2)
    idx = seed_idx
    steps = 0

    while steps < max_steps:
        next_idx = idx + direction
        if not (start_frame <= next_idx < end_frame):
            break

        prev_gray = cv2.cvtColor(frames[idx - start_frame], cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(frames[next_idx - start_frame], cv2.COLOR_BGR2GRAY)

        p_next, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, p, None, **lk_params)
        if st[0, 0] == 0:
            break

        x, y = p_next[0, 0]
        h, w = next_gray.shape
        if not (0 <= x < w and 0 <= y < h):
            break

        if all_labels[next_idx] is None:
            all_labels[next_idx] = {"x": float(x), "y": float(y)}

        p = p_next
        idx = next_idx
        steps += 1

# Run propagation from each keyframe
for i, coord in keyframe_labels.items():
    if not (start_frame <= i < end_frame):
        continue
    xy = (coord["x"], coord["y"])
    track_from_seed(i, xy, direction=+1, max_steps=max_steps)
    track_from_seed(i, xy, direction=-1, max_steps=max_steps)


# Save
Path("new_training/data/heel_labels_all.json").write_text(json.dumps(all_labels, indent=2))
print(f"Saved propagated labels for frames [{start_frame}, {end_frame})")
