# new_training/preview_trained_model.py
from pathlib import Path
import json

import cv2
import numpy as np
import torch

from new_training.train_model import TinyKeypointNet, DEVICE
from make_dataset import FRAMES_DIR, LABELS_PATH, IMAGE_SIZE

BASE = Path("new_training/data")
MODEL_PATH = Path("new_training/heel_model.pth")

# Load labels
labels_raw = json.loads(LABELS_PATH.read_text())
labels = {int(k): v for k, v in labels_raw.items() if v is not None}

# Load model
model = TinyKeypointNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Choose a few frames to visualize
sample_indices = sorted(list(labels.keys()))[::10]  # every 50th labeled frame

for frame_idx in sample_indices:
    # Load original image
    img_path = FRAMES_DIR / f"frame_{frame_idx:05d}.png"
    img = cv2.imread(str(img_path))
    h, w, _ = img.shape

    # Prepare input
    img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img_resized_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_input = img_resized_rgb.astype(np.float32) / 255.0
    img_input = torch.from_numpy(img_input).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    # Predict
    with torch.no_grad():
        pred_norm = model(img_input)[0].cpu().numpy()  # (2,)
    x_norm, y_norm = pred_norm
    x_pred = int(x_norm * w)
    y_pred = int(y_norm * h)

    # Ground truth
    gt = labels[frame_idx]
    x_gt = int(gt["x"])
    y_gt = int(gt["y"])

    # Draw GT (green) and prediction (red)
    cv2.circle(img, (x_gt, y_gt), 6, (0, 255, 0), -1)   # GT = green
    cv2.circle(img, (x_pred, y_pred), 6, (0, 0, 255), -1)  # pred = red

    cv2.putText(img, f"frame {frame_idx}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    cv2.imshow("GT (green) vs Pred (red) - q to quit, any key = next", img)
    key = cv2.waitKey(0)
    if key & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
