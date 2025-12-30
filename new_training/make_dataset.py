# new_training/make_dataset.py
from pathlib import Path
import json
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# -------------------------
# Config
# -------------------------
BASE = Path("new_training/data")
FRAMES_DIR = BASE / "frames"
LABELS_PATH = BASE / "heel_labels_all.json"

START_FRAME = 0
END_FRAME = 1400        # use [0, 1400)
IMAGE_SIZE = 256        # model input size (HxW)
TRAIN_FRACTION = 0.8    # 80% train, 20% val
BATCH_SIZE = 32


# -------------------------
# Load labels and collect valid frame indices
# -------------------------
labels_raw = json.loads(LABELS_PATH.read_text())

# Normalize keys to int
labels = {}
for k, v in labels_raw.items():
    i = int(k)
    labels[i] = v

valid_indices = [
    i for i in range(START_FRAME, END_FRAME)
    if (i in labels) and (labels[i] is not None)
]

print(f"Total valid labeled frames in [{START_FRAME}, {END_FRAME}): {len(valid_indices)}")

# Shuffle and split
random.shuffle(valid_indices)
split_idx = int(len(valid_indices) * TRAIN_FRACTION)
train_indices = valid_indices[:split_idx]
val_indices = valid_indices[split_idx:]

print(f"Train frames: {len(train_indices)}, Val frames: {len(val_indices)}")


# -------------------------
# Dataset class
# -------------------------
class HeelDataset(Dataset):
    def __init__(self, indices, frames_dir, labels_dict, image_size=256):
        self.indices = indices
        self.frames_dir = frames_dir
        self.labels_dict = labels_dict
        self.image_size = image_size

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        frame_idx = self.indices[idx]

        # Load image
        img_path = self.frames_dir / f"frame_{frame_idx:05d}.png"
        img = cv2.imread(str(img_path))  # BGR
        if img is None:
            raise FileNotFoundError(f"Could not read image {img_path}")

        h, w, _ = img.shape

        # Get label (in original pixel coordinates)
        lab = self.labels_dict[frame_idx]
        x_pix = lab["x"]
        y_pix = lab["y"]

        # Normalize to [0, 1] in original resolution
        x_norm = x_pix / w
        y_norm = y_pix / h
        target = np.array([x_norm, y_norm], dtype=np.float32)

        # Resize image to image_size x image_size
        img_resized = cv2.resize(img, (self.image_size, self.image_size))

        # Convert BGR to RGB
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        # Scale to [0,1] and CHW
        img_resized = img_resized.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1)  # (3, H, W)

        target_tensor = torch.from_numpy(target)  # (2,)

        return img_tensor, target_tensor, frame_idx  # include frame_idx for debugging


# Instantiate datasets and loaders
train_dataset = HeelDataset(train_indices, FRAMES_DIR, labels, image_size=IMAGE_SIZE)
val_dataset = HeelDataset(val_indices, FRAMES_DIR, labels, image_size=IMAGE_SIZE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Dataset and DataLoaders ready.")
