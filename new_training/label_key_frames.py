# label_keyframes.py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import json

frames_dir = Path("new_training/data/frames")
label_path = Path("new_training/data/heel_labels_keyframes.json")


start_frame = 0
end_frame = 1400
step = 50
keyframe_indices = list(range(start_frame, end_frame, step))

labels = {}
for idx in keyframe_indices:
    img_path = frames_dir / f"frame_{idx:05d}.png"
    img = mpimg.imread(img_path)

    plt.imshow(img)
    plt.title(f"Click heel for frame {idx}")
    xy = plt.ginput(1)[0]
    plt.close()

    labels[idx] = {"x": xy[0], "y": xy[1]}

label_path.write_text(json.dumps(labels, indent=2))
print("Saved keyframe labels for indices:", keyframe_indices)