import cv2
from pathlib import Path

video_path = Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1\synchronized_videos\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1_synced_Cam6.mp4")
output_dir = Path("new_training/data/frames")
output_dir.mkdir(parents=True, exist_ok=True)


cap = cv2.VideoCapture(str(video_path))
i = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite(str(output_dir / f"frame_{i:05d}.png"), frame)
    i += 1
cap.release()
f = 2