import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load the timestamp data
path_to_recording = Path(r'D:\2025-03-13_JSM_pilot\freemocap_data\2025-03-13T15_58_32_gmt-4_pilot_jsm_nih_trial_one')
path_to_recording = Path(r'D:\2025-03-13_JSM_pilot\freemocap_data\2025-03-13T16_20_37_gmt-4_pilot_jsm_treadmill_walking')
path_to_recording = Path(r'D:\2025-03-13_JSM_pilot\freemocap_data\2025-03-13T16_12_14_gmt-4_jsm_pilot_treadmill_one')



timestamp_path = path_to_recording/'synchronized_videos'/f'{path_to_recording.stem}_timestamps.csv'
timestamp_data = pd.read_csv(timestamp_path)

# Extract Unix timestamps
timeStampData = timestamp_data['timestamp_unix_seconds']
relative_time = timeStampData - timeStampData.iloc[0]

# Calculate differences between frames and FPS
differenceFrame = timeStampData.diff().dropna()
fpsFrame = 1 / differenceFrame

# FPS drift calculation (from ideal 30 FPS)
ideal_fps = 30
fps_drift = fpsFrame - ideal_fps

# Plot diagnostic figures
fig, axes = plt.subplots(2, 2, figsize=[15, 16])
fig.suptitle('Timestamp Diagnostics - Walk to Run')
fig.patch.set_facecolor("#F0F0F0")

# Camera timestamps (relative)
axes[0, 0].plot(relative_time, marker='.', linestyle='-', markersize=4)
axes[0, 0].set_xlabel("Frame")
axes[0, 0].set_ylabel("Time since start (s)")
axes[0, 0].set_title("Camera Timestamps (Relative)")

# Frame duration differences
axes[0, 1].plot(relative_time.iloc[1:], differenceFrame * 1000, marker=".", linestyle="none")
axes[0, 1].set_xlabel("Time since start (s)")
axes[0, 1].set_ylabel("Frame Duration (ms)")
axes[0, 1].set_title("Camera Frame Duration")

# Histogram of frame intervals
# axes[1, 0].hist(differenceFrame * 1000, bins=20, alpha=0.7)
# axes[1, 0].set_xlabel("Frame Interval (ms)")
# axes[1, 0].set_ylabel("Frequency")
# axes[1, 0].set_title("Frame Interval Distribution")

axes[1, 0].hist(differenceFrame * 1000, bins=50, alpha=0.7, log=True)
axes[1, 0].set_xlabel("Frame Interval (ms)")
axes[1, 0].set_ylabel("Log Frequency")
axes[1, 0].set_title("Frame Interval Distribution (Log scale)")

# Histogram of FPS distribution
axes[1, 1].hist(fpsFrame, bins=20, alpha=0.7)
axes[1, 1].set_xlabel("Frames per Second (FPS)")
axes[1, 1].set_ylabel("Frequency")
axes[1, 1].set_title("Frames per Second Distribution")



plt.tight_layout()
plt.show()
