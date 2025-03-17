import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load the timestamp data
path_to_recording = Path(r'D:\2025-03-13_JSM_pilot\freemocap_data\2025-03-13T16_12_14_gmt-4_jsm_pilot_treadmill_one')
# path_to_recording = Path(r'D:\2025-03-13_JSM_pilot\freemocap_data\2025-03-13T16_20_37_gmt-4_pilot_jsm_treadmill_walking')

timestamp_path = path_to_recording / 'synchronized_videos' / f'{path_to_recording.stem}_timestamps.csv'
timestamp_data = pd.read_csv(timestamp_path)

# Identify camera IDs dynamically
camera_ids = sorted(set(col.split('_')[1] for col in timestamp_data.columns if col.startswith('camera_')))

fig, axes = plt.subplots(3, 1, figsize=[15, 13])
fig.suptitle('Camera Comparison Diagnostics - Walk to Run')
fig.patch.set_facecolor("#F0F0F0")

# Plot timestamps per frame
for cam in camera_ids:
    col = f'camera_{cam}_timestamp_ns'
    if col in timestamp_data.columns:
        axes[0].scatter(timestamp_data.index, timestamp_data[col] / 1e6, label=f'Camera {cam}', s=4, alpha=.5)

axes[0].set_title("Camera Timestamps per Frame")
axes[0].set_xlabel('Frame Number')
axes[0].set_ylabel('Timestamp (ms)')
axes[0].legend()
axes[0].grid(True)

# Mean frame duration per camera
mean_frame_durations = {}
for cam in camera_ids:
    col = f'camera_{cam}_timestamp_ns'
    if col in timestamp_data.columns:
        frame_diff = timestamp_data[col].diff().dropna() / 1e6
        mean_duration_ms = frame_diff.mean() 
        mean_frame_durations[cam] = mean_duration_ms

# Plotting
bars = axes[2].bar(mean_frame_durations.keys(), mean_frame_durations.values(), color='skyblue')

# Zoom in on top part
min_duration = min(mean_frame_durations.values())
axes[2].set_ylim(min_duration - 0.5, min_duration + 0.5)

for bar, cam_id in zip(bars, mean_frame_durations.keys()):
    height = bar.get_height()
    axes[2].text(
        bar.get_x() + bar.get_width() / 2, height,
        f'{mean_frame_durations[cam_id]}',  # unrounded exact value
        ha='center', va='bottom', fontsize=8, 
    )

axes[2].set_title("Mean Frame Duration per Camera (ms)")
axes[2].set_xlabel("Camera ID")
axes[2].set_ylabel("Mean Frame Duration (ms)")
axes[2].grid(axis='y')

# Frame duration per frame
for cam in camera_ids:
    col = f'camera_{cam}_timestamp_ns'
    if col in timestamp_data.columns:
        frame_diff = timestamp_data[col].diff().dropna() / 1e6
        axes[1].scatter(timestamp_data.index[1:], frame_diff, label=f'Camera {cam}', s=4, alpha=0.7)

axes[1].set_title("Frame Duration per Camera (ms)")
axes[1].set_xlabel("Frame Number")
axes[1].set_ylabel("Frame Duration (ms)")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
