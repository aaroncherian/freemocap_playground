import pandas as pd
from pathlib import Path
from whittaker_eilers import WhittakerSmoother
import numpy as np
from scipy.signal import savgol_filter
from scipy.signal import butter, filtfilt


# Load the DLC CSV (skip the first two rows, which hold labels)
path_to_csv = Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_09_05_TF01_flexion_pos_2_8_trial_1\output_data\dlc\model_8\sesh_2023-06-07_12_09_05_TF01_flexion_pos_2_8_trial_1_synced_Cam5DLC_Resnet50_leg_dlc_6_shuffle1_snapshot_100_filtered.csv")

df = pd.read_csv(path_to_csv, header=[0, 1, 2])  # multi-index columns (scorer, bodypart, coord)
df = df.drop(index=[0,1])  # remove redundant rows seen earlier if necessary

# Get the right ankle columns
ankle_x = df[('DLC_Resnet50_leg_dlc_6_shuffle1_snapshot_100', 'right_ankle', 'x')].astype(float)
ankle_y = df[('DLC_Resnet50_leg_dlc_6_shuffle1_snapshot_100', 'right_ankle', 'y')].astype(float)
ankle_w = df[('DLC_Resnet50_leg_dlc_6_shuffle1_snapshot_100', 'right_ankle', 'likelihood')].astype(float)

# Combine into a single DataFrame
ankle_df = pd.DataFrame({'x': ankle_x, 'y': ankle_y, 'weight': ankle_w})

x = np.array(ankle_df['x'])
y = np.array(ankle_df['y'])
w = np.array(ankle_df['weight'])

whittaker_smoother = WhittakerSmoother(
    lmbda = 10,
    order = 2,
    data_length = len(ankle_df),
    weights = w.tolist()
)



whittaker_x = whittaker_smoother.smooth(x.tolist())
whittaker_y = whittaker_smoother.smooth(y.tolist())

savgol_x = savgol_filter(x, window_length=11, polyorder=2)
savgol_y = savgol_filter(y, window_length=11, polyorder=2)

# Define Butterworth filter parameters
order = 4
cutoff_hz = 7  # 7 Hz cutoff frequency
fs = 30  # Sampling frequency in Hz (change if your data has a different frame rate)

# Design Butterworth filter
b, a = butter(order, cutoff_hz / (0.5 * fs), btype='low')

# Apply filter to x and y
butter_x = filtfilt(b, a, x)
butter_y = filtfilt(b, a, y)

ankle_df['butterworth_x'] = butter_x
ankle_df['butterworth_y'] = butter_y

ankle_df['whittaker_x'] = whittaker_x
ankle_df['whittaker_y'] = whittaker_y

ankle_df['savgol_x'] = savgol_x
ankle_df['savgol_y'] = savgol_y

import matplotlib.pyplot as plt

# Create a figure with two subplots (one for x, one for y)
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# Plot X vs Smoothed X
axes[0].plot(ankle_df.index, ankle_df['x'], label='Raw X', color='black')
axes[0].plot(ankle_df.index, ankle_df['butterworth_x'], label='Butterworth X', linewidth=2, alpha = .5)
axes[0].plot(ankle_df.index, ankle_df['whittaker_x'], label='Whittaker X', linewidth=2, alpha = .5)
axes[0].plot(ankle_df.index, ankle_df['savgol_x'], label='Savitzky-Golay X', linewidth=2, alpha = .5)
axes[0].set_ylabel('X Position (pixels)')
axes[0].set_title('Right Ankle X Coordinate')
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.4)

# Plot Y vs Smoothed Y
axes[1].plot(ankle_df.index, ankle_df['y'], label='Raw Y', color='black')
axes[1].plot(ankle_df.index, ankle_df['butterworth_y'], label='Butterworth Y', linewidth=2, alpha = .5)
axes[1].plot(ankle_df.index, ankle_df['whittaker_y'], label='Whittaker Y', linewidth=2, alpha = .5)
axes[1].plot(ankle_df.index, ankle_df['savgol_y'], label='Savitzky-Golay Y', linewidth=2, alpha = .5)
axes[1].set_ylabel('Y Position (pixels)')
axes[1].set_xlabel('Frame')
axes[1].set_title('Right Ankle Y Coordinate')
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()