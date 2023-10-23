

import numpy as np
import pandas as pd

def calculate_framerate(timestamps):
    """
    Calculate the approximate framerate based on a series of timestamps.
    
    Parameters:
        timestamps (numpy array or pandas Series): An array of timestamps in seconds.
        
    Returns:
        float: Approximate framerate in frames per second.
    """
    # Calculate time differences between consecutive timestamps
    time_diffs = np.diff(timestamps)
    
    # Calculate the average time difference in seconds
    avg_time_diff = np.mean(time_diffs)
    
    # Calculate the framerate as the inverse of the average time difference
    framerate = 1 / avg_time_diff if avg_time_diff != 0 else 0
    
    return framerate

# Load your FreeMoCap timestamps CSV
csv_file = r"D:\2023-06-07_TF01\1.0_recordings\treadmill_calib\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1\synchronized_videos\timestamps\unix_synced_timestamps.csv"  # Replace with the path to your CSV file
df = pd.read_csv(csv_file)

# Replace -1 with NaN in the DataFrame
df.replace(-1, float('nan'), inplace=True)

# Calculate the mean timestamp for each frame, skipping NaN values
mean_freemocap_unix = df.iloc[:, 2:].mean(axis=1, skipna=True)

# Calculate the framerate based on the mean timestamps
framerate = calculate_framerate(mean_freemocap_unix)

print(f"The calculated framerate is: {framerate} frames per second.")