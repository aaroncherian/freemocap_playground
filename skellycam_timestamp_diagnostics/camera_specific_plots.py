import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load the timestamp data
path_to_recording = Path(r'D:\2025-03-13_JSM_pilot\freemocap_data\2025-03-13T16_12_14_gmt-4_jsm_pilot_treadmill_one')
time_range = range(3800,3900)


# time_range = None
# path_to_recording = Path(r'D:\2025-03-13_JSM_pilot\freemocap_data\2025-03-13T16_20_37_gmt-4_pilot_jsm_treadmill_walking')
# path_to_recording = Path(r'D:\2025-03-13_JSM_pilot\freemocap_data\2025-03-13T15_58_32_gmt-4_pilot_jsm_nih_trial_one')
timestamp_path = path_to_recording / 'synchronized_videos' / f'{path_to_recording.stem}_timestamps.csv'
timestamp_data = pd.read_csv(timestamp_path)

# Identify cameras dynamically
camera_ids = sorted(set(col.split('_')[1] for col in timestamp_data.columns if col.startswith('camera_')))

# Define the columns for each subplot category
timestamp_stage_cols = [
    'timestamp_ns',
    'initialized_timestamp_ns',
    'pre_grab_timestamp_ns',
    'pre_retrieve_timestamp_ns',
    'post_retrieve_timestamp_ns',
    'copy_to_buffer_timestamp_ns',
    'copy_from_buffer_timestamp_ns',
]

buffer_time_cols = ['time_spent_in_buffer_ns']

# additional_time_cols = [
#     # 'time_spent_grabbing_ns',
#     # 'time_waiting_to_retrieve_ns',
#     # 'time_spent_retrieving_ns',
#     'time_spent_waiting_to_be_put_into_buffer_ns'
# ]

# Convert timestamps from nanoseconds to milliseconds
relative_time = timestamp_data['timestamp_unix_seconds'] - timestamp_data['timestamp_unix_seconds'].iloc[0]

# Range for plotting
time_range = range(3800,3900)
# time_range = None

for camera_id in camera_ids:
    fig, axes = plt.subplots(2, 1, figsize=[15, 15])
    fig.patch.set_facecolor("#F0F0F0")

    # Plot timestamp stages
    for col in timestamp_stage_cols:
        full_col = f'camera_{camera_id}_{col}'
        if full_col in timestamp_data.columns:
            if time_range:
                axes[0].scatter(relative_time[time_range], timestamp_data[full_col][time_range] / 1e6, label=col, s=4, alpha=0.7)
            else:
                axes[0].scatter(relative_time, timestamp_data[full_col] / 1e6, label=col, s=4, alpha=0.7)

    axes[0].set_title(f"Camera {camera_id} - Timestamps (ms)")
    axes[0].set_xlabel('Time since start of recording (s)')
    axes[0].set_ylabel('Time (ms)')
    axes[0].legend()
    axes[0].grid(True)

    # Plot buffer time
    for col in buffer_time_cols:
        full_col = f'camera_{camera_id}_{col}'
        if full_col in timestamp_data.columns:
            if time_range:
                axes[1].scatter(relative_time[time_range], timestamp_data[full_col][time_range] / 1e6, label=col, s=4, alpha=0.7)
            else:
                axes[1].scatter(relative_time, timestamp_data[full_col]/ 1e6, label=col, s=4, alpha=0.7)
    axes[1].set_title(f"Camera {camera_id} - Time Spent in Buffer (ms)")
    axes[1].set_xlabel('Time since start of recording (s)')
    axes[1].set_ylabel('Time (ms)')
    axes[1].legend()
    axes[1].grid(True)

    # # Plot additional time metrics
    # for col in additional_time_cols:
    #     full_col = f'camera_{camera_id}_{col}'
    #     if full_col in timestamp_data.columns:
    #         if time_range:
    #             axes[2].scatter(relative_time[time_range], timestamp_data[full_col][time_range] / 1e6, label=col, s=4, alpha=0.7)
    #         else:
    #             axes[2].scatter(relative_time, timestamp_data[full_col] / 1e6, label=col, s=4, alpha=0.7)

    # axes[2].set_title(f"Camera {camera_id} - More metrics (ms)")
    # axes[2].set_xlabel('Time since start of recording (s)')
    # axes[2].set_ylabel('Time (ms)')
    # axes[2].legend()
    # axes[2].grid(True)

    plt.tight_layout()
    plt.show()
