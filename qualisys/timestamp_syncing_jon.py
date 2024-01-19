# %%
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rich import print

# %%
# Load FreeMoCap timestamps

freemocap_unix_timestamps_csv = Path(
    r"D:\2023-06-07_TF01\1.0_recordings\treadmill_calib\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1\synchronized_videos\timestamps\unix_synced_timestamps.csv")

freemocap_unix_timestamps_df = pd.read_csv(freemocap_unix_timestamps_csv)
freemocap_unix_timestamps_df.replace(-1, float('nan'), inplace=True)

# Calculate mean along each row, ignoring the first column
mean_freemocap_unix = freemocap_unix_timestamps_df.iloc[:, 2:].mean(axis=1, skipna=True)
range_freemocap_unix = freemocap_unix_timestamps_df.iloc[:, 2:].max(axis=1) - freemocap_unix_timestamps_df.iloc[:,
                                                                              2:].min(axis=1)
std_freemocap_unix = freemocap_unix_timestamps_df.iloc[:, 2:].std(axis=1)
first_timestamp = mean_freemocap_unix.iloc[0]

# # Given framerate
# given_framerate = 29.967667127642223
# time_interval = 1 / given_framerate

# # Generate a new list of timestamps based on the first timestamp and the given framerate
# num_frames = len(mean_freemocap_unix)
# freemocap_timestamps = [first_timestamp + i * time_interval for i in range(num_frames)]


freemocap_unix_timestamps_df = pd.concat([mean_freemocap_unix, range_freemocap_unix, std_freemocap_unix], axis=1)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
freemocap_unix_timestamps_df
# %%
freemocap_timestamps = mean_freemocap_unix.to_list()

# Give Series a name
mean_freemocap_unix.name = 'mean_timestamp'
range_freemocap_unix.name = 'range_timestamp'
std_freemocap_unix.name = 'std_timestamp'

# Concatenate
freemocap_unix_timestamps_df = pd.concat([mean_freemocap_unix, range_freemocap_unix, std_freemocap_unix], axis=1)

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
freemocap_unix_timestamps_df
# %%
# freemocap_timestamps = mean_freemocap_unix.to_list()

# %%
qualisys_tsv_path = Path(r"D:\2023-06-07_TF01\1.0_recordings\treadmill_calib\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1\qualisys\flexion_neutral_trial_1_tracked_with_header.tsv")

# %%
# Load qualisys tsv as dataframe (skipping header)
qualisys_header_line_count = 11  # should be constant for all Qualisys output? or does it depend on 'export' settings?

qualisys_original_df = pd.read_csv(qualisys_tsv_path,
                                   skiprows=qualisys_header_line_count,
                                   delimiter="\t")

qualisys_original_df

# %%
# Get the header data out


# Open the file first and read the headers
with open(qualisys_tsv_path, 'r') as f:
    header = [next(f).strip().split('\t') for _ in range(qualisys_header_line_count)]

# print(header)
# Convert to a dictionary
header_dict = {item[0].lower(): item[1:] for item in header}
# print(header_dict)


# %%
# Convert Qualisys start time to Unix

qualisys_start_timestamp = header_dict["time_stamp"]

qualisys_start = datetime.strptime(qualisys_start_timestamp[0], '%Y-%m-%d, %H:%M:%S.%f')
qualisys_start_unix = qualisys_start.timestamp()
print(f"Original - {qualisys_start_timestamp}, Unix - {qualisys_start_unix}")

# %%
# Create new column showing Qualisys Unix Time and Insert the new column right after 'timestamps_from_zero'
qualisys_original_df.insert(qualisys_original_df.columns.get_loc('Time') + 1, 'unix_timestamps',
                            qualisys_original_df['Time'] + qualisys_start_unix)

# %%
# for each frame (row) in the `mean_freemocap_unix`` timestamps, find the frames(rows) of the qualisys data occurring in the time between that timestamp and the next one and calculate the mean of each column within that range. Then, make a new dataframe called `qualisys_synchronized` with the same number of rows as `freemocap` and the same columns as `qualisys` and fill it with the calculated means.

synchronized_qualisys_rows = {}

for frame_number, freemocap_timestamp in enumerate(freemocap_timestamps):
    # get the timestamp of the next frame
    if frame_number + 1 < len(freemocap_timestamps):
        next_timestamp = freemocap_timestamps[frame_number + 1]

        rows_in_range = qualisys_original_df.loc[(qualisys_original_df['unix_timestamps'] >= freemocap_timestamp) & (
                qualisys_original_df['unix_timestamps'] < next_timestamp)]
        mean_row = rows_in_range.mean(axis=0, skipna=True)
    else:
        rows_in_range = qualisys_original_df.loc[(qualisys_original_df['unix_timestamps'] >= freemocap_timestamp) & (
                qualisys_original_df['unix_timestamps'] < freemocap_timestamp + 100)]
        # just grab the first frame as the '[mean]' row on the last freemocap frame
        mean_row = rows_in_range.iloc[0]
    synchronized_qualisys_rows[frame_number] = mean_row

# create a new dataframe from the qualisys_synchronized_rows list, with the same columns as the original qualisys dataframe
qualisys_synchronized_df = pd.DataFrame.from_dict(synchronized_qualisys_rows, orient='index',
                                                  columns=qualisys_original_df.columns)

# %%
# ensure that the sycnhronized qualisys data frame has the same number of columns as the original qualisys dataframe and the same number of rows as the freemocap data
assert qualisys_synchronized_df.shape[1] == qualisys_original_df.shape[
    1], "qualisys_synchronized_df does not have the same number of columns as qualisys_original_df"
assert qualisys_synchronized_df.shape[0] == len(
    freemocap_timestamps), "qualisys_synchronized_df does not have the same number of rows as freemocap_timestamps"

# print the number of rows and columns of the new dataframe vs the original vs the number of freemocap timestamps
print(
    f"qualisys_synchronized: {qualisys_synchronized_df.shape}, qualisys_df: {qualisys_original_df.shape}, mean_freemocap_unix: {mean_freemocap_unix.shape}")

# %%
# Save the new dataframe as a tsv
qualisys_synchronized_df.to_csv(Path(r"D:\2023-06-07_TF01\1.0_recordings\treadmill_calib\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1\qualisys\flexion_neutral_trial_1_tracked_with_header_synchronized.tsv"),
                                sep="\t", index=False)

# %%
# Plot  timeseries of the freemocap_right_foot_index against the `RTOE X` column of the qualisys data befor and after synchronization

freemocap_body_csv = Path(
    r"D:\2023-06-07_TF01\1.0_recordings\treadmill_calib\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1\output_data\mediapipe_body_3d_xyz.csv")

freemocap_body_df = pd.read_csv(freemocap_body_csv)
freemocap_right_toe_x = freemocap_body_df['right_foot_index_x']
freemocap_right_toe_y = freemocap_body_df['right_foot_index_y']
freemocap_right_toe_z = freemocap_body_df['right_foot_index_z']

original_qualisys_right_toe_x = qualisys_original_df['RTOE X']
original_qualisys_right_toe_y = qualisys_original_df['RTOE Y']
original_qualisys_right_toe_z = qualisys_original_df['RTOE Z']

synchronized_qualisys_right_toe_x = qualisys_synchronized_df['RTOE X']
synchronized_qualisys_right_toe_y = qualisys_synchronized_df['RTOE Y']
synchronized_qualisys_right_toe_z = qualisys_synchronized_df['RTOE Z']


# Creating dataframes for freemocap, original and synchronized qualisys data
freemocap_df_to_plot = pd.DataFrame({
    'x': freemocap_right_toe_x,
    'y': freemocap_right_toe_y,
    'z': freemocap_right_toe_z
})

original_qualisys_df_to_plot = pd.DataFrame({
    'x': original_qualisys_right_toe_x,
    'y': original_qualisys_right_toe_y,
    'z': original_qualisys_right_toe_z
})

synchronized_qualisys_df_to_plot = pd.DataFrame({
    'x': synchronized_qualisys_right_toe_x,
    'y': synchronized_qualisys_right_toe_y,
    'z': synchronized_qualisys_right_toe_z
})


import matplotlib.pyplot as plt
import numpy as np

# Generate some sample data for illustration
# In your actual code, these would come from your dataframes
freemocap_data =  freemocap_body_df['right_foot_index_z']  # Replace this with freemocap_body_df['right_foot_index_x']
qualisys_data = qualisys_synchronized_df['RTOE Z']  # Replace this with qualisys_synchronized_df['RTOE X']

# Ensure the two signals are of the same length (trimming the longer one if necessary)
min_length = min(len(freemocap_data), len(qualisys_data))
freemocap_data = freemocap_data[:min_length]
qualisys_data = qualisys_data[:min_length]

# Normalize the signals
def normalize(signal):
    return (signal - np.mean(signal)) / np.std(signal)

normalized_freemocap = normalize(freemocap_data)
normalized_qualisys = normalize(qualisys_data)

# Compute the cross-correlation
cross_corr = np.correlate(normalized_freemocap, normalized_qualisys, mode='full')

# Find the lag that maximizes the cross-correlation
optimal_lag = np.argmax(cross_corr) - (len(normalized_freemocap) - 1)
print(f"The optimal lag is: {optimal_lag}")

# Shift the qualisys data by the optimal lag
if optimal_lag > 0:
    shifted_qualisys = np.concatenate([np.zeros(optimal_lag), normalized_qualisys[:-optimal_lag]])
else:
    shifted_qualisys = np.concatenate([normalized_qualisys[-optimal_lag:], np.zeros(-optimal_lag)])

# Plot the original and shifted signals
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.title('Before Shift')
plt.plot(normalized_freemocap, label='FreeMoCap Data')
plt.plot(normalized_qualisys, label='Qualisys Data')
plt.legend()

plt.subplot(1, 2, 2)
plt.title('After Shift')
plt.plot(normalized_freemocap, label='FreeMoCap Data')
plt.plot(shifted_qualisys, label=f'Qualisys Data (Shifted by {optimal_lag} frames)')
plt.legend()

plt.show()

# # make a 2,1 subplot with the original timeseries on top and the synchronized timeseries on the bottom
# fig = make_subplots(rows=2, cols=1, subplot_titles=("Freemocap vs Original Qualisys", "Freemocap vs Synchronized Qualisys"))
# fig.update_xaxes(title_text="Frame Number", row=1, col=1)
# fig.update_yaxes(title_text="mm", row=1, col=1)

# fig.update_xaxes(title_text="Frame Number", row=2, col=1)
# fig.update_yaxes(title_text="mm", row=2, col=1)


# freemocap_colors = {'x': 'red', 'y': 'green', 'z': 'blue'}
# original_qualisys_colors = {'x': 'orange', 'y': 'turquoise', 'z': 'purple'}
# synchronized_qualisys_colors = {'x': 'pink', 'y': 'cyan', 'z': 'darkmagenta'}


# # Loop through each coordinate and add traces to subplots
# for coord in ['x', 'y', 'z']:
#     fig.add_trace(go.Scatter(x=freemocap_df_to_plot.index,
#                              y=freemocap_df_to_plot[coord],
#                              mode='lines+markers',
#                              name=f'freemocap right toe {coord}',
#                              line=dict(color=freemocap_colors[coord]),
#                              ),

#                   row=1,
#                   col=1)
#     fig.add_trace(go.Scatter(x=original_qualisys_df_to_plot.index,
#                              y=original_qualisys_df_to_plot[coord],
#                              mode='lines+markers',
#                              name=f'original qualisys right toe {coord}',
#                              line=dict(color=original_qualisys_colors[coord]),
#                              ),
#                   row=1,
#                   col=1)

#     fig.add_trace(go.Scatter(x=freemocap_df_to_plot.index,
#                              y=freemocap_df_to_plot[coord],
#                              mode='lines+markers',
#                              name=f'freemocap right toe {coord}',
#                              line=dict(color=freemocap_colors[coord]),
#                              ),
#                   row=2,
#                   col=1)

#     fig.add_trace(go.Scatter(x=synchronized_qualisys_df_to_plot.index,
#                              y=synchronized_qualisys_df_to_plot[coord],
#                              mode='lines+markers',
#                              name=f'synchronized qualisys right toe {coord}',
#                              line=dict(color=synchronized_qualisys_colors[coord]),
#                              ),
#                   row=2,
#                   col=1)

# fig.show()


# %%
