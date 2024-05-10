from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from qualisys.qualisys_marker_preprocessing.synced_qualisys_tsv_reformatting import reformat_synced_qualisys_tsv_data_from_folder

def create_freemocap_unix_timestamps(csv_path):
    df = pd.read_csv(csv_path)
    df.replace(-1, float('nan'), inplace=True)
    mean_timestamps = df.iloc[:, 2:].mean(axis=1, skipna=True)
    time_diff = np.diff(mean_timestamps)
    framerate = 1 / np.nanmean(time_diff)
    return mean_timestamps, framerate

def strip_qualisys_tsv(tsv_path, header_line_count):
    original_df = pd.read_csv(tsv_path, skiprows=header_line_count, delimiter="\t")
    with open(tsv_path, 'r') as f:
        header = [next(f).strip().split('\t') for _ in range(header_line_count)]
    header_dict = {item[0].lower(): item[1:] for item in header}
    return original_df, header_dict

def insert_qualisys_timestamp_column(df, start_timestamp, lag_in_seconds=0):
    """
    Insert a new column with Unix timestamps to the Qualisys dataframe.
    
    Parameters:
        df (pd.DataFrame): The original Qualisys dataframe with a 'Time' column in seconds.
        start_timestamp (str): The Qualisys start time as a string in the format '%Y-%m-%d, %H:%M:%S.%f'.
        lag_in_seconds (float, optional): The lag between Qualisys and FreeMoCap data in seconds. Default is 0.
        
    Returns:
        pd.DataFrame: The modified Qualisys dataframe with a new 'unix_timestamps' column.
    """
    start_time = datetime.strptime(start_timestamp[0], '%Y-%m-%d, %H:%M:%S.%f')
    start_unix = start_time.timestamp()
    
    # Adjust the 'Time' column based on the calculated lag in seconds
    adjusted_time = df['Time'] + lag_in_seconds
    
    # Insert the new column with Unix timestamps
    df.insert(df.columns.get_loc('Time') + 1, 'unix_timestamps', adjusted_time + start_unix)
    
    return df

def synchronize_qualisys_data(qualisys_df, freemocap_timestamps):
    synchronized_rows = {}
    for frame_number, timestamp in enumerate(freemocap_timestamps):
        if frame_number + 1 < len(freemocap_timestamps):
            next_timestamp = freemocap_timestamps[frame_number + 1]
            rows_in_range = qualisys_df.loc[(qualisys_df['unix_timestamps'] >= timestamp) & (qualisys_df['unix_timestamps'] < next_timestamp)]
            mean_row = rows_in_range.mean(axis=0, skipna=True)
        else:
            rows_in_range = qualisys_df.loc[(qualisys_df['unix_timestamps'] >= timestamp)]
            mean_row = rows_in_range.iloc[0]
        synchronized_rows[frame_number] = mean_row
    return pd.DataFrame.from_dict(synchronized_rows, orient='index', columns=qualisys_df.columns)


def normalize(signal: pd.Series) -> pd.Series:
    """
    Normalize a signal to have zero mean and unit variance.
    
    Parameters:
        signal (pd.Series): The signal to normalize.

    Returns:
        pd.Series: The normalized signal.
    """
    return (signal - signal.mean()) / signal.std()


def calculate_optimal_lag(freemocap_data: pd.Series, qualisys_data: pd.Series) -> int:
    """
    Calculate the optimal lag between FreeMoCap and Qualisys data using cross-correlation.

    Parameters:
        freemocap_data (pd.Series): The FreeMoCap data series to compare.
        qualisys_data (pd.Series): The Qualisys data series to compare.

    Returns:
        int: The optimal lag between the two data series.
    """
    # Ensure the two signals are of the same length (trimming the longer one if necessary)
    min_length = min(len(freemocap_data), len(qualisys_data))
    freemocap_data = freemocap_data[:min_length]
    qualisys_data = qualisys_data[:min_length]


    normalized_freemocap = normalize(freemocap_data)
    normalized_qualisys = normalize(qualisys_data)

    # Compute the cross-correlation
    cross_corr = np.correlate(normalized_freemocap, normalized_qualisys, mode='full')

    # Find the lag that maximizes the cross-correlation
    optimal_lag = np.argmax(cross_corr) - (len(normalized_freemocap) - 1)
    print(f"The optimal lag is: {optimal_lag}")

    return optimal_lag

def convert_lag_from_frames_to_seconds(lag_frames: int, framerate: float) -> float:
    """
    Convert a lag from frames to seconds.

    Parameters:
        lag_frames (int): The lag in frames.
        framerate (float): The framerate of the data.

    Returns:
        float: The lag in seconds.
    """
    return lag_frames / framerate

def plot_shifted_signals(freemocap_data: pd.Series, qualisys_data: pd.Series, optimal_lag: int):
    """
    Plot the original and shifted signals to visualize the synchronization.
    
    Parameters:
        freemocap_data (pd.Series): The FreeMoCap data series.
        qualisys_data (pd.Series): The Qualisys data series.
        optimal_lag (int): The optimal lag for synchronization.
    """
    # Normalize the signals
    normalized_freemocap = normalize(freemocap_data)
    normalized_qualisys = normalize(qualisys_data)

    # Shift the extended Qualisys data by the optimal lag
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

# Test the functions

# recording_folder_path = Path(r'D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_13_48_44_MDN_treadmill_2')
# freemocap_csv_path = recording_folder_path / 'synchronized_videos' / 'timestamps' / 'unix_synced_timestamps.csv'
# qualisys_tsv_path = recording_folder_path / 'qualisys' / 'MDN_treadmill_2_tracked.tsv'
# freemocap_body_csv = recording_folder_path / 'output_data' / 'mediapipe_body_3d_xyz.csv'
# header_line_count = 12


# recording_folder_path = Path(r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\mediapipe_MDN_Trial_2_yolo")
recording_folder_path = Path(r"D:\2024-04-25_P01\1.0_recordings\sesh_2024-04-25_14_45_59_P01_NIH_Trial1")
# freemocap_csv_path = recording_folder_path / 'synchronized_videos' / 'timestamps' / 'unix_synced_timestamps.csv'
freemocap_csv_path = recording_folder_path / 'synchronized_videos' / 'unix_synced_timestamps.csv'
qualisys_tsv_path = recording_folder_path / 'qualisys_data' / 'qualisys_exported_markers.tsv'
freemocap_body_csv = recording_folder_path / 'output_data' / 'mediapipe_body_3d_xyz.csv'
header_line_count = 11
synced_tsv_name = 'synchronized_qualisys_markers.tsv'

# freemocap_csv_path = Path(
#     r"D:\2023-06-07_TF01\1.0_recordings\treadmill_calib\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1\synchronized_videos\timestamps\unix_synced_timestamps.csv")

# qualisys_tsv_path = Path(r"D:\2023-06-07_TF01\1.0_recordings\treadmill_calib\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1\qualisys\flexion_neutral_trial_1_tracked_with_header.tsv")

# freemocap_body_csv = Path(
#     r"D:\2023-06-07_TF01\1.0_recordings\treadmill_calib\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1\output_data\mediapipe_body_3d_xyz.csv")


freemocap_timestamps, framerate = create_freemocap_unix_timestamps(freemocap_csv_path)
# framerate= 29.97653535971666
print(f"Calculated FreeMoCap framerate: {framerate}")

qualisys_df, header_dict = strip_qualisys_tsv(qualisys_tsv_path, header_line_count=header_line_count)
qualisys_start_timestamp = header_dict["time_stamp"]
qualisys_df_with_unix = insert_qualisys_timestamp_column(qualisys_df.copy(), qualisys_start_timestamp, lag_in_seconds=0)

synchronized_qualisys_df = synchronize_qualisys_data(qualisys_df_with_unix, freemocap_timestamps)
print(synchronized_qualisys_df.head())

freemocap_body_df = pd.read_csv(freemocap_body_csv)
freemocap_data =  freemocap_body_df['left_shoulder_y']
qualisys_data = synchronized_qualisys_df['LFrontShoulder Y'] 

optimal_lag = calculate_optimal_lag(freemocap_data[2000:4000], qualisys_data[2000:4000])
optimal_lag = 3

print(f"Optimal lag: {optimal_lag}")

plot_shifted_signals(freemocap_data, qualisys_data, optimal_lag)

lag_seconds = convert_lag_from_frames_to_seconds(optimal_lag, framerate)

print(f"Calculated lag in seconds: {lag_seconds}")

qualisys_df_with_unix_lag_corrected = insert_qualisys_timestamp_column(qualisys_df, qualisys_start_timestamp, lag_in_seconds=lag_seconds)

synchronized_qualisys_df = synchronize_qualisys_data(qualisys_df_with_unix_lag_corrected, freemocap_timestamps)

qualisys_data = synchronized_qualisys_df['LTOE Y'] 

optimal_lag = calculate_optimal_lag(freemocap_data[:], qualisys_data[:])

optimal_lag = 3

print(f"Optimal lag: {optimal_lag}")

plot_shifted_signals(freemocap_data, qualisys_data, optimal_lag)

assert synchronized_qualisys_df.shape[1] == qualisys_df.shape[
    1], "qualisys_synchronized_df does not have the same number of columns as qualisys_original_df"
assert synchronized_qualisys_df.shape[0] == len(
    freemocap_timestamps), "qualisys_synchronized_df does not have the same number of rows as freemocap_timestamps"

synchronized_qualisys_df.to_csv(recording_folder_path/'qualisys_data'/synced_tsv_name, sep="\t", index=False)
# synchronized_qualisys_df.to_csv(Path(r"D:\2023-06-07_TF01\1.0_recordings\treadmill_calib\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1\qualisys\flexion_neutral_trial_1_tracked_with_header_synchronized.tsv"),
#                                 sep="\t", index=False)

reformat_synced_qualisys_tsv_data_from_folder(recording_folder_path, synchronized_qualisys_df)
print('Saved synced TSV and reformatted CSV')
f = 2   