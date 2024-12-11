from pathlib import Path
import pandas as pd 
from datetime import datetime
import numpy as np
from rich.progress import track
from joint_center_mappings.full_body_joint_center_weights import joint_center_weights
from run_skellyforge import run_skellyforge_rotation
from skellymodels.model_info.mediapipe_model_info import MediapipeModelInfo

recording_folder_path = Path(r"D:\2024-10-30_treadmill_pilot\processed_data\sesh_2024-10-30_15_45_14_mdn_gait_7_exposure")

freemocap_csv_path = recording_folder_path / 'output_data' / 'unix_synced_timestamps.csv'
freemocap_data_path = recording_folder_path/ 'output_data'/ 'mediapipe_body_3d_xyz.npy'

qualisys_marker_tsv_path = recording_folder_path / 'output_data'/'component_qualisys_original' / 'qualisys_exported_markers.tsv'



def get_header_length(file_path):
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            if line.startswith('TRAJECTORY_TYPES'):  # Detect the specific line for the header's end
                print(f'Header found, skipping {i+1} rows')
                return i + 1  # Data starts right after the marker names row
    print('Header not found')
    return 0  # Default if no header is found

def get_starting_qualisys_timestamp(qualisys_tsv_path):
    with open(qualisys_tsv_path, 'r') as file:
        for line in file:
            if line.startswith('TIME_STAMP'):
                print('Timestamp line found:', line.strip())
                parts = line.strip().split('\t')
                starting_time_stamp = parts[1]

                datatime_time_stamp = datetime.strptime(starting_time_stamp, '%Y-%m-%d, %H:%M:%S.%f')
                unix_time_stamp = datatime_time_stamp.timestamp()
                return unix_time_stamp

                f = 2
        print('Timestamp not found')
        return None
    f = 2

def create_and_insert_unix_timestamp_column(df, start_timestamp, lag_in_seconds=0):
    """
    Insert a new column with Unix timestamps to the Qualisys dataframe.
    
    Parameters:
        df (pd.DataFrame): The original Qualisys dataframe with a 'Time' column in seconds.
        start_timestamp (str): The Qualisys start time as a string in the format '%Y-%m-%d, %H:%M:%S.%f'.
        lag_in_seconds (float, optional): The lag between Qualisys and FreeMoCap data in seconds. Default is 0.
        
    Returns:
        pd.DataFrame: The modified Qualisys dataframe with a new 'unix_timestamps' column.
    """

    # Adjust the 'Time' column based on the calculated lag in seconds
    adjusted_time = df['Time'] + lag_in_seconds
    
    # Insert the new column with Unix timestamps
    df.insert(df.columns.get_loc('Time') + 1, 'unix_timestamps', adjusted_time + start_timestamp)
    
    return df

def create_freemocap_unix_timestamps(csv_path):
    df = pd.read_csv(csv_path)
    df.replace(-1, float('nan'), inplace=True)
    mean_timestamps = df.iloc[:, 2:].mean(axis=1, skipna=True)
    time_diff = np.diff(mean_timestamps)
    framerate = 1 / np.nanmean(time_diff)
    print(f"Calculated FreeMoCap framerate: {framerate}")
    return mean_timestamps, framerate

def resample_qualisys_data(qualisys_df, freemocap_timestamps):
    print('Resampling Qualisys data...')
    synchronized_rows = {}
    for frame_number, timestamp in track(enumerate(freemocap_timestamps)):
        if frame_number + 1 < len(freemocap_timestamps):
            next_timestamp = freemocap_timestamps[frame_number + 1]
            rows_in_range = qualisys_df.loc[(qualisys_df['unix_timestamps'] >= timestamp) & (qualisys_df['unix_timestamps'] < next_timestamp)]
            mean_row = rows_in_range.mean(axis=0, skipna=True)
        else:
            rows_in_range = qualisys_df.loc[(qualisys_df['unix_timestamps'] >= timestamp)]
            mean_row = rows_in_range.iloc[0]
        synchronized_rows[frame_number] = mean_row
    return pd.DataFrame.from_dict(synchronized_rows, orient='index', columns=qualisys_df.columns)

def calculate_joint_centers(array_3d, joint_center_weights, marker_names):
    num_frames, num_markers, _ = array_3d.shape
    num_joints = len(joint_center_weights.keys())
    
    # Initialize an array to hold the joint centers
    joint_centers = np.zeros((num_frames, num_joints, 3))

    # Create a mapping from marker names to indices
    marker_to_index = {marker: i for i, marker in enumerate(marker_names)}
    
    # start_time = time.time()

    # Iterate over frames
    print('Calculating Qualisys joint centers')
    for frame in track(range(num_frames)):
        # if frame % 1000 == 0:
        #     elapsed_time = time.time() - start_time
        #     print(f"Finished frame {frame} of {num_frames}. Elapsed time: {elapsed_time:.2f} seconds")

        # Iterate over joints
        for j_idx, joint in enumerate(joint_center_weights.keys()):
            weighted_positions = []
            for marker, weight in joint_center_weights[joint].items():
                marker_idx = marker_to_index[marker]
                weighted_positions.append(array_3d[frame, marker_idx, :] * weight)
            
            # Sum along the 0-axis to get the joint center for this frame and joint
            joint_centers[frame, j_idx, :] = np.sum(weighted_positions, axis=0)
    
    return joint_centers

header_length = get_header_length(qualisys_marker_tsv_path)
qualisys_markers = pd.read_csv(qualisys_marker_tsv_path, delimiter='\t', skiprows=header_length)
qualisys_unix_start_time = get_starting_qualisys_timestamp(qualisys_marker_tsv_path)
qualisys_markers_with_unix = create_and_insert_unix_timestamp_column(qualisys_markers, qualisys_unix_start_time)

freemocap_timestamps, framerate = create_freemocap_unix_timestamps(freemocap_csv_path)

resampled_qualisys_markers = resample_qualisys_data(qualisys_markers_with_unix, freemocap_timestamps)


marker_data_columns =  resampled_qualisys_markers.columns[~ resampled_qualisys_markers.columns.str.contains(r'^(?:Frame|Time|unix_timestamps|Unnamed)', regex=True)]
marker_names = list(dict.fromkeys(col.split()[0] for col in marker_data_columns))
marker_data_flat = resampled_qualisys_markers[marker_data_columns].to_numpy()

num_frames = len(resampled_qualisys_markers)
num_markers = len(marker_names)
marker_data_formatted = marker_data_flat.reshape(num_frames, num_markers, 3)  # Shape: (frames, markers, dimensions)


qualisys_joint_centers = calculate_joint_centers(array_3d=marker_data_formatted, joint_center_weights=joint_center_weights, marker_names=marker_names)

rotated_qualisys_joint_centers = run_skellyforge_rotation(qualisys_joint_centers, list(joint_center_weights.keys()))


freemocap_joint_centers = np.load(freemocap_data_path)
rotated_freemocap_joint_centers = run_skellyforge_rotation(freemocap_joint_centers, MediapipeModelInfo.landmark_names)


f = 2