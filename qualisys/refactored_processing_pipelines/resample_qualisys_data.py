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
    """
    Resample Qualisys data to match FreeMoCap timestamps using bin averaging.
    
    Parameters:
    -----------
    qualisys_df : pandas.DataFrame
        DataFrame with Frame, Time, unix_timestamps and data columns
    freemocap_timestamps : array-like
        Target timestamps to resample to
        
    Returns:
    --------
    pandas.DataFrame
        Resampled data matching freemocap timestamps
    """
    print('Resampling Qualisys data...')
    
    if isinstance(freemocap_timestamps, pd.Series):
        freemocap_timestamps = freemocap_timestamps.to_numpy()
    
    # Create bins from timestamps
    bins = np.append(freemocap_timestamps, freemocap_timestamps[-1] + 
                    (freemocap_timestamps[-1] - freemocap_timestamps[-2]))
    
    # Assign each row to a bin (-1 means it's after the last timestamp)
    qualisys_df['bin'] = pd.cut(qualisys_df['unix_timestamps'], 
                               bins=bins, 
                               labels=range(len(freemocap_timestamps)),
                               include_lowest=True)
    
    # Group by bin and calculate mean
    # Note: dropna=False keeps bins that might be empty
    resampled = qualisys_df.groupby('bin', observed=True).mean(numeric_only=True)
    
    # Handle the last timestamp like the original
    if resampled.index[-1] == len(freemocap_timestamps) - 1:
        last_timestamp = freemocap_timestamps[-1]
        last_frame_data = qualisys_df[qualisys_df['unix_timestamps'] >= last_timestamp].iloc[0]
        resampled.iloc[-1] = last_frame_data[resampled.columns]
    
    resampled = resampled.reset_index(drop=True)
    
    return resampled

import numpy as np

def calculate_joint_centers(array_3d, joint_center_weights, marker_names):
    """
    Optimized calculation of joint centers for Qualisys data with 3D weights.

    Parameters:
        array_3d (np.ndarray): Shape (num_frames, num_markers, 3), 3D marker data.
        joint_center_weights (dict): Weights for each joint as {joint_name: {marker_name: [weight_x, weight_y, weight_z]}}.
        marker_names (list): List of marker names corresponding to array_3d.

    Returns:
        np.ndarray: Joint centers with shape (num_frames, num_joints, 3).
    """
    num_frames, num_markers, _ = array_3d.shape
    num_joints = len(joint_center_weights)

    # Create a mapping from marker names to indices
    marker_to_index = {marker: i for i, marker in enumerate(marker_names)}

    # Initialize weight matrix (num_joints, num_markers, 3)
    weights_matrix = np.zeros((num_joints, num_markers, 3))
    for j_idx, (joint, markers_weights) in enumerate(joint_center_weights.items()):
        for marker, weight in markers_weights.items():
            marker_idx = marker_to_index[marker]
            weights_matrix[j_idx, marker_idx, :] = weight  # Assign 3D weight

    # Compute joint centers
    # (num_frames, num_joints, 3) = (num_frames, num_markers, 3) @ (num_joints, num_markers, 3).T
    joint_centers = np.einsum('fmd,jmd->fjd', array_3d, weights_matrix)

    return joint_centers

def normalize(signal: pd.Series) -> pd.Series:
    """
    Normalize a signal to have zero mean and unit variance.
    
    Parameters:
        signal (pd.Series): The signal to normalize.

    Returns:
        pd.Series: The normalized signal.
    """
    return (signal - signal.mean()) / signal.std()

# def calculate_optimal_lag(freemocap_data:np.ndarray, qualisys_data:np.ndarray):
#     min_length = min(len(freemocap_data), len(qualisys_data))

#     freemocap_data = freemocap_data[:min_length]
#     qualisys_data = qualisys_data[:min_length]

#     normalized_freemocap = normalize(freemocap_data)
#     normalized_qualisys = normalize(qualisys_data)

#     cross_corr = np.correlate(normalized_freemocap, normalized_qualisys, mode='full')

#     optimal_lag = np.argmax(cross_corr) - (len(normalized_qualisys) - 1)
#     print(f"The optimal lag is: {optimal_lag}")

#     return optimal_lag

def calculate_optimal_lag(freemocap_data: np.ndarray, qualisys_data: np.ndarray):
    """
    Calculate the optimal lag for a single marker across all three dimensions (X, Y, Z).

    Parameters:
        freemocap_data (np.ndarray): FreeMoCap data of shape (frames, 1, 3) for a single marker.
        qualisys_data (np.ndarray): Qualisys data of shape (frames, 1, 3) for a single marker.

    Returns:
        np.ndarray: Optimal lags for each dimension (X, Y, Z).
    """
    # Ensure the data is shaped correctly
    assert freemocap_data.shape[1] == 3, "freemocap_data must be of shape (frames, 1, 3)"
    assert qualisys_data.shape[1] == 3, "qualisys_data must be of shape (frames, 1, 3)"


    # Calculate lags for each dimension
    optimal_lags = []
    for dim in range(3):  # Loop over X, Y, Z
        freemocap_dim = freemocap_data[:, dim]
        qualisys_dim = qualisys_data[:, dim]

        # Ensure the signals are the same length
        min_length = min(len(freemocap_dim), len(qualisys_dim))
        freemocap_dim = freemocap_dim[:min_length]
        qualisys_dim = qualisys_dim[:min_length]

        # Normalize the data
        normalized_freemocap = normalize(freemocap_dim)
        normalized_qualisys = normalize(qualisys_dim)

        # Compute cross-correlation
        cross_corr = np.correlate(normalized_freemocap, normalized_qualisys, mode='full')

        # Find the lag that maximizes the cross-correlation
        optimal_lag = np.argmax(cross_corr) - (len(normalized_qualisys) - 1)
        optimal_lags.append(optimal_lag)

    # Convert the result to a NumPy array
    optimal_lags = np.array(optimal_lags)
    print(f"The optimal lags for dimensions X, Y, Z are: {optimal_lags}")

    return optimal_lags
 
def reformat_dataframe_to_fmc_shaped_numpy_array(dataframe:pd.DataFrame):
    marker_dataframe_columns =  dataframe.columns[~ dataframe.columns.str.contains(r'^(?:Frame|Time|unix_timestamps|Unnamed)', regex=True)]
    marker_names = list(dict.fromkeys(col.split()[0] for col in marker_dataframe_columns))
    num_frames = len(dataframe)
    num_markers = len(marker_names)
    
    data_flat = dataframe[marker_dataframe_columns].to_numpy()
    return  data_flat.reshape(num_frames, num_markers, 3), marker_names  # Shape: (frames, markers, dimensions)

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


header_length = get_header_length(qualisys_marker_tsv_path)
qualisys_marker_trajectories = pd.read_csv(qualisys_marker_tsv_path, delimiter='\t', skiprows=header_length)
qualisys_unix_start_time = get_starting_qualisys_timestamp(qualisys_marker_tsv_path)

marker_data_formatted, marker_names = reformat_dataframe_to_fmc_shaped_numpy_array(qualisys_marker_trajectories)

marker_dataframe_columns =  qualisys_marker_trajectories.columns[~ qualisys_marker_trajectories.columns.str.contains(r'^(?:Frame|Time|unix_timestamps|Unnamed)', regex=True)]
qualisys_marker_names = list(dict.fromkeys(col.split()[0] for col in marker_dataframe_columns))
marker_data_flat = qualisys_marker_trajectories[marker_dataframe_columns].to_numpy()

num_frames = len(qualisys_marker_trajectories)
num_markers = len(qualisys_marker_names)

marker_data_formatted = marker_data_flat.reshape(num_frames, num_markers, 3)  # Shape: (frames, markers, dimensions)

qualisys_joint_center_trajectories_array = calculate_joint_centers(array_3d=marker_data_formatted, joint_center_weights=joint_center_weights, marker_names=qualisys_marker_names)
qualisys_joint_center_names = list(joint_center_weights.keys())
num_joints = np.shape(qualisys_joint_center_trajectories_array)[1]

qualisys_joint_center_trajectories = pd.DataFrame({
    'Frame': qualisys_marker_trajectories['Frame'],
    'Time': qualisys_marker_trajectories['Time'],
})

for joint_idx, joint_name in enumerate(qualisys_joint_center_names):
    for axis_idx, axis in enumerate(['x', 'y', 'z']):
        col_name = f"{joint_name} {axis}"
        qualisys_joint_center_trajectories[col_name] = qualisys_joint_center_trajectories_array[:, joint_idx, axis_idx]

qualisys_joint_center_trajectories_with_unix = create_and_insert_unix_timestamp_column(
    qualisys_joint_center_trajectories.copy(), 
    qualisys_unix_start_time
)
freemocap_timestamps, framerate = create_freemocap_unix_timestamps(freemocap_csv_path)
resampled_qualisys_joint_centers = resample_qualisys_data(qualisys_joint_center_trajectories_with_unix, freemocap_timestamps)

qualisys_joints_array,_ = reformat_dataframe_to_fmc_shaped_numpy_array(resampled_qualisys_joint_centers)

freemocap_joint_center_names = MediapipeModelInfo.landmark_names
freemocap_joint_centers = np.load(freemocap_data_path)

common_joint_centers = list(set(qualisys_joint_center_names) & set(freemocap_joint_center_names))

rotated_qualisys_joint_centers = run_skellyforge_rotation(qualisys_joints_array, qualisys_joint_center_names)
rotated_freemocap_joint_centers = run_skellyforge_rotation(freemocap_joint_centers, freemocap_joint_center_names)

optimal_lag_list = []
for joint_center in common_joint_centers:
    qualisys_marker_idx = qualisys_joint_center_names.index(joint_center)
    freemocap_marker_idx = freemocap_joint_center_names.index(joint_center)
    qualisys_marker_data = rotated_qualisys_joint_centers[:, qualisys_marker_idx, :]
    freemocap_marker_data = rotated_freemocap_joint_centers[:, freemocap_marker_idx, :]
    optimal_lag = calculate_optimal_lag(freemocap_marker_data, qualisys_marker_data)
    optimal_lag_list.append(optimal_lag)

median_lag = int(np.median(optimal_lag_list))
print(f'The median optimal lag for all common markers is: {median_lag}')

lag_seconds = convert_lag_from_frames_to_seconds(median_lag, framerate)

print(f"Calculated lag in seconds: {lag_seconds}")


lag_corrected_qualisys_joint_center_trajectories = create_and_insert_unix_timestamp_column(
    qualisys_joint_center_trajectories, 
    qualisys_unix_start_time,
    lag_in_seconds=lag_seconds
)


lag_corrected_resampled_qualisys_joint_centers = resample_qualisys_data(lag_corrected_qualisys_joint_center_trajectories, freemocap_timestamps)
lag_corrected_qualisys_joints_array,_ = reformat_dataframe_to_fmc_shaped_numpy_array(lag_corrected_resampled_qualisys_joint_centers)

lag_corrected_rotated_qualisys_joint_centers = run_skellyforge_rotation(lag_corrected_qualisys_joints_array, qualisys_joint_center_names)

optimal_lag_list = []
for joint_center in common_joint_centers:
    qualisys_marker_idx = qualisys_joint_center_names.index(joint_center)
    freemocap_marker_idx = freemocap_joint_center_names.index(joint_center)
    qualisys_marker_data = lag_corrected_rotated_qualisys_joint_centers[:, qualisys_marker_idx, :]
    freemocap_marker_data = rotated_freemocap_joint_centers[:, freemocap_marker_idx, :]
    optimal_lag = calculate_optimal_lag(freemocap_marker_data, qualisys_marker_data)
    optimal_lag_list.append(optimal_lag)

median_lag = int(np.median(optimal_lag_list))
print(f'The median optimal lag for all common markers is: {median_lag}')
f = 2


