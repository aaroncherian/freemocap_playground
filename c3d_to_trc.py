import numpy as np
import pandas as pd
import ezc3d

def c3d_to_trc(c3d_file, transform='OsimXYZ'):
    """
    Converts a C3D file to TRC format and produces a pandas DataFrame.
    
    Parameters:
        c3d_file (str): Path to the C3D file.
        transform (str): Coordinate transformation. Options are 'OsimXYZ' or 'ViconXYZ'. Defaults to 'OsimXYZ'.
    
    Returns:
        trc_df (pd.DataFrame): A DataFrame containing the TRC data.
    """
    # Load the C3D file
    c3d = ezc3d.c3d(c3d_file)
    
    # Get sampling frequency and frame information
    sampling_freq = c3d['header']['points']['frame_rate']
    first_frame = c3d['header']['points']['first_frame']
    last_frame = c3d['header']['points']['last_frame']
    frames = np.arange(first_frame, last_frame + 1)
    
    # Get marker data
    marker_data = c3d['data']['points']
    n_points = marker_data.shape[1]
    
    # Reshape marker data: [frames, n_points, 3] -> [frames, n_points*3]
    marker_data = marker_data[:3, :, :].transpose(2, 0, 1).reshape(frames.size, -1)
    
    # Apply transformation if needed
    if transform == 'OsimXYZ':
        marker_data = transform_coordinates(marker_data, transform)
    
    # Get labels
    labels = [label.strip() for label in c3d['parameters']['POINT']['LABELS']['value']]
    times = (frames - 1) / sampling_freq

    # Construct column names: e.g., 'Marker1_x', 'Marker1_y', 'Marker1_z'
    col_names = [f"{label}_{axis}" for label in labels for axis in 'xyz']
    
    # Create the TRC DataFrame
    trc_df = pd.DataFrame(data=np.column_stack([times, marker_data]), columns=['Time'] + col_names)
    
    return trc_df, labels

def transform_coordinates(marker_data, transform):
    """
    Transforms marker data to a specified coordinate system (e.g., OpenSim's coordinate system).
    
    Parameters:
        marker_data (numpy.ndarray): Marker data to transform.
        transform (str): Coordinate transformation. Options are 'OsimXYZ' or 'ViconXYZ'.
        
    Returns:
        numpy.ndarray: Transformed marker data.
    """
    # Assuming the input transform is a placeholder; actual implementation may vary.
    if transform == 'OsimXYZ':
        # Example: Swap Y and Z axes (Vicon's coordinate system -> OpenSim's coordinate system)
        marker_data[:, [1, 2]] = marker_data[:, [2, 1]]
    return marker_data

def save_trc(df, labels, output_path, sampling_freq):
    """
    Saves the TRC DataFrame to a TRC file.
    
    Parameters:
        df (pd.DataFrame): The TRC data as a DataFrame.
        labels (list): List of marker names (labels) corresponding to the markers in the data.
        output_path (str): Path to save the TRC file.
        sampling_freq (float): Sampling frequency of the data.
    """
    n_markers = (df.shape[1] - 1) // 3  # Number of markers (excluding 'Time' column)
    
    # Ensure that the length of the labels matches the number of markers
    if len(labels) != n_markers:
        raise ValueError(f"Number of labels ({len(labels)}) does not match number of markers ({n_markers}).")
    
    # TRC header lines
    header_lines = [
        "PathFileType\t4\t(X/Y/Z)\t{}\n".format(output_path),
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n",
        f"{sampling_freq:.2f}\t{sampling_freq:.2f}\t{df.shape[0]}\t{n_markers}\tmm\t{sampling_freq:.2f}\t1\t{df.shape[0]}\n",
        "Frame#\tTime\t" + "\t\t\t".join(labels) + "\n",  # Marker labels
             "\t\t" + "\t".join(['X{0}\tY{0}\tZ{0}'.format(i+1) for i in range(n_markers)]) + "\n"  # XYZ header
    ]
    
    # Write header and data to the file
    with open(output_path, 'w') as trc_file:
        trc_file.writelines(header_lines)
        df.to_csv(trc_file, sep='\t', index=True, header=False, float_format="%.6f")


# Example usage:
c3d_file = r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_13_48_44_MDN_treadmill_2\qualisys_data\MDN_treadmill_2_tracked.c3d"
trc_df,labels = c3d_to_trc(c3d_file, transform='OsimXYZ')
output_path = r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_13_48_44_MDN_treadmill_2\qualisys_data\MDN_treadmill_2_tracked.trc"
save_trc(trc_df, labels, output_path, sampling_freq=300)  #
