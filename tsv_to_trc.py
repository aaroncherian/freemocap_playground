import numpy as np
import pandas as pd

def tsv_to_trc(tsv_data, output_trc_path, transform='OsimXYZ', sampling_freq=100):
    """
    Converts TSV-formatted motion capture data to TRC format.
    
    Parameters:
        tsv_data (pd.DataFrame): DataFrame containing the TSV data.
        output_trc_path (str): Path to save the output TRC file.
        transform (str): Optional, specify coordinate system transformation ('OsimXYZ' for OpenSim).
        sampling_freq (float): Sampling frequency of the data.
    """
    # Filter out any columns that are unnamed or irrelevant
    marker_columns = [col for col in tsv_data.columns if col not in ['Time', 'Frame', 'unix_timestamps'] and not col.startswith('Unnamed')]
    
    # Get unique marker names by splitting the columns on spaces and keeping the first part (e.g., 'RIC' from 'RIC X')
    markers = []
    for col in marker_columns:
        marker_name = col.split(' ')[0]  # E.g., 'RIC' from 'RIC X'
        if marker_name not in markers:
            markers.append(marker_name)
    
    # Create an array for the marker data (frames x markers * 3)
    num_frames = len(tsv_data)
    num_markers = len(markers)
    
    # Initialize the TRC data storage (for XYZ coordinates of each marker)
    marker_data = np.zeros((num_frames, num_markers * 3))
    
    # Fill the marker data (extracting X, Y, and Z for each marker)
    for marker_idx, marker in enumerate(markers):
        for axis_idx, axis in enumerate(['X', 'Y', 'Z']):
            marker_column = f"{marker} {axis}"  # E.g., 'RIC X', 'RIC Y', 'RIC Z'
            if marker_column in tsv_data.columns:
                marker_data[:, marker_idx * 3 + axis_idx] = tsv_data[marker_column].to_numpy()
    
    # Apply OpenSim transformation if required
    if transform == 'OsimXYZ':
        # Swap Y and Z axes (Vicon to OpenSim conversion)
        marker_data[:, 1::3], marker_data[:, 2::3] = marker_data[:, 2::3], marker_data[:, 1::3]
    
    # Extract time and frame numbers
    time = tsv_data['Time'].to_numpy()
    
    # TRC header lines
    header_lines = [
        "PathFileType\t4\t(X/Y/Z)\t{}\n".format(output_trc_path),
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n",
        f"{sampling_freq:.2f}\t{sampling_freq:.2f}\t{num_frames}\t{num_markers}\tmm\t{sampling_freq:.2f}\t1\t{num_frames}\n",
        "Frame#\tTime\t" + "\t\t\t".join(markers) + "\n",
        "\t\t" + "\t".join([f'X{i+1}\tY{i+1}\tZ{i+1}' for i in range(num_markers)]) + "\n"
    ]
    
    # Write header and data to the TRC file
    with open(output_trc_path, 'w') as trc_file:
        # Write the header
        trc_file.writelines(header_lines)
        
        # Write the marker data (frame number, time, and marker coordinates)
        for frame_idx in range(num_frames):
            frame_number = frame_idx + 1  # Frames are 1-based in TRC files
            row_data = np.concatenate([[frame_number, time[frame_idx]], marker_data[frame_idx, :]])
            row_str = "\t".join(f"{val:.6f}" for val in row_data)
            trc_file.write(row_str + "\n")
    
    print(f"Saved TRC file to {output_trc_path}")

# Example usage:
# Assuming `tsv_data` is your parsed DataFrame from the TSV file
tsv_data = pd.read_csv(r"D:\2023-06-07_TF01\1.0_recordings\treadmill_calib\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1\validation\qualisys\qualisys_synced_markers.csv")
tsv_to_trc(tsv_data, r"D:\2023-06-07_TF01\1.0_recordings\treadmill_calib\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1\validation\qualisys\synchronized_markers.trc", transform='OsimXYZ', sampling_freq=30)
