import numpy as np
import os

def numpy_to_trc(skeleton_3d_data, marker_names, frame_rate, output_file):
    """
    Convert 3D skeleton data (numpy array) to OpenSim TRC format compatible with pose2sim.
    
    Parameters
    ----------
    skeleton_3d_data : np.ndarray
        Shape (num_frames, num_markers, 3)
    marker_names : list of str
        Marker names, len(marker_names) must equal num_markers
    frame_rate : float
        Sampling rate in Hz
    output_file : str
        Path to output TRC file
    
    Returns
    -------
    str
        Path to the created TRC file
    """
    num_frames, num_markers, _ = skeleton_3d_data.shape
    frame_nums = np.arange(1, num_frames + 1)
    time_vals = (frame_nums - 1) / frame_rate

    # Step 1: Create marker data dictionary like original code
    marker_data = {
        marker: {'X': skeleton_3d_data[:, i, 0],
                 'Y': skeleton_3d_data[:, i, 1],
                 'Z': skeleton_3d_data[:, i, 2]}
        for i, marker in enumerate(marker_names)
    }

    # Step 2: Write file exactly like csv_to_trc
    with open(output_file, 'w') as trc_file:
        # Line 1
        trc_file.write(f"PathFileType\t4\t(X/Y/Z)\t{os.path.basename(output_file)}\n")
        # Line 2
        trc_file.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        # Line 3
        trc_file.write(f"{int(frame_rate)}\t{int(frame_rate)}\t{num_frames}\t{num_markers}\tmm\t{int(frame_rate)}\t1\t{num_frames}\n")
        
        # Line 4: Marker names with THREE tabs between each marker (for X,Y,Z columns)
        marker_header = "Frame#\tTime"
        for marker in marker_names:
            marker_header += f"\t{marker}\t\t"  # marker name followed by two tabs (for Y,Z columns)
        marker_header = marker_header.rstrip("\t")
        trc_file.write(marker_header + "\n")

        # Line 5: Coordinate labels - numbered X1 Y1 Z1 X2 Y2 Z2 etc for OpenSim IK
        coord_labels = "\t\t"  # Two tabs to align with Frame# and Time columns
        for i in range(num_markers):
            coord_labels += f"X{i+1}\tY{i+1}\tZ{i+1}"
            if i < num_markers - 1:
                coord_labels += "\t"
        trc_file.write(coord_labels + "\n")
        
        # Empty line after headers (some TRC readers expect this)
        trc_file.write("\n")

        # Data rows
        for i in range(num_frames):
            row = [f"{frame_nums[i]}", f"{time_vals[i]:.6f}"]
            for marker in marker_names:
                # Coordinate transform
                x = marker_data[marker]['X'][i]
                y = marker_data[marker]['Y'][i]
                z = marker_data[marker]['Z'][i]
                opensim_x = y
                opensim_y = z
                opensim_z = x
                row.extend([f"{opensim_x:.6f}", f"{opensim_y:.6f}", f"{opensim_z:.6f}"])
            trc_file.write("\t".join(row) + "\n")
    
    print(f"Successfully created TRC file: {output_file}")
    return output_file

# Example usage
if __name__ == "__main__":
    from pathlib import Path
    import numpy as np
    from skellymodels.model_info.mediapipe_model_info import MediapipeModelInfo
    
    # Your paths
    tracker_name = 'mediapipe'
    path_to_recording_folder = Path(r'D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_13_48_44_MDN_treadmill_2')
    path_to_data = path_to_recording_folder / 'validation' / tracker_name / f'{tracker_name}_body_3d_xyz.npy'
    output_trc_path = path_to_recording_folder / 'validation' / tracker_name / f'{tracker_name}_body_3d_xyz.trc'
    
    # Load data
    skel3d_data = np.load(path_to_data)
    
    # Convert to TRC
    numpy_to_trc(
        skeleton_3d_data=skel3d_data,
        marker_names=MediapipeModelInfo.landmark_names,
        frame_rate=30,
        output_file=str(output_trc_path)
    )