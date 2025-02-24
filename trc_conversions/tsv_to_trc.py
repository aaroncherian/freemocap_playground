import pandas as pd
import numpy as np
import os
import re

def convert_qualisys_tsv_to_trc(input_file, output_file=None, frame_rate=None):
    """
    Convert Qualisys TSV file to OpenSim TRC format
    
    Parameters:
    -----------
    input_file : str
        Path to the Qualisys TSV file
    output_file : str, optional
        Path to output TRC file. If None, will use same name as input but with .trc extension
    frame_rate : float, optional
        Capture frame rate in Hz. If None, will be extracted from TSV metadata.
    
    Returns:
    --------
    output_file : str
        Path to the created TRC file
    """
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + '.trc'
    
    # Read the TSV file line by line to extract metadata and find data start
    metadata = {}
    marker_names = []
    data_start_line = 0
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        # Extract important metadata
        if line.startswith('NO_OF_FRAMES'):
            metadata['NO_OF_FRAMES'] = int(line.split('\t')[1])
        elif line.startswith('NO_OF_MARKERS'):
            metadata['NO_OF_MARKERS'] = int(line.split('\t')[1])
        elif line.startswith('FREQUENCY'):
            metadata['FREQUENCY'] = float(line.split('\t')[1])
        elif line.startswith('MARKER_NAMES'):
            marker_names = line.split('\t')[1:]
        elif line.startswith('Frame\tTime'):
            # Found the header row before data
            data_start_line = i + 1
            header_line = line
            break
    
    if not marker_names:
        raise ValueError("Could not find marker names in the TSV file")
        
    if frame_rate is None and 'FREQUENCY' in metadata:
        frame_rate = metadata['FREQUENCY']
        print(f"Using frequency from file: {frame_rate} Hz")
    elif frame_rate is None:
        raise ValueError("Frame rate not specified and not found in file metadata")
    
    # Parse column headers to understand the structure
    headers = header_line.split('\t')
    
    # Map markers to their column indices for X, Y, Z coordinates
    marker_columns = {}
    for marker in marker_names:
        marker_columns[marker] = {}
        for i, header in enumerate(headers):
            if header == f"{marker} X":
                marker_columns[marker]['X'] = i
            elif header == f"{marker} Y":
                marker_columns[marker]['Y'] = i
            elif header == f"{marker} Z":
                marker_columns[marker]['Z'] = i
    
    # Get indices for frame and time columns
    frame_idx = headers.index('Frame')
    time_idx = headers.index('Time')
    
    # Now read the actual data
    data_rows = []
    for i in range(data_start_line, len(lines)):
        if not lines[i].strip():
            continue
        data_rows.append(lines[i].strip().split('\t'))
    
    # Extract required data and convert coordinates
    parsed_data = []
    for row in data_rows:
        if len(row) < max(time_idx, frame_idx) + 1:
            continue  # Skip rows with insufficient data
            
        frame = int(float(row[frame_idx]))
        time = float(row[time_idx])
        
        marker_data = {}
        for marker in marker_names:
            try:
                # Get Qualisys coordinates
                qualisys_x = float(row[marker_columns[marker]['X']])
                qualisys_y = float(row[marker_columns[marker]['Y']])
                qualisys_z = float(row[marker_columns[marker]['Z']])
                
                # Transform to OpenSim coordinate system
                # OpenSim: X (forward), Y (up), Z (right)
                # Qualisys: X (forward), Y (left), Z (up)
                opensim_x = qualisys_x
                opensim_y = qualisys_z
                opensim_z = -qualisys_y
                
                marker_data[marker] = (opensim_x, opensim_y, opensim_z)
            except (ValueError, KeyError, IndexError) as e:
                # Handle missing data by setting NaN
                marker_data[marker] = (np.nan, np.nan, np.nan)
        
        parsed_data.append((frame, time, marker_data))
    
    # Start writing TRC file
    with open(output_file, 'w') as f:
        # Line 1: File type
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{os.path.abspath(output_file)}\n")
        
        # Line 2: Data info
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        
        # Line 3: Data values
        num_frames = len(parsed_data)
        start_frame = parsed_data[0][0] if parsed_data else 1
        f.write(f"{int(frame_rate)} {int(frame_rate)} {num_frames} {len(marker_names)}\tmm\t{int(frame_rate)} {start_frame}\t{num_frames}\n")
        
        # Line 4: Marker names header
        f.write("Frame#\tTime")
        for marker in marker_names:
            f.write(f"\t{marker}\t\t")
        f.write("\n")
        
        # Line 5: Coordinate names
        f.write("\t")  # Empty cell for Frame# column
        f.write("\t")  # Empty cell for Time column
        for _ in marker_names:
            f.write("X\tY\tZ\t")
        f.write("\n")
        
        # Data rows
        for frame, time, marker_data in parsed_data:
            f.write(f"{frame}\t{time:.1f}")
            
            for marker in marker_names:
                x, y, z = marker_data[marker]
                f.write(f"\t{x:.1f}\t{y:.1f}\t{z:.1f}")
            
            f.write("\n")
    
    print(f"Successfully converted {input_file} to {output_file}")
    print(f"Created TRC file with {len(marker_names)} markers and {num_frames} frames")
    return output_file

tsv_file = r"D:\2023-06-07_TF01\1.0_recordings\treadmill_calib\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1\qualisys_data\flexion_neutral_trial_1_tracked_with_header.tsv"
output_path = r"D:\2023-06-07_TF01\1.0_recordings\treadmill_calib\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1\qualisys_data\flexion_neutral_trial_1_tracked.trc"

convert_qualisys_tsv_to_trc(input_file=tsv_file, 
                   output_file=output_path,
                   frame_rate=300)

