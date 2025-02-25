import numpy as np
import pandas as pd
import os
import argparse
import re
from datetime import datetime

def csv_to_trc(input_file, output_file=None, frame_rate=None):
    """
    Robustly convert any motion capture file (CSV/TSV) to OpenSim TRC format.
    
    Parameters:
    -----------
    input_file : str
        Path to the input motion capture file
    output_file : str, optional
        Path to output TRC file. If None, will use same name as input but with .trc extension
    frame_rate : float, optional
        Capture frame rate in Hz. If None, will attempt to extract from file or calculate.
    debug : bool
        Whether to print debugging information
    
    Returns:
    --------
    output_file : str
        Path to the created TRC file
    """
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + '.trc'
    
    print(f"Converting {input_file} to {output_file}")
    
    # Step 1: Read the file, automatically detecting format
    try:
        # Try reading as CSV first
        df = pd.read_csv(input_file, sep=None, engine='python')
        print(f"Successfully read file with {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        print(f"Error reading file: {e}")
        print("Trying alternative parsing method...")
        
        # Try parsing the file manually line by line
        try:
            with open(input_file, 'r') as f:
                lines = f.readlines()
            
            # Look for data header line
            header_idx = None
            for i, line in enumerate(lines):
                if any(keyword in line for keyword in ['Frame', 'Time', 'FRAME']):
                    header_idx = i
                    break
            
            if header_idx is None:
                raise ValueError("Could not find header line with Frame/Time columns")
            
            # Parse header and data
            header = lines[header_idx].strip().split('\t' if '\t' in lines[header_idx] else ',')
            data = []
            for line in lines[header_idx+1:]:
                if line.strip():
                    row = line.strip().split('\t' if '\t' in line else ',')
                    if len(row) == len(header):
                        data.append(row)
            
            df = pd.DataFrame(data, columns=header)
            print(f"Manually parsed file with {len(df)} rows and {len(df.columns)} columns")
        except Exception as e2:
            raise ValueError(f"Failed to parse file: {e2}")
    
    # Step 2: Standardize column names
    df.columns = [col.strip() for col in df.columns]
    
    # Find time and frame columns
    time_col = next((col for col in df.columns if col.lower() in ['time', 't']), None)
    frame_col = next((col for col in df.columns if col.lower() in ['frame', 'frame#', 'frame_num']), None)
    
    if not time_col:
        raise ValueError("Could not find Time column")
    
    if not frame_col and 'index' in df.columns:
        frame_col = 'index'
    elif not frame_col:
        print("Warning: No frame column found, creating sequential frames")
        df['Frame'] = np.arange(len(df))
        frame_col = 'Frame'
    
    # Convert columns to numeric
    df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
    df[frame_col] = pd.to_numeric(df[frame_col], errors='coerce')
    
    # Step 3: Extract marker data
    # Find marker columns (X, Y, Z triplets)
    marker_columns = []
    marker_names = []
    
    for col in df.columns:
        if col == time_col or col == frame_col:
            continue
        
        # Extract marker name and axis
        # Look for patterns like "Marker X", "Marker.X", "Marker_X"
        match = re.search(r'(.+?)[\s\._]([XYZ])$', col)
        if match:
            marker_name, axis = match.groups()
            if marker_name not in marker_names:
                marker_names.append(marker_name)
            marker_columns.append((col, marker_name, axis))
    
    if not marker_names:
        # Try alternative pattern matching if no markers found
        # Look for patterns like "X1", "Y1", "Z1" indicating numbered coordinates
        xyz_pattern = re.compile(r'([XYZ])(\d+)')
        marker_indices = set()
        
        for col in df.columns:
            match = xyz_pattern.match(col)
            if match:
                axis, index = match.groups()
                marker_indices.add(int(index))
        
        if marker_indices:
            marker_names = [f"Marker_{i}" for i in sorted(marker_indices)]
            for i in sorted(marker_indices):
                for axis in ['X', 'Y', 'Z']:
                    col_name = f"{axis}{i}"
                    if col_name in df.columns:
                        marker_columns.append((col_name, f"Marker_{i}", axis))
    
    if not marker_names:
        raise ValueError("Could not identify marker data columns")
    
    print(f"Found {len(marker_names)} markers: {', '.join(marker_names)}")
    
    # Convert marker data to numeric
    for col, _, _ in marker_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Step 4: Detect or calculate frame rate
    if frame_rate is None:
        time_diffs = df[time_col].diff().dropna()
        if not time_diffs.empty:
            avg_diff = time_diffs.mean()
            if avg_diff > 0:
                frame_rate = round(1.0 / avg_diff)
                print(f"Calculated frame rate: {frame_rate} Hz")
            else:
                frame_rate = 100  # Default
                print(f"Warning: Could not calculate frame rate. Using default: {frame_rate} Hz")
        else:
            frame_rate = 100  # Default
            print(f"Warning: Could not calculate frame rate. Using default: {frame_rate} Hz")
    
    # Step 5: Organize data for TRC format
    num_frames = len(df)
    num_markers = len(marker_names)
    
    # Create marker data dictionary
    marker_data = {marker: {'X': np.zeros(num_frames), 'Y': np.zeros(num_frames), 'Z': np.zeros(num_frames)} 
                  for marker in marker_names}
    
    # Fill marker data
    for col, marker, axis in marker_columns:
        marker_data[marker][axis] = df[col].values
    
    # Step 6: Create the TRC file
    with open(output_file, 'w') as trc_file:
        # Line 1: PathFileType
        trc_file.write(f"PathFileType\t4\t(X/Y/Z)\t{os.path.basename(output_file)}\n")
        
        # Line 2: Header labels
        trc_file.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        
        # Line 3: Header values - ensure all values are integers where expected
        start_frame = int(df[frame_col].iloc[0])
        trc_file.write(f"{int(frame_rate)}\t{int(frame_rate)}\t{num_frames}\t{num_markers}\tmm\t{int(frame_rate)}\t{start_frame}\t{num_frames}\n")
        
        # Line 4: Column labels for markers
        marker_header = "Frame#\tTime"
        for marker in marker_names:
            marker_header += f"\t{marker}\t\t"
        marker_header = marker_header.rstrip("\t")
        trc_file.write(marker_header + "\n")
        
        # Line 5: Column labels for X,Y,Z coordinates (using X1, Y1, Z1, X2, Y2, Z2 format)
        coord_labels = "\t"  # empty cell for Frame# column
        for i, marker in enumerate(marker_names, start=1):
            coord_labels += f"\tX{i}\tY{i}\tZ{i}"
        trc_file.write(coord_labels + "\n")
        
        # Data rows
        for i in range(num_frames):
            row_data = [f"{int(df[frame_col].iloc[i])}", f"{df[time_col].iloc[i]:.6f}"]
            
            for marker in marker_names:
                # Get original data
                x = marker_data[marker]['X'][i]
                y = marker_data[marker]['Y'][i]
                z = marker_data[marker]['Z'][i]
                
                # Apply coordinate transformation (try different mappings)
                # Variant 1: Direct mapping (X→X, Y→Y, Z→Z)
                opensim_x = x
                opensim_y = z
                opensim_z = -y
                
                row_data.extend([f"{opensim_x:.6f}", f"{opensim_y:.6f}", f"{opensim_z:.6f}"])
            
            trc_file.write("\t".join(row_data) + "\n")
    
    print(f"Successfully created TRC file with {num_markers} markers and {num_frames} frames")
    
    return output_file



csv_file_path = r"D:\2023-06-07_TF01\1.0_recordings\treadmill_calib\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1\output_data\component_qualisys_synced\marker_data_synced.csv"
output_trc_path = r"D:\2023-06-07_TF01\1.0_recordings\treadmill_calib\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1\output_data\component_qualisys_synced\marker_data_synced.trc"
sampling_freq = 30  # Hz
csv_to_trc(csv_file_path, output_trc_path, frame_rate = sampling_freq)