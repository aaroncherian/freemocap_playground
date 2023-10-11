from qualisys.qualisys_generic_marker_mapping import qualisys_marker_mappings
from qualisys.qualisys_plotting import plot_3d_scatter
from qualisys.qualisys_joint_center_mapping import joint_center_weights
from qualisys.calculate_joint_centers import calculate_joint_centers
import pandas as pd
from pathlib import Path

def dataframe_to_numpy(df):
    # Get the list of unique markers in the order they appear for frame 0
    marker_order = df['marker'].unique().tolist()
    
    # Create a dictionary to map marker names to their order
    marker_order_dict = {marker: idx for idx, marker in enumerate(marker_order)}
    
    # Sort DataFrame by 'frame' and then by the custom marker order
    df['marker_rank'] = df['marker'].map(marker_order_dict)
    df_sorted = df.sort_values(by=['frame', 'marker_rank']).drop(columns=['marker_rank'])
    
    # Extract the x, y, z columns as a NumPy array
    coords_array = df_sorted[['x', 'y', 'z']].to_numpy()
    
    # Get the number of unique frames and markers
    num_frames = df['frame'].nunique()
    num_markers = len(marker_order)
    
    # Reshape the array into the desired shape (frames, markers, dimensions)
    reshaped_array = coords_array.reshape((num_frames, num_markers, 3))
    
    return reshaped_array




path_to_recording_folder = Path(r"D:\2023-06-07_JH\1.0_recordings\treadmill_calib\sesh_2023-06-07_12_06_15_JH_flexion_neutral_trial_1")
path_to_qualisys_folder = path_to_recording_folder / 'qualisys'
path_to_qualisys_csv = path_to_qualisys_folder / 'qualisys_markers_dataframe.csv'

df = pd.read_csv(path_to_qualisys_csv)

# Flatten the nested dictionary to easily map biomechanical to generic names
flat_mappings = {}
for joint, markers in qualisys_marker_mappings.items():
    for biomech, generic in markers.items():
        flat_mappings[biomech] = generic

# Filter rows to keep only the markers that are in the flat_mappings dictionary
df = df[df['marker'].isin(flat_mappings.keys())]

# Replace the marker names in the DataFrame
df['marker'] = df['marker'].replace(flat_mappings)

qualisys_frame_marker_dimension = dataframe_to_numpy(df)

# array_dict = {'Qualisys Markers': qualisys_frame_marker_dimension}

# plot_3d_scatter(array_dict)

marker_names = df['marker'].unique().tolist()

f = 2 

# plot_3d_scatter(df)


# joint_centers_df = calculate_joint_centers(df, joint_center_weights)
joint_centers = calculate_joint_centers(qualisys_frame_marker_dimension, joint_center_weights, marker_names)

array_dict = {'Qualisys Markers': qualisys_frame_marker_dimension, 'Qualisys Joint Centers': joint_centers}

plot_3d_scatter(array_dict)
f = 2 

# plot_3d_scatter(joint_centers_df)




f = 2
