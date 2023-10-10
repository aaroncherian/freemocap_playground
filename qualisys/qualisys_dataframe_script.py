from qualisys.qualisys_generic_marker_mapping import qualisys_marker_mappings
from qualisys.qualisys_plotting import plot_3d_scatter
from qualisys.qualisys_joint_center_mapping import joint_center_weights
from qualisys.calculate_joint_centers import calculate_joint_centers
import pandas as pd
from pathlib import Path

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

joint_centers_df = calculate_joint_centers(df, joint_center_weights)

plot_3d_scatter(joint_centers_df)
# plot_3d_scatter(df)




f = 2
