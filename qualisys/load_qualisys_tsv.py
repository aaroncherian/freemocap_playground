import pandas as pd
import numpy as np
from pathlib import Path
# Load the TSV into a DataFrame

path_to_recording_folder = Path(r"D:\2023-06-07_JH\1.0_recordings\treadmill_calib\sesh_2023-06-07_12_06_15_JH_flexion_neutral_trial_1")
path_to_qualisys_folder = path_to_recording_folder / 'qualisys'
path_to_tsv = path_to_qualisys_folder / 'flexion_neutral_trial_1_tracked.tsv'
path_to_save_numpy_array = path_to_qualisys_folder / 'qualisys_markers.npy'

# path_to_tsv = r"D:\2023-06-07_JH\1.0_recordings\treadmill_calib\sesh_2023-06-07_12_06_15_JH_flexion_neutral_trial_1\qualisys\flexion_neutral_trial_1_tracked.tsv"
df = pd.read_csv(path_to_tsv, sep='\t')

# Drop the 'Frame' and 'Time' columns
df.drop(columns=['Frame', 'Time'], inplace=True)

# Find the unique markers and map them to integers
unique_markers = {marker.split(' ')[0]: i for i, marker in enumerate(df.columns[::3])}

# Create the reorganized_data list with marker names as strings
reorganized_data_df = [
    [frame, col.split(' ')[0], row[col], row[f"{col.split(' ')[0]} Y"], row[f"{col.split(' ')[0]} Z"]]
    for frame, row in df.iterrows() for col in df.columns[::3]
]

# Create the reorganized_data list with marker names as integers
# reorganized_data_array = [
#     [frame, unique_markers[col.split(' ')[0]], row[col], row[f"{col.split(' ')[0]} Y"], row[f"{col.split(' ')[0]} Z"]]
#     for frame, row in df.iterrows() for col in df.columns[::3]
# ]   

# Convert the reorganized_data list to a DataFrame
reorganized_df = pd.DataFrame(reorganized_data_df, columns=['frame', 'marker', 'x', 'y', 'z'])
reorganized_df.to_csv(path_to_qualisys_folder / 'qualisys_markers_dataframe.csv', index=False)

# Convert the reorganized_data list to a NumPy array
# reorganized_array = np.array(reorganized_data_array)

# # Reshape the array to [frame, marker, dimension]
# reshaped_array = reorganized_array.reshape(df.shape[0], len(unique_markers), 5)[:, :, 2:]

# np.save(path_to_save_numpy_array, reshaped_array)
# f = 2
