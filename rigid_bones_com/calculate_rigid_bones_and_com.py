from pathlib import Path
import numpy as np

from skellymodels.create_model_skeleton import create_mediapipe_skeleton_model, create_rigid_mediapipe_skeleton_model
from skellymodels.model_info.rigid_body_mediapipe_model_info import RigidMediapipeModelInfo
from rigid_bones import enforce_rigid_bones_from_skeleton
from calculate_center_of_mass import calculate_center_of_mass_from_skeleton

import pandas as pd
import numpy as np

def reorder_csv_to_numpy_with_strip(csv_df, body_landmark_names):
    # Strip leading/trailing spaces from the column names
    csv_df.columns = csv_df.columns.str.strip()
    
    # Remove the extra hand middle markers
    csv_df = csv_df.loc[:, ~csv_df.columns.str.startswith(('left_hand_middle', 'right_hand_middle'))]
    
    # Create an empty list to store ordered data
    ordered_data = []

    # Loop through the provided body landmark names
    for marker in body_landmark_names:
        # Prepare the corresponding column names in the CSV
        x_col = f"{marker}_x"
        y_col = f"{marker}_y"
        z_col = f"{marker}_z"
        
        # Check if all the columns for the current marker exist in the CSV
        if all(col in csv_df.columns for col in [x_col, y_col, z_col]):
            # Extract the marker data as (frame, 3 dimensions)
            marker_data = csv_df[[x_col, y_col, z_col]].to_numpy()
            ordered_data.append(marker_data)
    
    # Stack the ordered data into a numpy array with the shape (frame, marker, dimension)
    if ordered_data:
        ordered_data = np.stack(ordered_data, axis=1)
        return ordered_data
    else:
        return None

path_to_recording_folder = Path(r'D:\2024-08-01_treadmill_KK_JSM_ATC\1.0_recordings\sesh_2024-08-01_16_18_26_JSM_wrecking_ball')
path_to_data = Path(r'D:\2024-08-01_treadmill_KK_JSM_ATC\1.0_recordings\sesh_2024-08-01_16_18_26_JSM_wrecking_ball\saved_data\csv\body_trajectories.csv')
df_new = pd.read_csv(path_to_data)
body_landmark_names = RigidMediapipeModelInfo.body_landmark_names

ordered_numpy_data_with_strip_new = reorder_csv_to_numpy_with_strip(df_new, body_landmark_names)


# # # path_to_non_rigid_npy = path_to_recording_folder/'output_data'/'origin_aligned_data'/'mediapipe_body_3d_xyz.npy'
# # folder_to_save = path_to_recording_folder/'output_data'/'rigid_bones_data'
# # folder_to_save.mkdir(exist_ok=True, parents=True)

# folder_to_save_com = path_to_recording_folder/'output_data'/'rigid_bones_data'/'center_of_mass_data'
# folder_to_save_com.mkdir(exist_ok=True, parents=True)

# non_rigid_data = np.load(path_to_non_rigid_npy)

# non_rigid_skeleton = create_mediapipe_skeleton_model()
# non_rigid_skeleton.integrate_freemocap_3d_data(non_rigid_data)

# rigid_marker_data = enforce_rigid_bones_from_skeleton(non_rigid_skeleton)

# rigid_skeleton = non_rigid_skeleton.copy()
# rigid_skeleton.integrate_rigid_marker_data(rigid_marker_data)


# rigid_data = rigid_skeleton.marker_data_as_numpy
# _,com_data = calculate_center_of_mass_from_skeleton(rigid_skeleton)

# np.save(folder_to_save/'mediapipe_body_3d_xyz.npy', rigid_data)
# np.save(folder_to_save_com/'total_body_center_of_mass_xyz.npy', com_data)


rigid_body_skeleton = create_rigid_mediapipe_skeleton_model()
rigid_body_skeleton.integrate_freemocap_3d_data(ordered_numpy_data_with_strip_new)

_, com_data = calculate_center_of_mass_from_skeleton(rigid_body_skeleton)

folder_to_save_com = path_to_recording_folder/'output_data'/'rigid_bones_data'/'center_of_mass_data'
np.save(folder_to_save_com/'total_body_center_of_mass_xyz.npy', com_data)
f = 2