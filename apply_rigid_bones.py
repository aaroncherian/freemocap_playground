from pathlib import Path
import numpy as np

from skellymodels.create_model_skeleton import create_mediapipe_skeleton_model
from rigid_bones_com.rigid_bones import enforce_rigid_bones_from_skeleton


path_to_recording_folder = Path(r'D:\cyr_wheel\cyr_recording')
data_to_use = 'mediapipe_body_3d_xyz.npy'

path_to_output_data = path_to_recording_folder/'output_data'/data_to_use

data = np.load(path_to_output_data)

mediapipe_model = create_mediapipe_skeleton_model()

mediapipe_model.integrate_freemocap_3d_data(data)

rigid_bones_data = enforce_rigid_bones_from_skeleton(mediapipe_model)

folder_to_save = path_to_recording_folder/'output_data'/'component_rigid_bones'
folder_to_save.mkdir(exist_ok=True, parents=True)

np.save(folder_to_save/'mediapipe_body_3d_xyz.npy', rigid_bones_data)

f = 2

