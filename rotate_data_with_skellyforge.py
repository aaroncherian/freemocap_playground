from skellyforge.freemocap_utils.postprocessing_widgets.postprocessing_functions.rotate_skeleton import align_skeleton_with_origin
from skellyforge.freemocap_utils.postprocessing_widgets.task_worker_thread import TaskWorkerThread
from skellyforge.freemocap_utils.postprocessing_widgets.postprocessing_functions.good_frame_finder import find_good_frame

from pathlib import Path
import numpy as np

from skellymodels.model_info.mediapipe_model_info import MediapipeModelInfo
from skellymodels.create_model_skeleton import create_mediapipe_skeleton_model
from rigid_bones_com.calculate_center_of_mass import calculate_center_of_mass_from_skeleton

path_to_recording = Path(r'D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_15_46_22_MDN_OneLeg_Trial2')
path_to_data = path_to_recording / 'output_data' / 'mediapipe_body_3d_xyz.npy'

path_to_save_data = path_to_recording / 'output_data' / 'aligned_data'/'mediapipe_body_3d_xyz.npy'
path_to_save_data.parent.mkdir(parents=True, exist_ok=True)

path_to_save_com_data = path_to_recording / 'output_data' / 'aligned_data'/ 'center_of_mass'/'mediapipe_total_body_center_of_mass_xyz.npy'
path_to_save_com_data.parent.mkdir(parents=True, exist_ok=True)

data = np.load(path_to_data)


good_frame = find_good_frame(skeleton_data=data,skeleton_indices= MediapipeModelInfo.body_landmark_names, initial_velocity_guess=.5)

aligned_data = align_skeleton_with_origin(skeleton_data=data, skeleton_indices=MediapipeModelInfo.body_landmark_names, good_frame=good_frame)[0]   

mediapipe_skeleton = create_mediapipe_skeleton_model()
mediapipe_skeleton.integrate_freemocap_3d_data(aligned_data)

_, total_body_com = calculate_center_of_mass_from_skeleton(mediapipe_skeleton)


np.save(path_to_save_data, aligned_data)
np.save(path_to_save_com_data, total_body_com)


f = 2