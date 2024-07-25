from skellyforge.freemocap_utils.postprocessing_widgets.postprocessing_functions.rotate_skeleton import align_skeleton_with_origin
from skellyforge.freemocap_utils.postprocessing_widgets.task_worker_thread import TaskWorkerThread
from skellyforge.freemocap_utils.postprocessing_widgets.postprocessing_functions.good_frame_finder import find_good_frame

from pathlib import Path
import numpy as np

from skellymodels.model_info.mediapipe_model_info import MediapipeModelInfo

path_to_recording = Path(r'D:\2023-06-07_TF01\1.0_recordings\treadmill_calib\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1')
path_to_data = path_to_recording / 'output_data' / 'mediapipe_body_3d_xyz.npy'

data = np.load(path_to_data)


good_frame = find_good_frame(skeleton_data=data,skeleton_indices= MediapipeModelInfo.body_landmark_names, initial_velocity_guess=.5)

aligned_data = align_skeleton_with_origin(skeleton_data=data, skeleton_indices=MediapipeModelInfo.body_landmark_names, good_frame=good_frame)[0]   

np.save(path_to_data, aligned_data)
f = 2