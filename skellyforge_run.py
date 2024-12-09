
from pathlib import Path
import numpy as np
from skellyforge import TaskWorkerThread, default_settings, TASK_INTERPOLATION, TASK_FILTERING
from skellymodels.model_info.mediapipe_model_info import MediapipeModelInfo
from skellymodels.create_model_skeleton import create_mediapipe_skeleton_model
from openpose.scatter_3d import plot_3d_scatter
from apply_rigid_bones import apply_rigid_bones
path_to_recording_folder = Path(r'D:\2024-04-25_P01\1.0_recordings\sesh_2024-04-25_15_55_43_P01_WalkRun_Trial2')
joint_centers_frame_marker_dimension = np.load(path_to_recording_folder/'output_data'/'component_mediapipe_depth_pro'/'mediapipe_depth_pro_body_3d_xyz.npy')

post_process_task_worker = TaskWorkerThread(raw_skeleton_data=joint_centers_frame_marker_dimension, landmark_names=MediapipeModelInfo.landmark_names, task_list= [TASK_INTERPOLATION, TASK_FILTERING], settings=default_settings)
post_process_task_worker.run()
filt_interp_joint_centers_frame_marker_dimension = post_process_task_worker.tasks[TASK_FILTERING]['result']

mediapipe_skeleton = create_mediapipe_skeleton_model()
rigidified_data = apply_rigid_bones(data_to_rigidify=filt_interp_joint_centers_frame_marker_dimension, skeleton_model=mediapipe_skeleton)

plot_3d_scatter({'rigid_data':rigidified_data[300:500,:,:], 'filtered_interpolated': filt_interp_joint_centers_frame_marker_dimension[300:500,:,:], 'original': joint_centers_frame_marker_dimension[300:500,:,:]})



f= 2