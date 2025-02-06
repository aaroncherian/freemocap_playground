
from pathlib import Path
import numpy as np
from skellyforge import TaskWorkerThread, default_settings, TASK_INTERPOLATION, TASK_FILTERING, TASK_FINDING_GOOD_FRAME, TASK_SKELETON_ROTATION


def run_skellyforge_rotation(raw_skeleton_data:np.ndarray, landmark_names:list,):
    post_process_task_worker = TaskWorkerThread(raw_skeleton_data=raw_skeleton_data, 
                                                landmark_names=landmark_names, 
                                                task_list= [TASK_INTERPOLATION, 
                                                            TASK_FILTERING,
                                                            TASK_FINDING_GOOD_FRAME,
                                                            TASK_SKELETON_ROTATION], 
                                                settings=default_settings)
    post_process_task_worker.run()
    filt_interp_joint_centers_frame_marker_dimension = post_process_task_worker.tasks[TASK_SKELETON_ROTATION]['result']
    print(f'Returning SkellyForged data for task {TASK_SKELETON_ROTATION} ')
    return filt_interp_joint_centers_frame_marker_dimension


if __name__ == '__main__':
    from skellymodels.model_info.mediapipe_model_info import MediapipeModelInfo
    from pathlib import Path

    path_to_recording_folder = Path(r'D:\2024-04-25_P01\1.0_recordings\sesh_2024-04-25_15_55_43_P01_WalkRun_Trial2')
    joint_centers_frame_marker_dimension = np.load(path_to_recording_folder/'output_data'/'component_mediapipe_depth_pro'/'mediapipe_depth_pro_body_3d_xyz.npy')
    run_skellyforge_rotation(joint_centers_frame_marker_dimension, MediapipeModelInfo.landmark_names)