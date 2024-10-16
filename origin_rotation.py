from skellyforge import TaskWorkerThread
from skellymodels.model_info.mediapipe_model_info import MediapipeModelInfo
import numpy as np
from pathlib import Path
from skellyforge.freemocap_utils.config import default_settings
from skellyforge.freemocap_utils.constants import TASK_FINDING_GOOD_FRAME, TASK_SKELETON_ROTATION, TASK_FILTERING, TASK_INTERPOLATION, PARAM_CUTOFF_FREQUENCY, PARAM_AUTO_FIND_GOOD_FRAME, PARAM_GOOD_FRAME


path_to_recording_folder = Path(r'D:\sfn\michael_wobble\recording_12_07_09_gmt-5__MDN_wobble_3')
path_to_3d_data = path_to_recording_folder/'output_data'/'mediapipe_body_3d_xyz.npy'
path_to_centered_3d_data_folder = path_to_recording_folder/'output_data'/'origin_aligned_data'
path_to_centered_3d_data_folder.mkdir(exist_ok=True, parents=True)
path_to_centered_3d_data = path_to_centered_3d_data_folder/'mediapipe_body_3d_xyz.npy'

results_dict = {}



adjusted_setting = default_settings.copy()

adjusted_setting[TASK_FILTERING][PARAM_CUTOFF_FREQUENCY] = 10
adjusted_setting[TASK_SKELETON_ROTATION][PARAM_AUTO_FIND_GOOD_FRAME] = False
adjusted_setting[TASK_SKELETON_ROTATION][PARAM_GOOD_FRAME] = 34

def handle_task_completed(task_name, result):
    print(f"Task {task_name} completed. Result:")
    print(result)
    results_dict[task_name] = result

    if task_name == TASK_SKELETON_ROTATION:
        np.save(path_to_centered_3d_data, results_dict[TASK_SKELETON_ROTATION])

    

task_list = [TASK_INTERPOLATION, TASK_FILTERING, TASK_FINDING_GOOD_FRAME, TASK_SKELETON_ROTATION]

worker_thread = TaskWorkerThread(raw_skeleton_data=np.load(path_to_3d_data), 
                 landmark_names=MediapipeModelInfo.landmark_names,
                 task_list=task_list,
                settings=adjusted_setting,
                task_completed_callback=handle_task_completed
)

worker_thread.start()

f = 2