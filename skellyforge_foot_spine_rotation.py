from pathlib import Path

from convert_and_export_spliced_data import split_and_export_data
import numpy as np

from skellyforge.freemocap_utils.postprocessing_widgets.task_worker_thread import TaskWorkerThread
from skellyforge.freemocap_utils.config import default_settings
from skellyforge.freemocap_utils.constants import (
    TASK_INTERPOLATION,
    TASK_FILTERING,
    TASK_FINDING_GOOD_FRAME,
    TASK_SKELETON_ROTATION,
)




TASK_POSTPROCESS_DATA = 'postprocess_data'
TASK_SPLIT_AND_EXPORT = 'split_and_export'




class DLCDataWorker:

    def __init__(self, path_to_recording_folder, output_folder_name, raw_data_file_name, task_list=[]):
        self.path_to_recording_folder = path_to_recording_folder
        self.output_folder_name = output_folder_name
        path_to_freemocap_data = path_to_recording_folder/output_folder_name/'raw_data'/raw_data_file_name
        self.freemocap_data = np.load(path_to_freemocap_data)
        
        self.available_tasks = {

            TASK_POSTPROCESS_DATA: self.post_process_data,
            TASK_SPLIT_AND_EXPORT: self.split_and_export_data
        }

        self.tasks = {task_name: {'function': self.available_tasks[task_name], 'result': None} for task_name in task_list}



    def post_process_data(self):

        raw_3d_data = self.freemocap_data
        # post_process_task_list = [TASK_INTERPOLATION, TASK_FILTERING, TASK_FINDING_GOOD_FRAME, TASK_SKELETON_ROTATION
        post_process_task_list = [TASK_INTERPOLATION, TASK_FILTERING, TASK_FINDING_GOOD_FRAME, TASK_SKELETON_ROTATION]
        post_process_settings = default_settings

        post_process_task_worker = TaskWorkerThread(raw_skeleton_data=raw_3d_data, task_list=post_process_task_list, settings=post_process_settings)
        post_process_task_worker.run()

        # return post_process_task_worker.tasks[TASK_SKELETON_ROTATION]['result']
        return post_process_task_worker.tasks[TASK_SKELETON_ROTATION]['result']
    
    def split_and_export_data(self):
        post_processed_data = self.tasks[TASK_POSTPROCESS_DATA]['result']
        return split_and_export_data(skel3d_frame_marker_xyz=post_processed_data, path_to_recording_folder=self.path_to_recording_folder, path_to_folder_where_we_will_save_this_data=self.path_to_recording_folder/output_folder_name)

    # def save_to_npy(self, task_name, data):
    #     """This takes care of saving all the raw data (2D DLC, 3D DLC, and the 3d spliced freemocap/dlc) to the raw data. Postprocessed stuff is handled in 'split_and_export' at the moment"""
    #     path_to_save_folder = self.path_to_recording_folder/self.output_folder_name/'raw_data'
    #     path_to_save_folder.mkdir(parents=True, exist_ok=True)
    #     filename_prefix = SAVE_CONFIG[task_name]
    #     filename = f"{filename_prefix}.npy"
    #     save_path = path_to_save_folder/ filename  # Assuming you want to save in a 'results' directory
    #     np.save(save_path, data)


    def run(self):
        for task_name, task_info in self.tasks.items():
            task_info['result'] = task_info['function']()

            # if task_name in SAVE_CONFIG:
            #     self.save_to_npy(task_name, task_info['result'])


def process_session_folder(session_folder_path, calibration_toml_path, tasks_to_run):
    session_folder = Path(session_folder_path)
    calibration_toml = Path(calibration_toml_path)
    
    # Iterate through each recording folder in the session folder
    for recording_folder in session_folder.iterdir():
        if recording_folder.is_dir():
            path_to_freemocap_data = recording_folder/'output_data'/'raw_data'/'reprojection_filtered_mediapipe3dData_numFrames_numTrackedPoints_spatialXYZ.npy'
            if path_to_freemocap_data.exists():  # Only proceed if the required data file exists
                worker = DLCDataWorker(recording_folder, calibration_toml, path_to_freemocap_data, tasks_to_run)
                worker.run()

if __name__ == '__main__':
    # session_folder_path = r'D:\2023-06-07_JH\1.0_recordings\treadmill_calib'
    # calibration_toml_path = r'D:\2023-06-07_TF01\1.0_recordings\sesh_2023-06-07_11_10_50_treadmill_calibration_01\sesh_2023-06-07_11_10_50_treadmill_calibration_01_camera_calibration.toml'
    tasks_to_run = [TASK_POSTPROCESS_DATA, TASK_SPLIT_AND_EXPORT]
    # process_session_folder(session_folder_path, calibration_toml_path, tasks_to_run)

    # recording_folder = Path(r'D:\2023-06-07_TF01\1.0_recordings\treadmill_calib\sesh_2023-06-07_12_50_56_TF01_leg_length_pos_25_trial_1')
    # path_to_freemocap_data = recording_folder/'mediapipe_yolo_dlc_output_data'/'raw_data'/'mediapipe3dData_numFrames_numTrackedPoints_spatialXYZ.npy'

    recording_folder = Path(r'D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\mediapipe_MDN_Trial_2_yolo')
    output_folder_name = 'output_data'
    raw_data_file_name = 'mediapipe3dData_numFrames_numTrackedPoints_spatialXYZ.npy'

    worker = DLCDataWorker(path_to_recording_folder=recording_folder, output_folder_name=output_folder_name, raw_data_file_name=raw_data_file_name, task_list=tasks_to_run)
    worker.run()

                    
