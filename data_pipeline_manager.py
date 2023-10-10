from pathlib import Path
# from compile_dlc_csvs import compile_dlc_csvs
# from reconstruct_dlc_data import reconstruct_dlc_data
# from splice_3d_data import splice_freemocap_and_dlc_leg_data
from convert_and_export_spliced_data import split_and_export_data
import numpy as np

from skellyforge.freemocap_utils.postprocessing_widgets.task_worker_thread import TaskWorkerThread
from skellyforge.freemocap_utils.config import default_settings
from skellyforge.freemocap_utils.constants import (
    TASK_INTERPOLATION,
    TASK_FILTERING,
    TASK_FINDING_GOOD_FRAME,
    TASK_SKELETON_ROTATION
)



TASK_COMPILE_CSVS = 'compile_dlc_csvs'
TASK_RECONSTRUCT_DLC_DATA = 'reconstruct_dlc_data'
TASK_SPLICE_3D_DATA = 'splice_3d_data'
TASK_POSTPROCESS_DATA = 'postprocess_data'
TASK_SPLIT_AND_EXPORT = 'split_and_export'


SAVE_CONFIG = {
    TASK_COMPILE_CSVS: 'deeplabcut2dData_numCams_numFrames_numTrackedPoints_pixelXY',
    TASK_RECONSTRUCT_DLC_DATA: 'deeplabcut3dData_numFrames_numTrackedPoints_spatialXYZ',
    TASK_SPLICE_3D_DATA: 'mediapipe_deeplabcut_3dData_spliced',
}

class DLCDataWorker:

    def __init__(self, path_to_recording_folder, path_to_calibration_toml, path_to_freemocap_data, task_list=[]):
        self.path_to_recording_folder = path_to_recording_folder
        self.path_to_calibration_toml = path_to_calibration_toml
        self.path_to_freemocap_data = path_to_freemocap_data
        self.freemocap_data = np.load(self.path_to_freemocap_data)
        
        self.available_tasks = {
            TASK_POSTPROCESS_DATA: self.post_process_data,
            TASK_SPLIT_AND_EXPORT: self.split_and_export_data
        }

        self.tasks = {task_name: {'function': self.available_tasks[task_name], 'result': None} for task_name in task_list}

    
    def post_process_data(self):

        raw_spliced_3d_data = self.tasks[TASK_SPLICE_3D_DATA]['result']
        post_process_task_list = [TASK_INTERPOLATION, TASK_FILTERING, TASK_FINDING_GOOD_FRAME, TASK_SKELETON_ROTATION]
        post_process_settings = default_settings

        post_process_task_worker = TaskWorkerThread(raw_skeleton_data=raw_spliced_3d_data, task_list=post_process_task_list, settings=post_process_settings)
        post_process_task_worker.run()

        return post_process_task_worker.tasks[TASK_SKELETON_ROTATION]['result']
    
    def split_and_export_data(self):
        post_processed_data = self.tasks[TASK_POSTPROCESS_DATA]['result']
        return split_and_export_data(skel3d_frame_marker_xyz=post_processed_data, path_to_recording_folder=self.path_to_recording_folder, path_to_folder_where_we_will_save_this_data=self.path_to_recording_folder/'output_data')

    def save_to_npy(self, task_name, data):
        """This takes care of saving all the raw data (2D DLC, 3D DLC, and the 3d spliced freemocap/dlc) to the raw data. Postprocessed stuff is handled in 'split_and_export' at the moment"""
        filename_prefix = SAVE_CONFIG[task_name]
        filename = f"{filename_prefix}.npy"
        save_path = self.path_to_recording_folder/'output_data'/'raw_data'/ filename  # Assuming you want to save in a 'results' directory
        np.save(save_path, data)


    def run(self):
        for task_name, task_info in self.tasks.items():
            task_info['result'] = task_info['function']()

            if task_name in SAVE_CONFIG:
                self.save_to_npy(task_name, task_info['result'])


def process_session_folder(session_folder_path, calibration_toml_path, tasks_to_run):
    session_folder = Path(session_folder_path)
    calibration_toml = Path(calibration_toml_path)
    
    # Iterate through each recording folder in the session folder
    for recording_folder in session_folder.iterdir():
        if recording_folder.is_dir():
            path_to_freemocap_data = recording_folder/'output_data'/'raw_data'/'mediapipe3dData_numFrames_numTrackedPoints_spatialXYZ.npy'
            if path_to_freemocap_data.exists():  # Only proceed if the required data file exists
                worker = DLCDataWorker(recording_folder, calibration_toml, path_to_freemocap_data, tasks_to_run)
                worker.run()

if __name__ == '__main__':
    session_folder_path = r'D:\2023-06-07_JH\1.0_recordings\treadmill_calib'
    calibration_toml_path = r'D:\2023-06-07_JH\1.0_recordings\sesh_2023-06-07_11_10_50_treadmill_calibration_01\sesh_2023-06-07_11_10_50_treadmill_calibration_01_camera_calibration.toml'
    tasks_to_run = [TASK_COMPILE_CSVS, TASK_RECONSTRUCT_DLC_DATA, TASK_SPLICE_3D_DATA, TASK_POSTPROCESS_DATA, TASK_SPLIT_AND_EXPORT]
    process_session_folder(session_folder_path, calibration_toml_path, tasks_to_run)

