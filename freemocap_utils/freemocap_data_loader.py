
from freemocap_config.folder_directory import DATA_FOLDER_NAME, MEDIAPIPE_3D_BODY_FILE_NAME

from pathlib import Path

import numpy as np

class FreeMoCapData():
    def __init__(self, path_to_freemocap_directory:Path, session_folder_name:str):

        self.load_mediapipe_body_data(path_to_freemocap_directory,session_folder_name)

    def load_mediapipe_body_data(self,path_to_freemocap_directory, session_folder_name):

        self.path_to_session_folder = path_to_freemocap_directory/session_folder_name
        self.path_to_mediapipe_body_data = self.path_to_session_folder/DATA_FOLDER_NAME/MEDIAPIPE_3D_BODY_FILE_NAME
        self.mediapipe_body_data = np.load(self.path_to_mediapipe_body_data)


