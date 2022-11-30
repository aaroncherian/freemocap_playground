from freemocap_utils import freemocap_data_loader

from pathlib import Path

path_to_freemocap_folder = Path(r'D:\ValidationStudy_aaron\FreeMoCap_Data')
sessionID = 'sesh_2022-11-02_13_55_55_atc_nih_balance'

freemocap_data = freemocap_data_loader.FreeMoCapData(path_to_freemocap_folder,sessionID)

freemocap_body_data = freemocap_data.mediapipe_body_data

f =2
