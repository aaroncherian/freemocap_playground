from skellymodels.models.tracking_model_info import ModelInfo, MediapipeModelInfo
from skellymodels.managers.human import Human
from skellymodels.managers.animal import Animal
from skellymodels.models.trajectory import Trajectory
from skellymodels.models.aspect import TrajectoryNames
from pathlib import Path
import numpy as np
import logging
from dlc_reconstruction.reconstruction_to_3d.postprocessing import process_and_filter_data


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

path_to_recording = Path(r'D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1')

path_to_output_data = path_to_recording/'output_data'
path_to_dlc_data = path_to_output_data/'dlc'

prosthetic_leg:Animal = Animal.from_data(path_to_dlc_data)

human_model_info:ModelInfo = MediapipeModelInfo()
human_data_raw = np.load(path_to_output_data/'raw_data'/'mediapipe_3dData_numFrames_numTrackedPoints_spatialXYZ.npy') 

human_data_processed = process_and_filter_data(
    data_3d = human_data_raw,
    landmark_names = human_model_info.tracked_point_names,
    cutoff_frequency=6,
    sampling_rate=30,
    filter_order=4
)


human_mediapipe:Human = Human.from_tracked_points_numpy_array(name = "human", model_info = human_model_info, tracked_points_numpy_array= human_data_processed)

human_mediapipe.calculate()

path_to_save_mediapipe_data = path_to_output_data/'mediapipe'
path_to_save_mediapipe_data.mkdir(parents=True, exist_ok=True)
human_mediapipe.save_out_numpy_data(path_to_save_mediapipe_data)
human_mediapipe.save_out_csv_data(path_to_save_mediapipe_data)
human_mediapipe.save_out_all_data_csv(path_to_save_mediapipe_data)
human_mediapipe.save_out_all_data_parquet(path_to_save_mediapipe_data)
human_mediapipe.save_out_all_xyz_numpy_data(path_to_save_mediapipe_data)


dlc_spliced_human_model_info = MediapipeModelInfo()
dlc_spliced_human_model_info.name = 'mediapipe_dlc'

human_mp_dlc:Human = Human.from_tracked_points_numpy_array(name = "human", model_info = dlc_spliced_human_model_info, tracked_points_numpy_array= human_data_processed)

markers_to_splice = prosthetic_leg.body.anatomical_structure.landmark_names

marker_indices = [human_mp_dlc.body.anatomical_structure.landmark_names.index(marker_name) for marker_name in markers_to_splice]

mediapipe_dlc_array = human_mp_dlc.body.xyz.as_array.copy()
mediapipe_dlc_array[:,marker_indices,:] = prosthetic_leg.body.xyz.as_array
spliced_trajectory = Trajectory(
    name = TrajectoryNames.XYZ.value,
    array = mediapipe_dlc_array,
    landmark_names= human_mp_dlc.body.anatomical_structure.landmark_names
)
human_mp_dlc.body.add_trajectory(
    {TrajectoryNames.XYZ.value:spliced_trajectory}
)

human_mp_dlc.calculate()

path_to_save_mediapipe_dlc_data = path_to_output_data/'mediapipe_dlc'
path_to_save_mediapipe_dlc_data.mkdir(parents=True, exist_ok=True)

human_mp_dlc.save_out_numpy_data(path_to_save_mediapipe_dlc_data)
human_mp_dlc.save_out_csv_data(path_to_save_mediapipe_dlc_data)
human_mp_dlc.save_out_all_data_csv(path_to_save_mediapipe_dlc_data)
human_mp_dlc.save_out_all_data_parquet(path_to_save_mediapipe_dlc_data)
human_mp_dlc.save_out_all_xyz_numpy_data(path_to_save_mediapipe_dlc_data)
f = 2