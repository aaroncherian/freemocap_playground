import logging
from dlc_reconstruction.dlc_to_3d import process_recording_session, process_and_filter_data
from pathlib import Path
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
from skellymodels.models.tracking_model_info import ModelInfo, MediapipeModelInfo
from skellymodels.managers.human import Human
from skellymodels.managers.animal import Animal
from skellymodels.models.trajectory import Trajectory
from skellymodels.models.aspect import TrajectoryNames
import numpy as np

path_to_recording = Path(r'D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_38_16_TF01_leg_length_neg_5_trial_1_ajc')
path_to_dlc_yaml= Path(r"C:\Users\aaron\Documents\GitHub\freemocap_playground\dlc_reconstruction\model_infos\prosthetic_leg.yaml")

process_recording_session(
    path_to_recording_folder=path_to_recording,
    path_to_dlc_yaml=path_to_dlc_yaml,
    use_skellyforge=True,
    filter_order=4,
    cutoff_frequency=7.0,
    sampling_rate=30.0,
    dlc_confidence_threshold=.5,
    create_visualization=False,
    interpolate=False
)



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