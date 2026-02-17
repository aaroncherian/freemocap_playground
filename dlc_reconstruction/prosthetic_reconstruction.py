import logging
from dlc_reconstruction.dlc_to_3d import process_recording_session, process_and_filter_data
from pathlib import Path
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
from skellymodels.models.tracking_model_info import ModelInfo, MediapipeModelInfo, RTMPoseModelInfo
from skellymodels.managers.human import Human
from skellymodels.managers.animal import Animal
from skellymodels.models.trajectory import Trajectory
from skellymodels.models.aspect import TrajectoryNames
import numpy as np
from copy import deepcopy

recording_root = Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera")

recordings_list = [
                    # recording_root/"sesh_2023-06-07_11_55_05_TF01_flexion_neg_5_6_trial_1",
                    # recording_root/"sesh_2023-06-07_12_03_15_TF01_flexion_neg_2_8_trial_1",
                    # recording_root/"sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1",
                    # recording_root/"sesh_2023-06-07_12_09_05_TF01_flexion_pos_2_8_trial_1",
                    # recording_root/"sesh_2023-06-07_12_12_36_TF01_flexion_pos_5_6_trial_1",
                    # recording_root/"sesh_2023-06-07_12_20_59_TF01_toe_angle_neg_6_trial_1",
                    # recording_root/"sesh_2023-06-07_12_25_38_TF01_toe_angle_neg_3_trial_1",
                    # recording_root/"sesh_2023-06-07_12_28_46_TF01_toe_angle_neutral_trial_1",
                    # recording_root/"sesh_2023-06-07_12_31_49_TF01_toe_angle_pos_3_trial_1",
                    # recording_root/"sesh_2023-06-07_12_34_37_TF01_toe_angle_pos_6_trial_1",
                    # recording_root/"sesh_2023-06-07_12_38_16_TF01_leg_length_neg_5_trial_1",
                    # recording_root/"sesh_2023-06-07_12_43_15_TF01_leg_length_neg_25_trial_1",
                    # recording_root/"sesh_2023-06-07_12_46_54_TF01_leg_length_neutral_trial_1",
                    # recording_root/"sesh_2023-06-07_12_50_56_TF01_leg_length_pos_25_trial_1",
                    recording_root/"sesh_2023-06-07_12_55_21_TF01_leg_length_pos_5_trial_1"
                    ]

    #for leg length positive, because of some length issues between qualisys and freemocap, use 'prosthetic_leg.body.xyz.as_array[:2785, dlc_idx[name], :]' to get the proper shape in the splicing lines

# path_to_recording = recording_root/"sesh_2023-06-07_11_55_05_TF01_flexion_neg_5_6_trial_1"
path_to_dlc_yaml= Path(r"C:\Users\aaron\Documents\GitHub\freemocap_playground\dlc_reconstruction\model_infos\prosthetic_leg.yaml")

# process_recording_session(
#     path_to_recording_folder=path_to_recording,
#     path_to_dlc_yaml=path_to_dlc_yaml,
#     use_skellyforge=True,
#     filter_order=4,
#     cutoff_frequency=7.0,
#     sampling_rate=30.0,
#     dlc_confidence_threshold=.5,
#     create_visualization=False,
#     interpolate=False
# )

for path_to_recording in recordings_list:
    model_info = RTMPoseModelInfo()
    tracker = "rtmpose"

    path_to_output_data = path_to_recording/'output_data'
    path_to_dlc_data = path_to_output_data/'dlc' 

    prosthetic_leg:Animal = Animal.from_data(path_to_dlc_data)

    human_model_info:ModelInfo = model_info


    human_rtmpose:Human = Human.from_data(path_to_output_data/tracker)
    # dlc_spliced_human_model_info = model_info
    # dlc_spliced_human_model_info.name = "rtmpose_dlc"

    human_rtmpose_dlc:Human = deepcopy(human_rtmpose)
    human_rtmpose_dlc.name = "rtmpose_dlc"
    human_rtmpose_dlc.tracker = "rtmpose_dlc"
    human_rtmpose_dlc.model_info.name = "rtmpose_dlc"

    # --- indices ---
    body_names = human_rtmpose_dlc.body.anatomical_structure.landmark_names
    rtmpose_idx = {n:i for i,n in enumerate(body_names)}

    dlc_names = prosthetic_leg.body.anatomical_structure.landmark_names
    dlc_idx = {n:i for i,n in enumerate(dlc_names)}

    # --- copy RTMPose body xyz ---
    body_xyz = human_rtmpose_dlc.body.xyz.as_array.copy()

    # DLC toe label (single point)
    dlc_toe = "right_foot_index"

    # Replace these by same-name if available in DLC
    for name in ["right_knee", "right_ankle", "right_heel"]:
        if name in rtmpose_idx and name in dlc_idx:
            body_xyz[:, rtmpose_idx[name], :] = prosthetic_leg.body.xyz.as_array[:, dlc_idx[name], :]

    # Replace BOTH toes using the single DLC toe
    for toe_name in ["right_big_toe", "right_small_toe"]:
        if toe_name in rtmpose_idx and dlc_toe in dlc_idx:
            body_xyz[:, rtmpose_idx[toe_name], :] = prosthetic_leg.body.xyz.as_array[:, dlc_idx[dlc_toe], :]
    # replace the body's XYZ trajectory (not the whole tracked_points)
    spliced_body_traj = Trajectory(
        name=TrajectoryNames.XYZ.value,
        array=body_xyz,
        landmark_names=body_names
    )
    human_rtmpose_dlc.body.add_trajectory({TrajectoryNames.XYZ.value: spliced_body_traj})
    for aspect_name, aspect in human_rtmpose_dlc.aspects.items():
        aspect.metadata["tracker_type"] = 'rtmpose_dlc'     


    out = path_to_output_data / "rtmpose_dlc"
    out.mkdir(parents=True, exist_ok=True)
    human_rtmpose_dlc.save_out_all_xyz_numpy_data(out)
    human_rtmpose_dlc.save_out_all_data_parquet(out)
    human_rtmpose_dlc.save_out_all_data_csv(out)
    human_rtmpose_dlc.save_out_numpy_data(out)
    human_rtmpose_dlc.save_out_csv_data(out)


# human_mp_dlc:Human = Human.from_tracked_points_numpy_array(name = "human", model_info = dlc_spliced_human_model_info, tracked_points_numpy_array= human_data_processed)



# markers_to_splice = prosthetic_leg.body.anatomical_structure.landmark_names

# marker_indices = [human_mp_dlc.body.anatomical_structure.landmark_names.index(marker_name) for marker_name in markers_to_splice]

# mediapipe_dlc_array = human_mp_dlc.body.xyz.as_array.copy()
# mediapipe_dlc_array[:,marker_indices,:] = prosthetic_leg.body.xyz.as_array
# spliced_trajectory = Trajectory(
#     name = TrajectoryNames.XYZ.value,
#     array = mediapipe_dlc_array,
#     landmark_names= human_mp_dlc.body.anatomical_structure.landmark_names
# )
# human_mp_dlc.body.add_trajectory(
#     {TrajectoryNames.XYZ.value:spliced_trajectory}
# )

# human_mp_dlc.calculate()

# path_to_save_mediapipe_dlc_data = path_to_output_data/'mediapipe_dlc'
# path_to_save_mediapipe_dlc_data.mkdir(parents=True, exist_ok=True)

# human_mp_dlc.save_out_numpy_data(path_to_save_mediapipe_dlc_data)
# human_mp_dlc.save_out_csv_data(path_to_save_mediapipe_dlc_data)
# human_mp_dlc.save_out_all_data_csv(path_to_save_mediapipe_dlc_data)
# human_mp_dlc.save_out_all_data_parquet(path_to_save_mediapipe_dlc_data)
# human_mp_dlc.save_out_all_xyz_numpy_data(path_to_save_mediapipe_dlc_data)
# f = 2