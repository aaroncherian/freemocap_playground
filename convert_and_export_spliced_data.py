import numpy as np
from pathlib import Path
from analysis.calculate_center_of_mass import run_center_of_mass_calculations
from export.convert_mediapipe_npy_to_csv import convert_mediapipe_npy_to_csv
from typing import Union
import pandas as pd

from freemocap.core_processes.post_process_skeleton_data.estimate_skeleton_segment_lengths import (
    estimate_skeleton_segment_lengths,
    mediapipe_skeleton_segment_definitions,
)

from freemocap.core_processes.detecting_things_in_2d_images.mediapipe_stuff.data_models.mediapipe_skeleton_names_and_connections import (
    mediapipe_names_and_connections_dict,
)
from freemocap.utilities.save_dictionary_to_json import save_dictionary_to_json
from freemocap.system.paths_and_filenames.file_and_folder_names import (
    MEDIAPIPE_BODY_3D_DATAFRAME_CSV_FILE_NAME,
    CENTER_OF_MASS_FOLDER_NAME,
    SEGMENT_CENTER_OF_MASS_NPY_FILE_NAME,
    TOTAL_BODY_CENTER_OF_MASS_NPY_FILE_NAME
)


def save_skeleton_array_to_npy(
    array_to_save: np.ndarray, skeleton_file_name: str, path_to_folder_where_we_will_save_this_data: Union[str, Path]
):
    if not skeleton_file_name.endswith(".npy"):
        skeleton_file_name += ".npy"
    Path(path_to_folder_where_we_will_save_this_data).mkdir(parents=True, exist_ok=True)
    np.save(
        str(Path(path_to_folder_where_we_will_save_this_data) / skeleton_file_name),
        array_to_save,
    )


def split_and_export_data(skel3d_frame_marker_xyz, path_to_recording_folder, path_to_folder_where_we_will_save_this_data):
    path_to_folder_where_we_will_save_this_data.mkdir(parents=True, exist_ok=True)
    save_skeleton_array_to_npy(
        array_to_save=skel3d_frame_marker_xyz,
        skeleton_file_name="mediaPipeSkel_3d_body_hands_face.npy",
        path_to_folder_where_we_will_save_this_data=path_to_folder_where_we_will_save_this_data,
    )

    segment_COM_frame_imgPoint_XYZ, totalBodyCOM_frame_XYZ = run_center_of_mass_calculations(
        processed_skel3d_frame_marker_xyz=skel3d_frame_marker_xyz
    )

    save_skeleton_array_to_npy(
        array_to_save=segment_COM_frame_imgPoint_XYZ,
        skeleton_file_name=SEGMENT_CENTER_OF_MASS_NPY_FILE_NAME,
        path_to_folder_where_we_will_save_this_data=Path(path_to_folder_where_we_will_save_this_data)
        / CENTER_OF_MASS_FOLDER_NAME,
    )

    save_skeleton_array_to_npy(
        array_to_save=totalBodyCOM_frame_XYZ,
        skeleton_file_name=TOTAL_BODY_CENTER_OF_MASS_NPY_FILE_NAME,
        path_to_folder_where_we_will_save_this_data=Path(path_to_folder_where_we_will_save_this_data)
        / CENTER_OF_MASS_FOLDER_NAME,
    )



    # break up big NPY and save out csv's
    convert_mediapipe_npy_to_csv(
        mediapipe_3d_frame_trackedPoint_xyz=skel3d_frame_marker_xyz,
        output_data_folder_path=path_to_folder_where_we_will_save_this_data,
    )

    path_to_skeleton_body_csv = (
        path_to_folder_where_we_will_save_this_data / MEDIAPIPE_BODY_3D_DATAFRAME_CSV_FILE_NAME
    )
    skeleton_dataframe = pd.read_csv(path_to_skeleton_body_csv)

    skeleton_segment_lengths_dict = estimate_skeleton_segment_lengths(
        skeleton_dataframe=skeleton_dataframe,
        skeleton_segment_definitions=mediapipe_skeleton_segment_definitions,
    )

    save_dictionary_to_json(
        save_path=path_to_folder_where_we_will_save_this_data,
        file_name="mediapipe_skeleton_segment_lengths.json",
        dictionary=skeleton_segment_lengths_dict,
    )

    save_dictionary_to_json(
        save_path=path_to_folder_where_we_will_save_this_data,
        file_name="mediapipe_names_and_connections_dict.json",
        dictionary=mediapipe_names_and_connections_dict,
    )

    # export_to_blender.export_to_blender(
    #     recording_folder_path= path_to_recording_folder,
    #     blender_file_path=get_blender_file_path(path_to_recording_folder),
    #     blender_exe_path=get_best_guess_of_blender_path.get_best_guess_of_blender_path(),
    # )


if __name__ == "__main__":
    path_to_recording_folder = Path(r'D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_13_48_44_MDN_treadmill_2')
    path_to_np_array = path_to_recording_folder/'output_data'/'mediapipe_postprocessed_3d_xyz.npy'
    path_to_folder_where_we_will_save_this_data = path_to_recording_folder/'output_data'

    skel3d_frame_marker_xyz = np.load(path_to_np_array)
    split_and_export_data(skel3d_frame_marker_xyz, path_to_recording_folder, path_to_folder_where_we_will_save_this_data)