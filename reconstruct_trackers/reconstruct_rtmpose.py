
from __future__ import annotations
from pathlib import Path
import logging
import numpy as np
from typing import List, Optional, Union

from reconstruct_trackers.reconstruction_to_3d.reconstruction import reconstruct_3d
from reconstruct_trackers.reconstruction_to_3d.visualization import plot_3d_scatter
from reconstruct_trackers.reconstruction_to_3d.postprocessing import process_and_filter_data
from skellymodels.managers.human import Human
from skellymodels.models.tracking_model_info import ModelInfo, RTMPoseModelInfo, ViTPoseWholeBodyModelInfo, ViTPose25ModelInfo, MediapipeModelInfo
# Configure logger
logger = logging.getLogger(__name__)


def process_recording_session(
    path_to_recording_folder: Union[str, Path],
    model_info: ModelInfo,
    path_to_calibration_toml: Optional[Path] = None,
    use_skellyforge: bool = True,
    filter_order: int = 4,
    cutoff_frequency: float = 6.0,
    sampling_rate: float = 90.0,
    landmark_names: Optional[List[str]] = None,
    create_visualization: bool = True,
) -> np.ndarray:
    """
    Process a recording session from 2D DLC data to 3D reconstruction with optional filtering.
    
    Args:
        path_to_recording_folder: Path to the recording folder
        path_to_calibration_toml: Path to the calibration TOML file (if None, will search in recording folder)
        path_to_folder_of_dlc_csvs: Path to the folder containing DLC CSV files (if None, will use 'dlc_data' subfolder)
        use_skellyforge: Whether to use SkellyForge for filtering the data
        filter_order: Order of the Butterworth filter
        cutoff_frequency: Cutoff frequency for the Butterworth filter
        sampling_rate: Sampling rate of the data in Hz
        confidence_threshold: Confidence threshold for DLC data
        landmark_names: Names of the landmarks (if None, will use default landmarks)
        create_visualization: Whether to create a 3D visualization
        
    Returns:
        Processed 3D data array
    """

    # Convert path to Path object if it's a string
    path_to_recording_folder = Path(path_to_recording_folder)
    
    # Set default paths if not provided
    if path_to_calibration_toml is None:
        path_to_calibration_toml = list(path_to_recording_folder.glob('*calibration.toml'))[0]
    
    # Set default landmark names if not provided
    landmark_names = model_info.tracked_point_names
    tracker_name = model_info.name

    # Output directory setup
    path_to_output_folder = path_to_recording_folder / 'output_data' / tracker_name
    path_to_output_folder.mkdir(parents=True, exist_ok=True)

    path_to_raw_data_folder = path_to_output_folder/ 'raw_data'
    path_to_raw_data_folder.mkdir(parents=True, exist_ok=True)  
    path_to_2d_data = path_to_raw_data_folder/f'{tracker_name}_2dData_numCams_numFrames_numTrackedPoints_pixelXY.npy'
    data_2d = np.load(path_to_2d_data)[:,:,:,:2]  # Load 2D data and keep only x and y coordinates

    logger.info("Reconstructing 3D data")
    data_3d = reconstruct_3d(data_2d, path_to_calibration_toml)
    
    # Save raw 3D data
    logger.info("Saving raw 3D data")
    np.save(
        path_to_raw_data_folder / f'{tracker_name}_3dData_numFrames_numTrackedPoints_spatialXYZ.npy', 
        data_3d
    )

    # Apply filtering if requested
    if use_skellyforge:
        logger.info("Applying SkellyForge filters to 3D data")
        data_3d = process_and_filter_data(
            data_3d,
            landmark_names,
            cutoff_frequency,
            sampling_rate,
            filter_order
        )

    logger.info("Processing with skeleton model")

    
    skeleton = Human.from_tracked_points_numpy_array( #not a human but it still does the job
        name="human",
        model_info=model_info,
        tracked_points_numpy_array=data_3d
    )
    skeleton.calculate()
    skeleton.save_out_numpy_data(path_to_output_folder=path_to_output_folder)
    skeleton.save_out_csv_data(path_to_output_folder=path_to_output_folder)
    skeleton.save_out_all_data_csv(path_to_output_folder=path_to_output_folder)
    skeleton.save_out_all_data_parquet(path_to_output_folder=path_to_output_folder)
    skeleton.save_out_all_xyz_numpy_data(path_to_output_folder=path_to_output_folder)

    if create_visualization:
        logger.info("Creating 3D visualization")
        data_dict = {'dlc_data': data_3d}
        plot_3d_scatter(data_dict)
    
    return data_3d


# def main():

#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    

#     # recording_root = Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera")

#     # recordings_list = [recording_root/"sesh_2023-06-07_12_38_16_TF01_leg_length_neg_5_trial_1",
#     #                    recording_root/"sesh_2023-06-07_12_43_15_TF01_leg_length_neg_25_trial_1",
#     #                    recording_root/"sesh_2023-06-07_12_46_54_TF01_leg_length_neutral_trial_1",
#     #                    recording_root/"sesh_2023-06-07_12_50_56_TF01_leg_length_pos_25_trial_1",
#     #                    recording_root/"sesh_2023-06-07_12_55_21_TF01_leg_length_pos_5_trial_1"]



#     recordings_list = [r"D:\2025_07_31_JSM_pilot\freemocap\2025-07-31_16-52-16_GMT-4_jsm_treadmill_2",
#                        r"D:\2025_09_03_OKK\freemocap\2025-09-03_14-56-30_GMT-4_okk_treadmill_1",
#                        r"D:\2025_09_03_OKK\freemocap\2025-09-03_15-04-04_GMT-4_okk_treadmill_2",
#                        r"D:\2025-11-04_ATC\2025-11-04_15-33-01_GMT-5_atc_treadmill_1",
#                        r"D:\2025-11-04_ATC\2025-11-04_15-44-06_GMT-5_atc_treadmill_2",
#                        r"D:\2026_01_26_KK\2026-01-16_14-15-39_GMT-5_kk_treadmill_1",
#                        r"D:\2026_01_26_KK\2026-01-16_14-25-46_GMT-5_kk_treadmill_2",
#                        r"D:\2026-01-30-JTM\2026-01-30_11-21-06_GMT-5_JTM_treadmill_1",
#                        r"D:\2026-01-30-JTM\2026-01-30_11-32-56_GMT-5_JTM_treadmill_2"
#                        ]

#     for recording in recordings_list:
#         process_recording_session(
#             path_to_recording_folder=recording,
#             model_info=RTMPoseModelInfo(),
#             use_skellyforge=True,
#             filter_order=4,
#             cutoff_frequency=7.0,
#             sampling_rate=30.0,
#             create_visualization=False
#         )
#     # process_recording_session(
#     #     path_to_recording_folder=r'D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_28_46_TF01_toe_angle_neutral_trial_1',
#     #     model_info=ViTPoseWholeBodyModelInfo(),
#     #     use_skellyforge=True,
#     #     filter_order=4,
#     #     cutoff_frequency=7.0,
#     #     sampling_rate=30.0,
#     #     create_visualization=True
#     # )


# if __name__ == '__main__':
#     main()


import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# If these are in your project, import them as usual:
# from yourpkg import process_recording_session, RTMPoseModelInfo


def _init_worker_logging():
    # Each process configures its own logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(processName)s - %(name)s - %(levelname)s - %(message)s",
    )


def _process_one(recording: str) -> tuple[str, bool, str | None]:
    """
    Runs one recording. Returns (recording, success, error_message).
    Important: create RTMPoseModelInfo() inside the subprocess to avoid pickling issues.
    """
    _init_worker_logging()
    log = logging.getLogger("worker")

    try:
        log.info("Starting: %s", recording)

        process_recording_session(
            path_to_recording_folder=recording,
            model_info=MediapipeModelInfo(),
            use_skellyforge=True,
            filter_order=4,
            cutoff_frequency=6.0,
            sampling_rate=30.0,
            create_visualization=False,
        )

        log.info("Finished: %s", recording)
        return (recording, True, None)

    except Exception as e:
        log.exception("FAILED: %s", recording)
        return (recording, False, f"{type(e).__name__}: {e}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    log = logging.getLogger("main")

    # recordings_list = [
    #     # r"D:\validation\data\2025_09_03_OKK\freemocap\2025-09-03_14-56-30_GMT-4_okk_treadmill_1",
    #     # r"D:\validation\data\2025_09_03_OKK\freemocap\2025-09-03_15-04-04_GMT-4_okk_treadmill_2",
    #     # r"D:\validation\data\2025_09_03_OKK\freemocap\2025-09-03_14-24-21_GMT-4_okk_nih_1",
    #     # r"D:\validation\data\2025_09_03_OKK\freemocap\2025-09-03_14-38-45_GMT-4_okk_nih_2",
    #     # r"D:\validation\data\2025-11-04_ATC\2025-11-04_15-33-01_GMT-5_atc_treadmill_1",
    #     # r"D:\validation\data\2025-11-04_ATC\2025-11-04_15-44-06_GMT-5_atc_treadmill_2",
    #     r"D:\validation\data\2025-11-04_ATC\2025-11-04_15-18-21_GMT-5_atc_nih_2",
    #     r"D:\validation\data\2025-11-04_ATC\2025-11-04_15-02-28_GMT-5_atc_nih_1",
    #     r"D:\validation\data\2025_07_31_JSM_pilot\freemocap\2025-07-31_16-00-42_GMT-4_jsm_nih_trial_1",
    #     r"D:\validation\data\2025_07_31_JSM_pilot\freemocap\2025-07-31_16-16-23_GMT-4_jsm_nih_trial_2",
    #     r"D:\validation\data\2025_07_31_JSM_pilot\freemocap\2025-07-31_16-35-10_GMT-4_jsm_treadmill_trial_1",
    #     r"D:\validation\data\2025_07_31_JSM_pilot\freemocap\2025-07-31_16-52-16_GMT-4_jsm_treadmill_2",
    #     r"D:\validation\data\2026_01_26_KK\2026-01-16_13-41-17_GMT-5_kk_nih_1",
    #     r"D:\validation\data\2026_01_26_KK\2026-01-16_13-58-41_GMT-5_kk_nih_2",
    #     r"D:\validation\data\2026_01_26_KK\2026-01-16_14-15-39_GMT-5_kk_treadmill_1",
    #     r"D:\validation\data\2026_01_26_KK\2026-01-16_14-25-46_GMT-5_kk_treadmill_2",
    #     r"D:\validation\data\2026-01-30-JTM\2026-01-30_11-21-06_GMT-5_JTM_treadmill_1",
    #     r"D:\validation\data\2026-01-30-JTM\2026-01-30_11-32-56_GMT-5_JTM_treadmill_2",
    #     r"D:\validation\data\2026-01-30-JTM\2026-01-30_10-40-03_GMT-5_JTM_nih_1",
    #     r"D:\validation\data\2026-01-30-JTM\2026-01-30_10-57-13_GMT-5_JTM_nih_2"

    # # ]
    # recording_root = Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera")
    # # recordings_list = [recording_root/"sesh_2023-06-07_12_46_54_TF01_leg_length_neutral_trial_1"]
    # recordings_list = [recording_root/"sesh_2023-06-07_11_55_05_TF01_flexion_neg_5_6_trial_1",
    #                    recording_root/"sesh_2023-06-07_12_03_15_TF01_flexion_neg_2_8_trial_1",
    #                    recording_root/"sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1",
    #                    recording_root/"sesh_2023-06-07_12_09_05_TF01_flexion_pos_2_8_trial_1",
    #                    recording_root/"sesh_2023-06-07_12_12_36_TF01_flexion_pos_5_6_trial_1",
    #                    recording_root/"sesh_2023-06-07_12_20_59_TF01_toe_angle_neg_6_trial_1",
    #                    recording_root/"sesh_2023-06-07_12_25_38_TF01_toe_angle_neg_3_trial_1",
    #                    recording_root/"sesh_2023-06-07_12_28_46_TF01_toe_angle_neutral_trial_1",
    #                    recording_root/"sesh_2023-06-07_12_31_49_TF01_toe_angle_pos_3_trial_1",
    #                    recording_root/"sesh_2023-06-07_12_34_37_TF01_toe_angle_pos_6_trial_1"
    #                    ]

    recording_root = Path(r"D:\validation\data\2026_03_04_ML")
    
    recordings_list = [
                    recording_root / "2026-03-04_19-12-07_GMT-5_ml_nih_trial_1",
                    #    recording_root / "2026-03-04_19-27-37_GMT-5_ml_nih_trial_2",
                    #    recording_root / "2026-03-04_19-40-22_GMT-5_ml_treadmill_trial_1",
                    #    recording_root / "2026-03-04_19-49-19_GMT-5_ml_treadmill_trial_2"
                       ]

    # For Windows + GPU-heavy workloads, you often want fewer workers than CPU cores.
    max_workers = min(5, os.cpu_count() or 4)

    log.info("Launching %d workers for %d recordings", max_workers, len(recordings_list))

    results: list[tuple[str, bool, str | None]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_process_one, rec): rec for rec in recordings_list}

        for fut in as_completed(futures):
            results.append(fut.result())

    ok = [r for r in results if r[1]]
    bad = [r for r in results if not r[1]]

    log.info("Done. Success: %d, Failed: %d", len(ok), len(bad))
    for rec, success, err in bad:
        log.error("FAILED: %s | %s", rec, err)


if __name__ == "__main__":
    # REQUIRED on Windows to avoid recursive process spawning
    main()