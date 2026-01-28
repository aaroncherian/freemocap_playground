
from pathlib import Path
import logging
import numpy as np
from typing import List, Optional, Union

from reconstruct_trackers.reconstruction_to_3d.reconstruction import reconstruct_3d
from reconstruct_trackers.reconstruction_to_3d.visualization import plot_3d_scatter
from reconstruct_trackers.reconstruction_to_3d.postprocessing import process_and_filter_data
from skellymodels.managers.human import Human
from skellymodels.models.tracking_model_info import ModelInfo, RTMPoseModelInfo, ViTPoseWholeBodyModelInfo, ViTPose25ModelInfo
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

    path_to_raw_data_folder = path_to_output_folder / 'raw_data'
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

    if create_visualization:
        logger.info("Creating 3D visualization")
        data_dict = {'dlc_data': data_3d}
        plot_3d_scatter(data_dict)
    
    return data_3d


def main():

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    process_recording_session(
        path_to_recording_folder=r'D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_28_46_TF01_toe_angle_neutral_trial_1',
        model_info=ViTPoseWholeBodyModelInfo(),
        use_skellyforge=True,
        filter_order=4,
        cutoff_frequency=7.0,
        sampling_rate=30.0,
        create_visualization=True
    )


if __name__ == '__main__':
    main()