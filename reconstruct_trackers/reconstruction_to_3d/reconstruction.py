
from pathlib import Path
import numpy as np
import logging
import multiprocessing
from typing import Union, Tuple, Optional, Any

from reconstruct_trackers.anipose_utils.anipose_object_loader import load_anipose_calibration_toml_from_path

# Configure logger
logger = logging.getLogger(__name__)


def reconstruct_3d(data_2d: np.ndarray, calibration_toml_path: Union[str, Path]) -> np.ndarray:
    """
    Reconstruct 3D data from 2D tracking data using camera calibration information.
    
    Args:
        data_2d: 2D tracking data with shape (n_cameras, n_frames, n_points, 2)
        calibration_toml_path: Path to the Anipose calibration TOML file
        
    Returns:
        Reconstructed 3D data with shape (n_frames, n_points, 3)
    """
    anipose_calibration_object = load_anipose_calibration_toml_from_path(calibration_toml_path)

    data_3d, reprojection_error, camera_reprojection_error = triangulate_3d_data(
        anipose_calibration_object=anipose_calibration_object,
        data_2d=data_2d,
        use_triangulate_ransac=False,
        kill_event=None,
    )

    return data_3d


def triangulate_3d_data(
    anipose_calibration_object: Any,
    data_2d: np.ndarray,
    use_triangulate_ransac: bool = False,
    kill_event: Optional[multiprocessing.Event] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Triangulate 3D points from 2D camera views using Anipose calibration.
    
    Args:
        anipose_calibration_object: Anipose calibration object
        data_2d: 2D tracking data with shape (n_cameras, n_frames, n_points, 2)
        use_triangulate_ransac: Whether to use RANSAC for triangulation
        kill_event: Multiprocessing event to stop processing
        
    Returns:
        Tuple containing:
        - 3D triangulated data with shape (n_frames, n_points, 3)
        - Reprojection error per point with shape (n_frames, n_points)
        - Reprojection error per camera with shape (n_cameras, n_frames, n_points)
    """
    number_of_cameras = data_2d.shape[0]
    number_of_frames = data_2d.shape[1]
    number_of_tracked_points = data_2d.shape[2]
    number_of_spatial_dimensions = data_2d.shape[3]

    if number_of_spatial_dimensions != 2:
        logger.error(f"Expected 2D data but got {number_of_spatial_dimensions} dimensions")
        raise ValueError("Input data must have 2 spatial dimensions")

    data2d_flat = data_2d.reshape(number_of_cameras, -1, 2)

    logger.info(
        f"Reconstructing 3D points from 2D data with shape: \n"
        f"number_of_cameras: {number_of_cameras},\n"
        f"number_of_frames: {number_of_frames}, \n"
        f"number_of_tracked_points: {number_of_tracked_points}"
    )

    # Triangulate 2D points to 3D
    if use_triangulate_ransac:
        logger.info("Using RANSAC triangulation method")
        data3d_flat = anipose_calibration_object.triangulate_ransac(data2d_flat, progress=True, kill_event=kill_event)
    else:
        logger.info("Using standard triangulation method")
        data3d_flat = anipose_calibration_object.triangulate(data2d_flat, progress=True, kill_event=kill_event)

    # Reshape to frames × points × xyz
    data3d = data3d_flat.reshape(number_of_frames, number_of_tracked_points, 3)

    # Calculate reprojection errors
    reprojection_error_flat = anipose_calibration_object.reprojection_error(data3d_flat, data2d_flat, mean=True)
    reprojection_error_full = anipose_calibration_object.reprojection_error(data3d_flat, data2d_flat, mean=False)
    
    # Reshape reprojection errors
    reprojection_error_by_camera = np.linalg.norm(reprojection_error_full, axis=2).reshape(
        number_of_cameras, number_of_frames, number_of_tracked_points
    )
    reprojection_error = reprojection_error_flat.reshape(number_of_frames, number_of_tracked_points)

    return data3d, reprojection_error, reprojection_error_by_camera