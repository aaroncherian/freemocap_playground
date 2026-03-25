
from pathlib import Path
import numpy as np
import logging
import multiprocessing
from typing import Union, Tuple, Optional, Any

from reconstruct_trackers.anipose_utils.anipose_object_loader import load_anipose_calibration_toml_from_path
from reconstruct_trackers.anipose_utils.freemocap_anipose import CameraGroup
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
        image_2d_data=data_2d,
        use_triangulate_ransac=False,
        kill_event=None,
    )

    return data_3d


def triangulate_3d_data(
        anipose_calibration_object: CameraGroup,
        image_2d_data: np.ndarray,
        use_triangulate_ransac: bool = False,
        kill_event: multiprocessing.Event = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    number_of_cameras = image_2d_data.shape[0]
    number_of_frames = image_2d_data.shape[1]
    number_of_tracked_points = image_2d_data.shape[2]
    number_of_spatial_dimensions = image_2d_data.shape[3]

    if not number_of_spatial_dimensions == 2:
        logger.error(
            f"This is supposed to be 2D data but, number_of_spatial_dimensions: {number_of_spatial_dimensions}"
        )
        raise ValueError

    # reshape data to collapse across 'frames' so it becomes [number_of_cameras,
    # number_of_2d_points(numFrames*numPoints), XY]
    data2d_flat = image_2d_data.reshape(number_of_cameras, -1, 2)

    logger.info(
        f"Reconstructing 3d points from 2d points with shape: \n"
        f"number_of_cameras: {number_of_cameras},\n"
        f"number_of_frames: {number_of_frames}, \n"
        f"number_of_tracked_points: {number_of_tracked_points},\n"
        f"number_of_spatial_dimensions: {number_of_spatial_dimensions}"
    )

    if use_triangulate_ransac:
        logger.info("Using `triangulate_ransac` method")
        data3d_flat = anipose_calibration_object.triangulate_ransac(data2d_flat, progress=True, kill_event=kill_event)
    else:
        logger.info("Using simple `triangulate` method ")
        # data3d_flat = anipose_calibration_object.triangulate(data2d_flat, progress=True, kill_event=kill_event)
        data3d_flat = anipose_calibration_object.triangulate(data2d_flat, progress=True, kill_event=kill_event, number_of_tracked_points=number_of_tracked_points)

    spatial_data3d_numFrames_numTrackedPoints_XYZ = data3d_flat.reshape(number_of_frames, number_of_tracked_points, 3)

    data3d_reprojectionError_flat = anipose_calibration_object.reprojection_error(data3d_flat, data2d_flat, mean=True)
    data3d_reprojectionError_full = anipose_calibration_object.reprojection_error(data3d_flat, data2d_flat, mean=False)
    reprojectionError_cam_frame_marker = np.linalg.norm(data3d_reprojectionError_full, axis=2).reshape(
        number_of_cameras, number_of_frames, number_of_tracked_points
    )

    reprojection_error_data3d_numFrames_numTrackedPoints = data3d_reprojectionError_flat.reshape(
        number_of_frames, number_of_tracked_points
    )

    return (
        spatial_data3d_numFrames_numTrackedPoints_XYZ,
        reprojection_error_data3d_numFrames_numTrackedPoints,
        reprojectionError_cam_frame_marker,
    )