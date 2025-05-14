from freemocap.core_processes.capture_volume_calibration.triangulate_3d_data import triangulate_3d_data
from freemocap.core_processes.capture_volume_calibration.anipose_camera_calibration.get_anipose_calibration_object import (
    load_anipose_calibration_toml_from_path,
)
from pathlib import Path
from scipy.spatial import distance_matrix
import numpy as np

# --- Board layout ---
charuco_rows = 2  # number of corner rows = squares_y - 1
charuco_cols = 4  # number of corner cols = squares_x - 1
expected_num_points = charuco_rows * charuco_cols  # 24

square_size_mm = 58
per_frame_means = []

# --- Load and clean 2D data ---
path_to_recording = Path(r'D:\2025-04-28-calibration')
path_to_2d_data = path_to_recording/'output_data'/'raw_data'/'charuco_2dData_numCams_numFrames_numTrackedPoints_pixelXY.npy'
# path_to_cal_toml = path_to_recording/f'{path_to_recording.stem}_camera_calibration.toml'
path_to_cal_toml = path_to_recording/f'2025-04-28-calibration_camera_calibration_aligned.toml'
path_to_save_3d_data = path_to_recording/'output_data'/'aligned_charuco_3d.npy'

data_2d = np.load(path_to_2d_data, allow_pickle=True)
shape = data_2d.shape
clean_array = np.full(shape, np.nan, dtype=np.float32)

#need to turn data from dtype = object to float
for cam in range(shape[0]):
    for frame in range(shape[1]):
        for point in range(shape[2]):
            xy = data_2d[cam, frame, point]
            if xy is not None and xy[0] is not None and xy[1] is not None:
                clean_array[cam, frame, point] = [xy[0], xy[1]]

data_3d, *_ = triangulate_3d_data(
    anipose_calibration_object=load_anipose_calibration_toml_from_path(path_to_cal_toml),
    image_2d_data=clean_array
)

np.save(path_to_save_3d_data, data_3d)

# --- Define true neighbor pairs from layout ---
def get_charuco_neighbor_pairs(rows: int, cols: int):
    neighbors = []
    for row in range(rows):
        for col in range(cols):
            idx = row * cols + col
            if col < cols - 1:
                neighbors.append((idx, idx + 1))      # right neighbor
            if row < rows - 1:
                neighbors.append((idx, idx + cols))   # bottom neighbor
    return neighbors

neighbor_pairs = get_charuco_neighbor_pairs(charuco_rows, charuco_cols)

# --- Loop over frames ---
for frame_index in range(data_3d.shape[0]):
    points_3d = data_3d[frame_index]

    # Skip frames with missing points
    if np.isnan(points_3d).any():
        continue

    # Compute distances only between defined neighbors
    frame_dists = []
    for i, j in neighbor_pairs:
        pt1, pt2 = points_3d[i], points_3d[j]
        if not np.isnan(pt1).any() and not np.isnan(pt2).any():
            dist = np.linalg.norm(pt2 - pt1)
            frame_dists.append(dist)

    if frame_dists:
        per_frame_means.append(np.mean(frame_dists))

# --- Summarize results ---
per_frame_means = np.array(per_frame_means)

print(f"\nFrames with all {expected_num_points} points: {len(per_frame_means)}")

print("\nTrue-Neighbor Distance Stats:")
print(f"Mean:   {per_frame_means.mean():.2f} mm")
print(f"Median: {np.median(per_frame_means):.2f} mm")
print(f"Std:    {per_frame_means.std():.2f} mm")
print(f"Mean error vs {square_size_mm}mm: {per_frame_means.mean() - square_size_mm:.2f} mm")
