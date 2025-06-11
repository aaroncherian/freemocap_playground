import numpy as np

class CharucoGroundplaneError(RuntimeError):
    """Raised when no frame satisfies the ‘all-corners-visible & stationary’ criteria."""
def get_charuco_x_and_y_idx(number_of_squares_width:int,
                            number_of_squares_height:int):
    """
    For a given board definition, get the corner marker indexes needed to make the x and y vectors
    """
    
    num_cols = number_of_squares_width - 1  # corner columns
    num_rows = number_of_squares_height - 1  # corner rows

    idx_x = num_cols * (num_rows - 1)
    idx_y = num_cols - 1

    return idx_x, idx_y


def get_unit_vector(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)

def compute_basis_vectors_of_new_reference(charuco_frame: np.ndarray,
                                           number_of_squares_width: int,
                                           number_of_squares_height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    origin = charuco_frame[0]

    idx_x, idx_y = get_charuco_x_and_y_idx(
        charuco_frame=charuco_frame,
        number_of_squares_width = number_of_squares_width,
        number_of_squares_height= number_of_squares_height
    )

    x_vec = charuco_frame[idx_x] - origin
    y_vec = charuco_frame[idx_y] - origin

    x_hat = get_unit_vector(x_vec)
    y_hat_raw = get_unit_vector(y_vec)
    z_hat = get_unit_vector(np.cross(x_hat, y_hat_raw))
    y_hat = get_unit_vector(np.cross(z_hat, x_hat))

    return x_hat, y_hat, z_hat


def get_charuco_frame(charuco_3d_data: np.ndarray):
    return charuco_3d_data[10, :, :]

def get_frames_under_velocity_threshold(points_velocity: np.ndarray,
                                       threshold: float = 1):
    frames_list = []
    for frame_number, frame  in enumerate(range(points_velocity.shape[0])):
        if np.isnan(points_velocity[frame]).any():
            continue
        if np.nanmax(points_velocity[frame]) < threshold:
            frames_list.append(frame_number+1) # +1 to account for the diff operation
    return frames_list    

def find_lowest_velocity_frame(points_velocity:np.ndarray, 
                         init_threshold:float = 1,
                         iteration_increment: float = .1) -> int:

    frames_list = get_frames_under_velocity_threshold(points_velocity=points_velocity,
                                                        threshold=init_threshold)
    if len(frames_list) == 1:
        return frames_list[0]
    
    elif len(frames_list) > 1:
        threshold = init_threshold - iteration_increment
        return find_lowest_velocity_frame(points_velocity=points_velocity,
                            init_threshold=threshold,
                            iteration_increment=iteration_increment)
    
    elif len(frames_list) == 0:
        iteration_increment = iteration_increment / 2
        threshold = init_threshold + iteration_increment*2
        return find_lowest_velocity_frame(points_velocity=points_velocity,
                            init_threshold=threshold,
                            iteration_increment=iteration_increment)



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # required for 3D plotting
import numpy as np

def set_axes_equal(ax):
    """Make 3D axes have equal scale — required for accurate 3D geometry."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]
    max_range = max(x_range, y_range, z_range)

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
    ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
    ax.set_zlim3d([z_middle - max_range / 2, z_middle + max_range / 2])

def plot_charuco_frame_3d(charuco_data: np.ndarray, frame_idx: int):
    """
    Plot the 3D positions of ChArUco corners at a given frame index, with equal axes.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    frame = charuco_data[frame_idx]  # shape (N, 3)
    valid_mask = ~np.isnan(frame[:, 0])  # ignore NaN points

    xs, ys, zs = frame[valid_mask].T

    ax.scatter(xs, ys, zs, c='blue', s=40)
    ax.set_title(f"ChArUco Corners - Frame {frame_idx}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=135)
    ax.grid(True)

    set_axes_equal(ax)  # <-- this ensures spatial accuracy
    plt.tight_layout()
    plt.show()


def find_still_frame(points_velocity: np.ndarray):
    max_velocity_per_frame = np.nanmax(points_velocity, axis=1)
    if np.all(np.isnan(max_velocity_per_frame)):
        raise CharucoGroundplaneError("No frame found where all 3 required ChArUco corners are visible")
    return int(np.nanargmin(max_velocity_per_frame))

def find_good_frame(charuco_data:np.ndarray,
                    number_of_squares_width:int,
                    number_of_squares_height:int,
                    frame_to_use: int = 0,
                    search_range: int = 120):
    
    if frame_to_use == 0:
        slice_to_search = slice(0, search_range)
    elif frame_to_use == -1:
        slice_to_search = slice(-search_range, None)
    elif frame_to_use > 0:
        start_frame = max(0, frame_to_use - search_range)
        end_frame = min(charuco_data.shape[0], frame_to_use + search_range)
        slice_to_search = slice(start_frame, end_frame)
    else:
        raise ValueError(f"Invalid value for frame_to_use: {frame_to_use}")

    idx_x, idx_y = get_charuco_x_and_y_idx(
        number_of_squares_width = number_of_squares_width,
        number_of_squares_height= number_of_squares_height
    )

    charuco_corners = charuco_data[slice_to_search,[0, idx_y, idx_x]]
    charuco_corners_velocity = np.linalg.norm(np.diff(charuco_corners, axis = 0), axis = 2)
    try:
        best_velocity_frame = find_still_frame(points_velocity=charuco_corners_velocity)
    except ValueError:
        print("No valid frames found within the specified range.")
        return None

    frame_offset = slice_to_search.start or 0
    best_position_frame = best_velocity_frame + frame_offset + 1
    return best_position_frame
    

    f =2

if __name__ ==  "__main__":
    from pathlib import Path
    charuco_data = np.load(r"D:\2025-04-28-calibration\output_data\charuco_3d_xyz.npy")
    num_squares_width = 5
    num_squares_height = 3

    find_good_frame(charuco_data=charuco_data,
                    number_of_squares_width=num_squares_width,
                    number_of_squares_height=num_squares_height,
                    frame_to_use=1440)


