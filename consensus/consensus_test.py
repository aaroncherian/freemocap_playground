import plotly.graph_objs as go
import numpy as np
import itertools

## if continuing on this, the todo would be loading jon's treadmill data, finding the exact right frame, getting qualisys data, and doing the alignment 

class FreeMoCap3DPlotter:
    def __init__(self, data, frame):
        """
        Initializes the 3D plotter with FreeMoCap-shaped data and the frame to plot.
        
        Parameters:
            data (numpy array): 3D data of shape (frames, num_points, 3).
            frame (int): The frame index to plot.
        """
        self.data = data
        self.frame = frame
        
        # Extract the 3D points for the specified frame
        self.frame_data = self.data[self.frame]
        
        # Calculate the axis ranges to ensure equal length
        x_range = [np.min(self.frame_data[:, 0]), np.max(self.frame_data[:, 0])]
        y_range = [np.min(self.frame_data[:, 1]), np.max(self.frame_data[:, 1])]
        z_range = [np.min(self.frame_data[:, 2]), np.max(self.frame_data[:, 2])]
        
        max_range = max(np.ptp(x_range), np.ptp(y_range), np.ptp(z_range))
        mid_x = np.mean(x_range)
        mid_y = np.mean(y_range)
        mid_z = np.mean(z_range)

        # Create initial 3D scatter plot
        self.fig = go.Figure(data=[go.Scatter3d(
            x=self.frame_data[:, 0],
            y=self.frame_data[:, 1],
            z=self.frame_data[:, 2],
            opacity=.7,
            mode='markers',
            marker=dict(
                size=5,
                color='blue',  # Color of the markers
            )
        )])

        # Update layout for equal axis length
        self.fig.update_layout(
            title=f"3D Plot for Frame {self.frame}",
            scene=dict(
                xaxis=dict(range=[mid_x - max_range / 2, mid_x + max_range / 2]),
                yaxis=dict(range=[mid_y - max_range / 2, mid_y + max_range / 2]),
                zaxis=dict(range=[mid_z - max_range / 2, mid_z + max_range / 2]),
                aspectmode='cube'  # Ensures the aspect ratio is 1:1:1
            )
        )

    def add_new_3d_data(self, data, frame):
        """
        Adds new 3D data to the plot.
        
        Parameters:
            data (numpy array): 3D data of shape (frames, num_points, 3).
            frame (int): The frame index to plot.
        """
        self.data = data
        self.frame = frame
        
        # Extract the 3D points for the specified frame
        self.frame_data = self.data[self.frame]
        
        # Add new data to the plot
        self.fig.add_trace(go.Scatter3d(
            x=self.frame_data[:, 0],
            y=self.frame_data[:, 1],
            z=self.frame_data[:, 2],
            mode='markers',
            opacity=.7,
            marker=dict(
                size=5,
                color='green',
            )
        ))

    def add_points(self, points, color='red', size=5):
        """
        Adds additional points to the existing plot.
        
        Parameters:
            points (numpy array): Additional 3D points to plot of shape (num_points, 3).
            color (str): Color of the additional points.
            size (int): Size of the additional points.
        """
        # Add new points to the plot
        self.fig.add_trace(go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            opacity=.4,
            marker=dict(
                size=size,
                color=color,
            )
        ))

    def show(self):
        """Displays the plot."""
        self.fig.show()

def generate_camera_combinations(num_cameras):
    """
    Generate all possible combinations of camera indices.
    
    Args:
        num_cameras (int): Total number of cameras.
        
    Returns:
        List of lists, where each sublist contains the indices of cameras used in a combination.
    """
    combinations = []
    camera_indices = list(range(num_cameras))
    
    for r in range(2, num_cameras + 1):  # Start from 2 cameras to avoid single camera case
        combinations.extend(itertools.combinations(camera_indices, r))
    
    return combinations

def apply_mask_to_data(points_2d, camera_combination):
    """
    Mask out cameras that are not in the given combination.
    
    Args:
        points_2d (np.array): 2D points from each camera, shape (num_cams, num_frames, num_points, 2).
        camera_combination (list of int): Indices of cameras to be used.
        
    Returns:
        np.array: Masked 2D points with the same shape as input.
    """
    num_cams, num_frames, num_points, _ = points_2d.shape
    points_2d_masked = np.full((num_cams, num_frames, num_points, 2), np.nan)  # Initialize with NaNs
    
    for i in camera_combination:
        points_2d_masked[i] = points_2d[i]
    
    return points_2d_masked

if __name__ == '__main__':
    # Load the FreeMoCap data
    from pathlib import Path
    from freemocap_anipose import CameraGroup
    from skellymodels.model_info.mediapipe_model_info import MediapipeModelInfo

    frame_to_use = 50
    number_of_cameras = 4
    marker_to_use = 'nose'
    # path_to_folder = Path(r'D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_13_37_32_MDN_treadmill_1')
    path_to_folder = Path(r'D:\steen_pantsOn_gait')
    
    
    # path_to_calibration_toml = path_to_folder/ 'sesh_2023-05-17_12_49_06_calibration_3_camera_calibration.toml'
    path_to_calibration_toml = path_to_folder/ 'steen_calibration_camera_calibration.toml'
    path_to_2d_data = path_to_folder/ 'output_data' / 'raw_data'/ 'mediapipe2dData_numCams_numFrames_numTrackedPoints_pixelXY.npy'
    path_to_3d_data = path_to_folder/ 'output_data' / 'raw_data' / 'mediapipe3dData_numFrames_numTrackedPoints_spatialXYZ.npy'
    
    data_2d = np.load(path_to_2d_data)
    data_2d = data_2d[:, frame_to_use, MediapipeModelInfo.landmark_names.index(marker_to_use), 0:2]
    data_2d = data_2d[:, np.newaxis, np.newaxis, :]


    # data2d_flat = data_2d.reshape(number_of_cameras, -1, 2)

    data_3d = np.load(path_to_3d_data)[:,0:len(MediapipeModelInfo.body_landmark_names),:]

    calibration_object = CameraGroup.load(path_to_calibration_toml)

    # point_3d = calibration_object.triangulate(data2d_flat, progress=True)
    plotter = FreeMoCap3DPlotter(data_3d, frame_to_use)
    # plotter.add_points(point_3d, color='red')
    # plotter.show()

    combinations = generate_camera_combinations(number_of_cameras)
    point_cloud = []
    
    # for camera_combo in combinations:
    #     masked_data = apply_mask_to_data(data_2d, camera_combo)
    #     masked_data_flat = masked_data.reshape(number_of_cameras, -1, 2)
    #     point_3d = calibration_object.triangulate(masked_data_flat, progress=True)
    #     plotter.add_points(point_3d, color='red')
    #     point_cloud.append(point_3d)

    new_data = np.load(r'D:\steen_pantsOn_gait_3_cameras\output_data\raw_data\mediapipe3dData_numFrames_numTrackedPoints_spatialXYZ.npy')[:,0:len(MediapipeModelInfo.body_landmark_names),:]
    plotter.add_new_3d_data(new_data, frame_to_use)
    plotter.show()
    
    f = 2

