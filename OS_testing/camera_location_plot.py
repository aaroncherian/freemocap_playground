from pathlib import Path
import toml
import numpy as np
import plotly.graph_objects as go
import cv2  # OpenCV for Rodrigues transformation

def get_calibration_data(path_to_folder_of_tomls: Path):
    """
    Loads calibration data from all TOML files in the specified folder.
    """
    return {
        calibration_data['metadata']['system']: calibration_data
        for file in path_to_folder_of_tomls.glob('*.toml')
        if (calibration_data := toml.load(file))
    }

def extract_camera_data(path_to_folder_of_tomls, num_cams=3):
    """
    Extracts translation (position) and rotation (orientation) data for cameras from TOML calibration files.
    """
    calibration_dict = get_calibration_data(path_to_folder_of_tomls)

    camera_data = {}
    for system_name, calibration_data in calibration_dict.items():
        camera_data[system_name] = {
            f'cam_{i}': {
                "translation": np.array(calibration_data[f'cam_{i}']['translation']),
                "rotation": np.array(calibration_data[f'cam_{i}']['rotation'])
            }
            for i in range(num_cams)
        }
    
    return camera_data

def plot_cameras_3d(camera_data):
    """
    Creates a 3D plot of camera positions and orientations across different systems.
    """
    fig = go.Figure()

    # Define colors for different systems
    system_colors = {"macos": "blue", "windows": "red", "ubuntu": "green"}

    # Define arrow length for visualization
    arrow_length = 500

    # Plot each system's camera positions and orientations
    for system, cams in camera_data.items():
        color = system_colors.get(system, "black")

        for cam_name, cam_info in cams.items():
            # Extract position and rotation
            pos = cam_info["translation"]
            rot_vector = cam_info["rotation"]

            # Convert Rodrigues vector to rotation matrix
            R, _ = cv2.Rodrigues(rot_vector)

            # Compute forward, right, and up direction vectors (from camera's local coordinate frame)
            forward_dir = R @ np.array([0, 0, 1])  # Local Z-axis (forward)
            right_dir = R @ np.array([1, 0, 0])    # Local X-axis (right)
            up_dir = R @ np.array([0, 1, 0])       # Local Y-axis (up)

            # Plot camera position as a point
            fig.add_trace(go.Scatter3d(
                x=[pos[0]], y=[pos[1]], z=[pos[2]],
                mode="markers",
                marker=dict(size=8, color=color, symbol="circle"),
                name=f"{system} - {cam_name}"
            ))

            # Plot orientation arrows (forward, right, up)
            fig.add_trace(go.Cone(
                x=[pos[0]], y=[pos[1]], z=[pos[2]],
                u=[forward_dir[0] * arrow_length],
                v=[forward_dir[1] * arrow_length],
                w=[forward_dir[2] * arrow_length],
                colorscale=[[0, color], [1, color]],
                showscale=False,
                name=f"{system} - {cam_name} Forward"
            ))

            fig.add_trace(go.Cone(
                x=[pos[0]], y=[pos[1]], z=[pos[2]],
                u=[right_dir[0] * (arrow_length * 0.5)],
                v=[right_dir[1] * (arrow_length * 0.5)],
                w=[right_dir[2] * (arrow_length * 0.5)],
                colorscale=[[0, "gray"], [1, "gray"]],
                showscale=False,
                name=f"{system} - {cam_name} Right"
            ))

            fig.add_trace(go.Cone(
                x=[pos[0]], y=[pos[1]], z=[pos[2]],
                u=[up_dir[0] * (arrow_length * 0.5)],
                v=[up_dir[1] * (arrow_length * 0.5)],
                w=[up_dir[2] * (arrow_length * 0.5)],
                colorscale=[[0, "black"], [1, "black"]],
                showscale=False,
                name=f"{system} - {cam_name} Up"
            ))

    # Configure plot layout
    fig.update_layout(
        title="3D Camera Position and Orientation Across Systems (Recording 1)",
        scene=dict(
            xaxis_title="X Axis",
            yaxis_title="Y Axis",
            zaxis_title="Z Axis",
            aspectmode='cube'
        )
    )

    # Display plot
    fig.show()

# Example usage (update with actual path)
first_recording_path = Path(r"D:/system_testing/calibrations/calibrations_one")  # Adjust based on actual structure
camera_data = extract_camera_data(first_recording_path, num_cams=3)

# Generate 3D visualization
plot_cameras_3d(camera_data)
