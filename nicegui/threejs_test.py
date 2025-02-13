from pathlib import Path
import numpy as np
from nicegui import ui

# Load data
path_to_data = Path(r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_13_48_44_MDN_treadmill_2\output_data\aligned_data\mediapipe_body_3d_xyz.npy")
body_data = np.load(path_to_data)

num_frames, num_markers, _ = body_data.shape  # Get dataset dimensions

# Function to retrieve frame data
def get_frame_data(frame: int = 0):
    return body_data[frame] / 100  # Scale down positions

spheres = {}  # Dictionary to store sphere objects

# Create the NiceGUI scene
with ui.scene().classes('w-[800px] h-[800px]') as scene:
    data = get_frame_data(0)  # Initialize first frame
    for i, marker in enumerate(data):
        spheres[i] = scene.sphere(radius=0.2).material('#4488ff').move(*marker)

# Function to update sphere positions
def update_spheres(frame: int):
    new_data = get_frame_data(frame)
    for i, marker in enumerate(new_data):
        spheres[i].move(*marker)  # Move each sphere to new position

# Slider to change frames
slider = ui.slider(min=0, max=num_frames-1, step=1, value=0) \
    .on_value_change(lambda e: update_spheres(int(e.value)))

# Display the current frame number
ui.label().bind_text_from(slider, 'value', lambda v: f'Frame: {int(v)}')

ui.run()
