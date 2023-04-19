import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

from pathlib import Path

from freemocap_utils.mediapipe_skeleton_builder import mediapipe_indices,mediapipe_connections,build_skeleton


# Example skeleton data (random)
path_to_data_folder = Path(r'C:\Users\aaron\FreeMocap_Data\recording_sessions')

skeleton_data = np.load(path_to_data_folder/'recording_15_19_00_gmt-4__brit_baseline'/'output_data'/'average_step_3d.npy')
num_frames = skeleton_data.shape[0]

mediapipe_skeleton = build_skeleton(skeleton_data,mediapipe_indices, mediapipe_connections)
# Define the function to update the plot

def update_skeleton(frame, skeleton_data, scatter_plot):
    scatter_plot._offsets3d = (skeleton_data[int(frame), :, 0], skeleton_data[int(frame), :, 1], skeleton_data[int(frame), :, 2])
    fig.canvas.draw_idle()
    

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
plt.subplots_adjust(bottom=0.2)

# Calculate axes means
mx_skel = np.nanmean(skeleton_data[:, 0:33, 0])
my_skel = np.nanmean(skeleton_data[:, 0:33, 1])
mz_skel = np.nanmean(skeleton_data[:, 0:33, 2])
skel_3d_range = 900

# Set the axes limits
ax.set_xlim(mx_skel - skel_3d_range, mx_skel + skel_3d_range)
ax.set_ylim(my_skel - skel_3d_range, my_skel + skel_3d_range)
ax.set_zlim(mz_skel - skel_3d_range, mz_skel + skel_3d_range)

frame_count = skeleton_data.shape[0]

scatter_plot = ax.scatter(skeleton_data[0, :, 0], skeleton_data[0, :, 1], skeleton_data[0, :, 2])

# Set up the slider
slider_ax = plt.axes([0.15, 0.05, 0.7, 0.03])
slider = Slider(slider_ax, 'Frame', 0, frame_count - 1, valinit=0, valstep=1)

# Set up the update function for the slider
def slider_update(val):
    update_skeleton(val, skeleton_data, scatter_plot)

slider.on_changed(slider_update)

plt.show()