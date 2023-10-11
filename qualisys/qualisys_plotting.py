import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
import numpy as np
import pandas as pd

def plot_3d_scatter_from_dataframe(df):
    def plot_frame(f):
        ax.clear()
        frame_data = df[df['frame'] == f]
        ax.scatter(frame_data['x'], frame_data['y'], frame_data['z'], c='blue', label='Qualisys')
        ax.set_xlim([mean_x-ax_range, mean_x+ax_range])
        ax.set_ylim([mean_y-ax_range, mean_y+ax_range])
        ax.set_zlim([mean_z-ax_range, mean_z+ax_range])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title(f"Frame {f}")
        fig.canvas.draw_idle()

    mean_x = np.mean(df['x'])
    mean_y = np.mean(df['y'])
    mean_z = np.mean(df['z'])

    ax_range = 1000


    fig = plt.figure(figsize=[10, 8])
    ax = fig.add_subplot(111, projection='3d')
    slider_ax = plt.axes([0.25, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    frame_slider = Slider(slider_ax, 'Frame', df['frame'].min(), df['frame'].max(), valinit=0, valstep=1)

    def update(val):
        frame = int(frame_slider.val)
        plot_frame(frame)

    frame_slider.on_changed(update)
    plot_frame(0)
    plt.show()



def plot_frame_data(ax, frame_data_dict, mean_vals, ax_range):
    for label, frame_data in frame_data_dict.items():
        ax.scatter(frame_data[:, 0], frame_data[:, 1], frame_data[:, 2], label=label)
    mean_x, mean_y, mean_z = mean_vals
    ax.set_xlim([mean_x - ax_range, mean_x + ax_range])
    ax.set_ylim([mean_y - ax_range, mean_y + ax_range])
    ax.set_zlim([mean_z - ax_range, mean_z + ax_range])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

def plot_3d_scatter(array_dict):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plt.subplots_adjust(bottom=0.25)
    
    # Compute the overall mean and axis range
    first_array = next(iter(array_dict.values()))
    mean_x = np.mean(first_array[:, :, 0])
    mean_y = np.mean(first_array[:, :, 1])
    mean_z = np.mean(first_array[:, :, 2])
    mean_vals = (mean_x, mean_y, mean_z)
    
    ax_range = 1000

    # Initial frame data
    frame_data_dict = {label: array_3d[0] for label, array_3d in array_dict.items()}
    plot_frame_data(ax, frame_data_dict, mean_vals, ax_range)

    axcolor = 'lightgoldenrodyellow'
    axframe = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)

    sframe = Slider(axframe, 'Frame', 0, len(next(iter(array_dict.values()))) - 1, valinit=0, valstep=1)

    def update(val):
        frame_idx = int(sframe.val)
        ax.clear()
        frame_data_dict = {label: array_3d[frame_idx] for label, array_3d in array_dict.items()}
        plot_frame_data(ax, frame_data_dict, mean_vals, ax_range)
        ax.set_title(f"Frame {frame_idx}")
        fig.canvas.draw_idle()

    sframe.on_changed(update)

    plt.show()