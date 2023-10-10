import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
import numpy as np
import pandas as pd

def plot_3d_scatter(df):
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
