from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QWidget,QFileDialog,QPushButton,QVBoxLayout

import matplotlib
matplotlib.use('Qt5Agg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from pathlib import Path
import numpy as np

from freemocap_utils.mediapipe_skeleton_builder import mediapipe_indices,mediapipe_connections,build_skeleton


class SkeletonViewWidget(QWidget):

    session_folder_loaded_signal = pyqtSignal()

    def __init__(self, plot_title:str):
        super().__init__()

        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        self.plot_title = plot_title
        self.fig,self.ax = self.initialize_skeleton_plot()
        self._layout.addWidget(self.fig)

        self.skeleton_loaded = False

        self.current_xlim = None
        self.current_ylim = None
        self.current_zlim = None


    def load_skeleton(self,skeleton_3d_data:np.ndarray):

        self.skeleton_3d_data = skeleton_3d_data
        self.mediapipe_skeleton = build_skeleton(self.skeleton_3d_data,mediapipe_indices,mediapipe_connections)
        self.reset_skeleton_3d_plot()

        self.skeleton_loaded = True

            
    def load_skeletons(self, skeletons_3d_data_list: list[np.ndarray], colors: list[str]):

        self.skeletons_3d_data_list = skeletons_3d_data_list
        self.colors = colors
        self.mediapipe_skeletons = [build_skeleton(skeleton_3d_data, mediapipe_indices, mediapipe_connections)
                                     for skeleton_3d_data in skeletons_3d_data_list]
        self.reset_skeleton_3d_plot()
        self.skeleton_loaded = True

    

    def initialize_skeleton_plot(self):
        fig = Mpl3DPlotCanvas(self, width=8, height=8, dpi=100)
        ax = fig.figure.axes[0]

        ax.set_title(self.plot_title)
        return fig, ax

    def reset_skeleton_3d_plot(self):
        self.ax.cla()
        self.calculate_axes_means()
        self.skel_xs, self.skel_ys, self.skel_zs = zip(*[self.get_x_y_z_data(skeleton_data, 0) for skeleton_data in self.skeletons_3d_data_list])
        for i, (skel_x, skel_y, skel_z) in enumerate(zip(self.skel_xs, self.skel_ys, self.skel_zs)):
            self.plot_skel(0, skel_x, skel_y, skel_z, self.colors[i])



    def calculate_axes_means(self):
        all_skeleton_data = np.concatenate(self.skeletons_3d_data_list, axis=1)
        self.mx_skel = np.nanmean(all_skeleton_data[:, :, 0])
        self.my_skel = np.nanmean(all_skeleton_data[:, :, 1])
        self.mz_skel = np.nanmean(all_skeleton_data[:, :, 2])
        self.skel_3d_range = 900

    def plot_skel(self, frame_number, skel_x, skel_y, skel_z, color):
        for skeleton_3d_data, mediapipe_skeleton, color in zip(self.skeletons_3d_data_list, self.mediapipe_skeletons, self.colors):
            skel_x, skel_y, skel_z = skeleton_3d_data[frame_number, :, 0], skeleton_3d_data[frame_number, :, 1], skeleton_3d_data[frame_number, :, 2]
            self.ax.scatter(skel_x, skel_y, skel_z, color=color)
            self.plot_skeleton_bones(frame_number, mediapipe_skeleton, color)

            trail_length = 30
            joints_to_add_trails = [23, 24, 25, 26, 27, 28]  # Left hip, knee, ankle, and right hip, knee, ankle

            for joint in joints_to_add_trails:
                start_frame = max(0, frame_number - trail_length)
                for i in range(start_frame, frame_number):
                    trail_x = skeleton_3d_data[i:i + 2, joint, 0]
                    trail_y = skeleton_3d_data[i:i + 2, joint, 1]
                    trail_z = skeleton_3d_data[i:i + 2, joint, 2]
                    self.ax.plot(trail_x, trail_y, trail_z, color=color, alpha=(i - start_frame + 1) / (trail_length + 1))
                    

        if self.current_xlim:
            self.ax.set_xlim([self.current_xlim[0],self.current_xlim[1]])
            self.ax.set_ylim([self.current_ylim[0],self.current_ylim[1]])
            self.ax.set_zlim([self.current_zlim[0],self.current_zlim[1]])
        else:
            self.ax.set_xlim([self.mx_skel-self.skel_3d_range, self.mx_skel+self.skel_3d_range])
            self.ax.set_ylim([self.my_skel-self.skel_3d_range, self.my_skel+self.skel_3d_range])
            self.ax.set_zlim([self.mz_skel-self.skel_3d_range, self.mz_skel+self.skel_3d_range])
        
        self.ax.set_title(self.plot_title)
        self.fig.figure.canvas.draw_idle()

    def plot_skeleton_bones(self, frame_number, mediapipe_skeleton, color):
        this_frame_skeleton_data = mediapipe_skeleton[frame_number]
        for connection in this_frame_skeleton_data.keys():
            line_start_point = this_frame_skeleton_data[connection][0]
            line_end_point = this_frame_skeleton_data[connection][1]

            bone_x, bone_y, bone_z = [line_start_point[0], line_end_point[0]], [line_start_point[1], line_end_point[1]], [
                line_start_point[2], line_end_point[2]]

            self.ax.plot(bone_x, bone_y, bone_z, color=color)

    def get_x_y_z_data(self, skeleton_data: np.ndarray, frame_number: int):
        skel_x = skeleton_data[frame_number, :, 0]
        skel_y = skeleton_data[frame_number, :, 1]
        skel_z = skeleton_data[frame_number, :, 2]

        return skel_x, skel_y, skel_z

    def replot(self, frame_number: int):
        self.current_xlim = self.ax.get_xlim()
        self.current_ylim = self.ax.get_ylim()
        self.current_zlim = self.ax.get_zlim()
        self.ax.cla()
        for i, skeleton_data in enumerate(self.skeletons_3d_data_list):
            skel_x, skel_y, skel_z = self.get_x_y_z_data(skeleton_data, frame_number)
            self.plot_skel(frame_number, skel_x, skel_y, skel_z, self.colors[i])


class Mpl3DPlotCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=4, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111,projection = '3d')
        super(Mpl3DPlotCanvas, self).__init__(fig)




