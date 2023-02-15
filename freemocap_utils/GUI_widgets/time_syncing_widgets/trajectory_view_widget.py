
from PyQt6.QtWidgets import QWidget,QFileDialog,QPushButton,QVBoxLayout

import matplotlib
matplotlib.use('Qt5Agg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from freemocap_utils.mediapipe_skeleton_builder import mediapipe_indices
from freemocap_utils.GUI_widgets.time_syncing_widgets.qualisys_indices import qualisys_indices

import numpy as np

class TrajectoryPlots(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=15, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.x_ax = fig.add_subplot(311)
        self.y_ax = fig.add_subplot(312)
        self.z_ax = fig.add_subplot(313)

        super(TrajectoryPlots, self).__init__(fig)

class TrajectoryViewWidget(QWidget):

    def __init__(self):
        super().__init__()

        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        self.fig, self.ax_list = self.initialize_skeleton_plot()
        self.previous_ax_lims = {0:[[],[]], 1:[[],[]], 2: [[],[]]} #keep track of where you previously zoomed in on the graph

        toolbar = NavigationToolbar(self.fig, self)

        self._layout.addWidget(toolbar)
        self._layout.addWidget(self.fig)

    def initialize_skeleton_plot(self):
        fig = TrajectoryPlots(self, width=15, height=10, dpi=100)
        self.x_ax = fig.figure.axes[0]
        self.y_ax = fig.figure.axes[1]
        self.z_ax = fig.figure.axes[2]

        self.ax_list = [self.x_ax,self.y_ax,self.z_ax]
        return fig, self.ax_list

    def get_mediapipe_indices(self,marker_to_plot):
        mediapipe_index = mediapipe_indices.index(marker_to_plot)
        return mediapipe_index

    def get_qualisys_indices(self,marker_to_plot):
        qualisys_index = None

        try:
            qualisys_index = qualisys_indices.index(marker_to_plot)
        except ValueError:
            pass

        return qualisys_index

    

    def update_plot(self,marker_to_plot:str, freemocap_data:np.ndarray, qualisys_data:np.ndarray, freemocap_start_end_frames:list, qualisys_start_end_frames:list, reset_ax_limits = False):
        mediapipe_index = self.get_mediapipe_indices(marker_to_plot)
        for dimension, ax in enumerate(self.ax_list):
            if self.previous_ax_lims[dimension][0]: #if the previous ax_lims list has been started, update it
                self.previous_ax_lims[dimension] = [[ax.get_xlim()],[ax.get_ylim()]]
                # self.previous_ax_lims[0] = ax.get_xlim()
                # self.previous_ax_lims[1] = ax.get_ylim()
                

            ax.cla()
            ax.plot(freemocap_data[freemocap_start_end_frames[0]:freemocap_start_end_frames[1],mediapipe_index,dimension] - freemocap_data[freemocap_start_end_frames[0],mediapipe_index,dimension], label = 'FreeMoCap', alpha = .7)
            
            if reset_ax_limits:
                self.previous_ax_lims = {0:[[],[]], 1:[[],[]], 2: [[],[]]}

            if self.previous_ax_lims[dimension][0]:
                ax.set_xlim(self.previous_ax_lims[dimension][0][0])
                ax.set_ylim(self.previous_ax_lims[dimension][1][0])
            elif not self.previous_ax_lims[dimension][0]:
                self.previous_ax_lims[dimension][0] = ax.get_xlim()
                self.previous_ax_lims[dimension][1] = ax.get_ylim()
            ax.legend()
        qualisys_index = self.get_qualisys_indices(marker_to_plot)

        if qualisys_index:
            for dimension, ax, in enumerate(self.ax_list):
                ax.plot(qualisys_data[qualisys_start_end_frames[0]:qualisys_start_end_frames[1],qualisys_index,dimension] - qualisys_data[qualisys_start_end_frames[0],qualisys_index,dimension], label = 'Qualisys', alpha = .7)
                ax.legend()
        self.fig.figure.canvas.draw_idle()



