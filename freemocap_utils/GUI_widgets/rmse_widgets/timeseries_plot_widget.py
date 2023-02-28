
from PyQt6.QtWidgets import QWidget,QFileDialog,QPushButton,QVBoxLayout

import matplotlib
matplotlib.use('Qt5Agg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from freemocap_utils.mediapipe_skeleton_builder import mediapipe_indices
from freemocap_utils.GUI_widgets.time_syncing_widgets.qualisys_indices import qualisys_indices

import numpy as np

class TimeSeriesPlots(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=15, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.x_ax = fig.add_subplot(311)
        self.y_ax = fig.add_subplot(312)
        self.z_ax = fig.add_subplot(313)

        super(TimeSeriesPlots, self).__init__(fig)

class TimeSeriesViewWidget(QWidget):

    def __init__(self):
        super().__init__()

        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        self.fig, self.ax_list = self.initialize_skeleton_plot()

        toolbar = NavigationToolbar(self.fig, self)

        self._layout.addWidget(toolbar)
        self._layout.addWidget(self.fig)

    def initialize_skeleton_plot(self):
        fig = TimeSeriesPlots(self, width=15, height=10, dpi=100)
        self.x_ax = fig.figure.axes[0]
        self.y_ax = fig.figure.axes[1]
        self.z_ax = fig.figure.axes[2]

        # self.x_ax.set_ylabel('X Axis (mm)')
        # self.y_ax.set_ylabel('Y Axis (mm)')
        # self.z_ax.set_ylabel('Z Axis (mm)')

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

    

    def update_plot(self,marker_to_plot:str, freemocap_data:np.ndarray, qualisys_data:np.ndarray):
        mediapipe_index = self.get_mediapipe_indices(marker_to_plot)
        
        for dimension, ax in enumerate(self.ax_list):
            ax.cla()
            ax.plot(freemocap_data[:,mediapipe_index,dimension], label = 'FreeMoCap', alpha = .7)
            ax.legend()
            #ax.set_ylabel('X Axis (mm)')
            
        
        qualisys_index = self.get_qualisys_indices(marker_to_plot)

        if qualisys_index:
            for dimension, ax, in enumerate(self.ax_list):
                ax.plot(qualisys_data[:,qualisys_index,dimension], label = 'Qualisys', alpha = .7)
                ax.legend()
                
        
        self.fig.figure.canvas.draw_idle()



