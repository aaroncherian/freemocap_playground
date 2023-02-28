
from PyQt6.QtWidgets import QWidget,QFileDialog,QPushButton,QVBoxLayout

import matplotlib
matplotlib.use('Qt5Agg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from freemocap_utils.mediapipe_skeleton_builder import mediapipe_indices
from freemocap_utils.GUI_widgets.time_syncing_widgets.qualisys_indices import qualisys_indices
from freemocap_utils.GUI_widgets.rmse_widgets.RMSE_calculator import calculate_rmse_dataframe

import numpy as np

import seaborn as sns

class RMSEPlots(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=15, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.x_ax = fig.add_subplot(311)
        self.y_ax = fig.add_subplot(312)
        self.z_ax = fig.add_subplot(313)

        super(RMSEPlots, self).__init__(fig)

class RMSEViewWidget(QWidget):

    def __init__(self):
        super().__init__()

        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        self.fig, self.ax_list = self.initialize_bar_plot()

        self._layout.addWidget(self.fig)

    def initialize_bar_plot(self):
        sns.set_theme(style = 'whitegrid')

        fig = RMSEPlots(self, width=15, height=15, dpi=100)
        self.x_ax = fig.figure.axes[0]
        self.y_ax = fig.figure.axes[1]
        self.z_ax = fig.figure.axes[2]

        self.ax_list = [self.x_ax,self.y_ax,self.z_ax]
        
        return fig, self.ax_list

    def update_plot(self,qualisys_data:np.ndarray,freemocap_data:np.ndarray):

        dimension_list = ['x', 'y', 'z']
        rmse_dataframe = calculate_rmse_dataframe(qualisys_data=qualisys_data, freemocap_data=freemocap_data)

        for (dimension,ax) in zip(dimension_list, self.ax_list):
            ax.cla()

            sns.barplot(
                data= rmse_dataframe.loc[rmse_dataframe['dimension'] == dimension],
                x= "marker", y= f"rmse",
                errorbar="sd", palette="dark", alpha = .6,
                ax = ax
            )
            ax.set_ylabel(f'rmse {dimension}')

            ax.set_ylim([0,70])

            if not dimension == 'z':
                ax.set(xlabel= None)
                ax.set(xticks = [])
            else:
                ax.tick_params(axis = 'x', rotation = 90)

            # ax.bar_label(ax.containers[0],)


        self.fig.figure.canvas.draw_idle()

