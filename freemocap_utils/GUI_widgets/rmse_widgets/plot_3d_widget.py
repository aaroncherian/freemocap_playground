from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSlider
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import Slider

import numpy as np 

class Scatter3DWidget(QWidget):
    def __init__(self, freemocap_data, qualisys_data, parent=None):
        super().__init__(parent)

        self.freemocap_data = freemocap_data
        self.qualisys_data = qualisys_data
        
        self.layout = QVBoxLayout()
        self.setMinimumSize(500, 700)
        
        self.figure = Figure(figsize=[10, 8])
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        self.ax = self.figure.add_subplot(111, projection='3d')

        self.slider = QSlider()
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(qualisys_data) - 1)
        self.slider.valueChanged.connect(self.update_plot)
        self.layout.addWidget(self.slider)

        self.setLayout(self.layout)

        self.plot_frame(0)

    def update_plot(self):
        frame = self.slider.value()
        self.plot_frame(frame)

    def plot_frame(self, f):
        mean_x = (np.mean(self.qualisys_data[:, :, 0]) + np.mean(self.freemocap_data[:, :, 0])) / 2
        mean_y = (np.mean(self.qualisys_data[:, :, 1]) + np.mean(self.freemocap_data[:, :, 1])) / 2
        mean_z = (np.mean(self.qualisys_data[:, :, 2]) + np.mean(self.freemocap_data[:, :, 2])) / 2

        ax_range = 1000
        limit_x = mean_x + ax_range
        limit_y = mean_y + ax_range
        limit_z = mean_z + ax_range

        self.ax.clear()
        self.ax.scatter(self.qualisys_data[f, :, 0], self.qualisys_data[f, :, 1], self.qualisys_data[f, :, 2], c='blue', label='Qualisys')
        self.ax.scatter(self.freemocap_data[f, :, 0], self.freemocap_data[f, :, 1], self.freemocap_data[f, :, 2], c='red', label='FreeMoCap')
        self.ax.set_xlim([-limit_x, limit_x])
        self.ax.set_ylim([-limit_y, limit_y])
        self.ax.set_zlim([-limit_z, limit_z])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.legend()
        self.ax.set_title(f"Frame {f}")
        self.canvas.draw()

    def update_data(self, freemocap_data, qualisys_data):
        self.freemocap_data = freemocap_data
        self.qualisys_data = qualisys_data
        self.update_plot() # Refresh the plot with the new data
