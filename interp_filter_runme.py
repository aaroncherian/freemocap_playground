
import socket
from pathlib import Path
import sys

import numpy as np

from fmc_validation_toolbox import skeleton_filtering, skeleton_interpolation
from fmc_validation_toolbox.mediapipe_skeleton_builder import mediapipe_indices


import matplotlib
matplotlib.use('qt5agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QTabWidget, QVBoxLayout
import matplotlib.pyplot as plt

this_computer_name = socket.gethostname()

if this_computer_name == 'DESKTOP-F5LCT4Q':
    #freemocap_validation_data_path = Path(r"C:\Users\aaron\Documents\HumonLab\Spring2022\ValidationStudy\FreeMocap_Data")
    #freemocap_data_folder_path = Path(r'D:\freemocap2022\FreeMocap_Data')
    freemocap_data_folder_path = Path(r'D:\ValidationStudy2022\FreeMocap_Data')
else:
    freemocap_data_folder_path = Path(r'C:\Users\Aaron\Documents\sessions\FreeMocap_Data')

#sessionID = 'sesh_2022-05-12_15_13_02'  
#sessionID = 'sesh_2022-06-28_12_55_34'

sessionID = 'sesh_2022-05-24_16_10_46_JSM_T1_WalkRun'
data_array_folder = 'DataArrays'
array_name = 'mediaPipeSkel_3d.npy'

sampling_rate = 30
cutoff = 7
order = 4

data_array_folder_path = freemocap_data_folder_path / sessionID / data_array_folder
skel3d_raw_data = np.load(data_array_folder_path / array_name)

#skel3d_raw_data = skel3d_raw_data[0:6600, :, :]

skel_3d_interpolated = skeleton_interpolation.interpolate_skeleton(skel3d_raw_data)
skel_3d_filtered = skeleton_filtering.filter_skeleton(skel_3d_interpolated,cutoff,sampling_rate,order)

np.save(data_array_folder_path/'mediaPipeSkel_3d_filtered.npy', skel_3d_filtered)

class plotWindow():
    def __init__(self, parent=None):
        self.app = QApplication(sys.argv)
        self.MainWindow = QMainWindow()
        self.MainWindow.__init__()
        self.MainWindow.setWindowTitle("plot window")
        self.canvases = []
        self.figure_handles = []
        self.toolbar_handles = []
        self.tab_handles = []
        self.current_window = -1
        self.tabs = QTabWidget()
        self.MainWindow.setCentralWidget(self.tabs)
        self.MainWindow.resize(1280, 900)
        self.MainWindow.show()

    def addPlot(self, title, figure):
        new_tab = QWidget()
        layout = QVBoxLayout()
        new_tab.setLayout(layout)

        figure.subplots_adjust(left=0.05, right=0.99, bottom=0.05, top=0.91, wspace=0.2, hspace=0.2)
        new_canvas = FigureCanvas(figure)
        new_toolbar = NavigationToolbar(new_canvas, new_tab)

        layout.addWidget(new_canvas)
        layout.addWidget(new_toolbar)
        self.tabs.addTab(new_tab, title)

        self.toolbar_handles.append(new_toolbar)
        self.canvases.append(new_canvas)
        self.figure_handles.append(figure)
        self.tab_handles.append(new_tab)

    def show(self):
        self.app.exec_()

pw = plotWindow()

for marker in range(33):
    this_marker_filtered_data = skel_3d_filtered[:,marker,:]
    this_marker_interpolated_data = skel_3d_interpolated[:,marker,:]
    this_marker_raw_data = skel3d_raw_data[:,marker,:]

    joint_name = mediapipe_indices[marker]      

    f = plt.figure()
    f.suptitle('{} Position'.format(joint_name))
    ax1 = f.add_subplot(311)
    ax2 = f.add_subplot(312)
    ax3 = f.add_subplot(313)

    ax1.set_ylabel('X')
    ax2.set_ylabel('Y')
    ax3.set_ylabel('Z')

    ax1.plot(this_marker_raw_data[:,0])
    ax1.plot(this_marker_filtered_data[:,0], c = 'r')

    ax2.plot(this_marker_raw_data[:,1])
    ax2.plot(this_marker_filtered_data[:,1], c = 'r')

    ax3.plot(this_marker_raw_data[:,2])
    ax3.plot(this_marker_filtered_data[:,2], c = 'r'  )
    pw.addPlot(joint_name, f)

    #print(joint_name, ':{}'.format(this_marker_filtered_data.shape))
    print(joint_name, ': raw| ', np.count_nonzero(np.isnan(this_marker_raw_data)), ' interpolated| ', np.count_nonzero(np.isnan(this_marker_interpolated_data)), ' filtered| ', np.count_nonzero(np.isnan(this_marker_filtered_data)))
pw.show()

f = 2