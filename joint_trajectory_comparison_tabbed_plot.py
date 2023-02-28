from pathlib import Path

from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QTabWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from freemocap_utils.mediapipe_skeleton_builder import mediapipe_indices
from freemocap_utils import freemocap_data_loader
import sys

import numpy as np
import matplotlib.pyplot as plt



qualisys_indices = [
'head',
'left_ear',
'right_ear',
'cspine',
'left_shoulder',
'right_shoulder',
'left_elbow',
'right_elbow',
'left_wrist',
'right_wrist',
'left_index',
'right_index',
'left_hip',
'right_hip',
'left_knee',
'right_knee',
'left_ankle',
'right_ankle',
'left_heel',
'right_heel',
'left_foot_index',
'right_foot_index',
]
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



path_to_data_folder = Path(r'D:\ValidationStudy_numCams\FreeMoCap_Data')

sessionID_list = ['sesh_2022-05-24_16_10_46_WalkRun_front','sesh_2022-05-24_16_10_46_WalkRun_front_side','sesh_2022-05-24_16_10_46_WalkRun_front_back','sesh_2022-05-24_16_10_46_JSM_T1_WalkRun']
labels = ['front', 'front_side','front_back','front_side_back']

# path_to_data_folder = Path(r'D:\ValidationStudy2022\FreeMoCap_Data')
# sessionID_list = ['sesh_2022-05-24_15_55_40_JSM_T1_BOS']
# labels = ['FreeMoCap']
# sessionID_list = ['sesh_2022-05-24_16_10_46_WalkRun_front','sesh_2022-05-24_16_10_46_WalkRun_front_side','sesh_2022-05-24_16_10_46_JSM_T1_WalkRun']
# labels = ['front', 'front_side','front_side_back']

#path_to_qualysis_session_folder = Path(r"D:\ValidationStudy_numCams\FreeMoCap_Data\qualisys_sesh_2022-05-24_16_02_53_JSM_T1_WalkRun")
#qualisys_data = np.load(path_to_qualysis_session_folder/'DataArrays'/'qualisys_origin_aligned_skeleton_3D.npy')
path_to_qualisys_session_folder = Path(r"D:\ValidationStudy2022\FreeMocap_Data\qualisys_sesh_2022-05-24_16_02_53_JSM_T1_WalkRun")
qualisys_data = np.load(path_to_qualisys_session_folder/'DataArrays'/'qualisys_marker_data_29Hz.npy')
samples = qualisys_data.shape[0]

qualisys_sliced = qualisys_data[:,:,:]



freemocap_sessions_dict = {}
for count,sessionID in enumerate(sessionID_list):
    freemocap_sessions_dict[count] = freemocap_data_loader.FreeMoCapDataLoader(path_to_data_folder/sessionID)

mediapipe_joint_data_dict = {}

for count,session_data in enumerate(freemocap_sessions_dict.values()):
    mediapipe_data = session_data.load_mediapipe_body_data() 
    mediapipe_joint_data = mediapipe_data[1162:6621,:,:]
    mediapipe_joint_data_dict[count] = mediapipe_joint_data



# baseline_session = 'front_side_back'
# baseline_session_index = labels.index(baseline_session)

# differences_list = []
# for count, joint_data in enumerate(mediapipe_joint_data_dict.values()):
#     if not baseline_session_index == count:
#         differences_list.append(joint_data-mediapipe_joint_data_dict[baseline_session_index])


plot_win = plotWindow()

for index in range(len(mediapipe_indices)):
    joint_to_plot = mediapipe_indices[index]

    figure = plt.figure()

    x_ax = figure.add_subplot(311)
    y_ax = figure.add_subplot(312)
    z_ax = figure.add_subplot(313)

    
    if joint_to_plot in qualisys_indices:
        qual_index = qualisys_indices.index(joint_to_plot)
        x_ax.plot(qualisys_sliced[:,qual_index,0]- qualisys_sliced[0,qual_index,0], label = 'Qualisys',  color = 'black', linewidth = 2, alpha = 1)
        y_ax.plot(qualisys_sliced[:,qual_index,1]- qualisys_sliced[0,qual_index,1], label = 'Qualisys',  color = 'black',linewidth = 2,alpha = 1)
        z_ax.plot(qualisys_sliced[:,qual_index,2]- qualisys_sliced[0,qual_index,2], label = 'Qualisys',  color = 'black',linewidth = 2,alpha = 1)


    for count,joint_data in enumerate(mediapipe_joint_data_dict.values()):
        x_ax.plot(joint_data[:,index,0]-joint_data[0,index,0], label = labels[count], alpha = .3)
        y_ax.plot(joint_data[:,index,1]-joint_data[0,index,1], label = labels[count], alpha = .3)
        z_ax.plot(joint_data[:,index,2]-joint_data[0,index,2], label = labels[count], alpha = .3)

    # for count,joint_diff in enumerate(differences_list):
    #     x_ax.plot(joint_diff[:,index,0], label = labels[count])
    #     y_ax.plot(joint_diff[:,index,1], label = labels[count])
    #     z_ax.plot(joint_diff[:,index,2], label = labels[count])


    x_ax.legend()
    y_ax.legend()
    z_ax.legend()

    x_ax.set_ylabel('X Axis Position (mm)')
    y_ax.set_ylabel('Y Axis Position (mm)')
    z_ax.set_ylabel('Z Axis Position (mm)')
    z_ax.set_xlabel('Frame #')

    plot_win.addPlot(f'{joint_to_plot} Position Trajectory',figure)
    figure.suptitle(f'{joint_to_plot} Position Trajectory')

plot_win.show()
