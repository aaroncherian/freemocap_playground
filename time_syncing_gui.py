
from pathlib import Path
import numpy as np

from PyQt6.QtWidgets import QMainWindow, QWidget, QApplication, QHBoxLayout,QVBoxLayout

from freemocap_utils.GUI_widgets.time_syncing_widgets.trajectory_view_widget import TrajectoryViewWidget
from freemocap_utils.GUI_widgets.time_syncing_widgets.marker_selector_widget import MarkerSelectorWidget
from freemocap_utils.GUI_widgets.time_syncing_widgets.start_end_frame_selector_widget import FrameSelectorWidget


class MainWindow(QMainWindow):
    def __init__(self, qualisys_data, freemocap_data):
        super().__init__()

        self.setWindowTitle("My App")

        layout = QVBoxLayout()

        widget = QWidget()

        self.qualisys_data = qualisys_data
        self.freemocap_data = freemocap_data

        self.qualisys_start_end_frames = [0,qualisys_data.shape[0]]
        self.freemocap_start_end_frames = [0, freemocap_data.shape[0]]

        self.marker_selector_widget = MarkerSelectorWidget()
        layout.addWidget(self.marker_selector_widget)

        self.trajectory_view_widget = TrajectoryViewWidget()
        layout.addWidget(self.trajectory_view_widget)

        self.frame_selector_widget = FrameSelectorWidget(self.freemocap_start_end_frames,self.qualisys_start_end_frames)
        layout.addWidget(self.frame_selector_widget)

        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.connect_signals_to_slots()

    def connect_signals_to_slots(self):
        self.marker_selector_widget.marker_to_plot_updated_signal.connect(lambda: self.trajectory_view_widget.update_plot(self.marker_selector_widget.current_marker,self.freemocap_data,self.qualisys_data, self.freemocap_start_end_frames, self.qualisys_start_end_frames, reset_ax_limits=True))
        self.frame_selector_widget.frame_intervals_updated_signal.connect(self.update_plot_with_new_frame_intervals)

    def update_plot_with_new_frame_intervals(self):
        self.qualisys_start_end_frames = self.frame_selector_widget.qualisys_start_end_frames_new
        self.freemocap_start_end_frames = self.frame_selector_widget.freemocap_start_end_frames_new 
        self.trajectory_view_widget.update_plot(self.marker_selector_widget.current_marker, freemocap_data= self.freemocap_data, qualisys_data= self.qualisys_data, freemocap_start_end_frames=self.freemocap_start_end_frames, qualisys_start_end_frames=self.qualisys_start_end_frames)
if __name__ == "__main__":

    
    path_to_qualisys_session_folder = Path(r"D:\2023-05-17_MDN_NIH_data\qtm_data\qualisys_MDN_NIH_Trial2")
    #path_to_qualisys_session_folder = Path(r"D:\ValidationStudy2022\FreeMocap_Data\qualisys_sesh_2022-05-24_16_02_53_JSM_T1_BOS")
    #path_to_qualisys_session_folder = Path(r"D:\ValidationStudy2022\FreeMocap_Data\qualisys_sesh_2022-05-24_16_02_53_JSM_T1_NIH")
    #qualisys_data = np.load(path_to_qualisys_session_folder/'DataArrays'/'downsampled_qualisys_3D.npy')
    #qualisys_data = np.load(path_to_qualisys_session_folder/'DataArrays'/'qualisys_marker_data_30Hz.npy')
    qualisys_data = np.load(path_to_qualisys_session_folder/'output_data'/'downsampled_qualisys_skel_3d.npy')

    path_to_freemocap_session_folder = Path(r'D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_14_40_56_MDN_NIH_Trial2')
    #path_to_freemocap_session_folder = Path(r'D:\ValidationStudy2022\FreeMocap_Data\sesh_2022-05-24_15_55_40_JSM_T1_BOS')
    #path_to_freemocap_session_folder = Path(r'D:\ValidationStudy2022\FreeMocap_Data\sesh_2022-05-24_16_02_53_JSM_T1_NIH')
    #freemocap_data = np.load(path_to_freemocap_session_folder/'DataArrays'/'freemocap_marker_data_30Hz.npy')
    freemocap_data = np.load(path_to_freemocap_session_folder/'output_data'/'mediaPipeSkel_3d_body_hands_face.npy')

    ##77 for BOS
    ##1160

    #np.save(path_to_freemocap_session_folder/'DataArrays'/'freemocap_data_resized.npy', freemocap_data[26:10304,:,:])

    app = QApplication([])
    win = MainWindow(qualisys_data,freemocap_data)

    win.show()
    app.exec()
