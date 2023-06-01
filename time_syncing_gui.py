
from pathlib import Path
import numpy as np

from PyQt6.QtWidgets import QMainWindow, QWidget, QApplication, QHBoxLayout,QVBoxLayout, QPushButton

from freemocap_utils.GUI_widgets.time_syncing_widgets.trajectory_view_widget import TrajectoryViewWidget
from freemocap_utils.GUI_widgets.time_syncing_widgets.marker_selector_widget import MarkerSelectorWidget
from freemocap_utils.GUI_widgets.time_syncing_widgets.start_end_frame_selector_widget import FrameSelectorWidget
from freemocap_utils.GUI_widgets.time_syncing_widgets.clipping_widget import ClipWidget


class FileManager():
    def __init__(self, path_to_freemocap_session_folder:Path, path_to_qualisys_session_folder:Path):
        
        self.path_to_qualisys_session_folder = path_to_qualisys_session_folder
        self.path_to_freemocap_session_folder = path_to_freemocap_session_folder

        self.data_folder_name = 'output_data'

        path_to_qualisys_data = self.path_to_qualisys_session_folder/self.data_folder_name/'downsampled_qualisys_skel_3d.npy'
        path_to_freemocap_data = self.path_to_freemocap_session_folder/self.data_folder_name/'mediaPipeSkel_3d_body_hands_face.npy'
        self.qualisys_data = np.load(path_to_qualisys_data)
        self.freemocap_data = np.load(path_to_freemocap_data)

        path_to_qualisys_com_data = self.path_to_qualisys_session_folder/self.data_folder_name/'center_of_mass'/'downsampled_total_body_center_of_mass_xyz.npy'
        self.qualisys_com_data = np.load(path_to_qualisys_com_data)
        f = 2
        
    def save_clipped_qualisys_data(self, clipped_qualisys_data:np.ndarray, clipped_qualisys_com_data:np.ndarray):
        path_to_save_qualisys_data = self.path_to_qualisys_session_folder/self.data_folder_name/'clipped_qualisys_skel_3d.npy'
        np.save(path_to_save_qualisys_data,clipped_qualisys_data)

        path_to_save_qualisys_com_data = self.path_to_qualisys_session_folder/self.data_folder_name/'center_of_mass'/'total_body_center_of_mass_xyz.npy'
        np.save(path_to_save_qualisys_com_data,clipped_qualisys_com_data)

class MainWindow(QMainWindow):
    def __init__(self, path_to_freemocap_session_folder, path_to_qualisys_session_folder):
        super().__init__()

        self.setWindowTitle("My App")

        layout = QVBoxLayout()

        widget = QWidget()

        self.file_manager = FileManager(path_to_freemocap_session_folder=path_to_freemocap_session_folder, path_to_qualisys_session_folder=path_to_qualisys_session_folder)

        self.qualisys_data = self.file_manager.qualisys_data
        self.freemocap_data = self.file_manager.freemocap_data
        self.qualisys_com_data = self.file_manager.qualisys_com_data

        self.qualisys_start_end_frames = [0,self.qualisys_data.shape[0]]
        self.freemocap_start_end_frames = [0, self.freemocap_data.shape[0]]

        self.marker_selector_widget = MarkerSelectorWidget()
        layout.addWidget(self.marker_selector_widget)

        self.trajectory_view_widget = TrajectoryViewWidget()
        layout.addWidget(self.trajectory_view_widget)

        self.frame_selector_widget = FrameSelectorWidget(self.freemocap_start_end_frames,self.qualisys_start_end_frames)
        layout.addWidget(self.frame_selector_widget)

        self.data_clip_widget = ClipWidget(qualisys_3d_data=self.qualisys_data, freemocap_3d_data= self.freemocap_data, qualisys_com_data=self.qualisys_com_data)
        layout.addWidget(self.data_clip_widget)

        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.connect_signals_to_slots()

    def connect_signals_to_slots(self):
        self.marker_selector_widget.marker_to_plot_updated_signal.connect(lambda: self.trajectory_view_widget.update_plot(self.marker_selector_widget.current_marker,self.freemocap_data,self.qualisys_data, self.freemocap_start_end_frames, self.qualisys_start_end_frames, reset_ax_limits=True))
        self.frame_selector_widget.frame_intervals_updated_signal.connect(self.update_plot_with_new_frame_intervals)
        self.data_clip_widget.data_clipped_signal.connect(self.update_plot_with_clipped_data)

        self.data_clip_widget.save_clipped_data_signal.connect(self.file_manager.save_clipped_qualisys_data)

    def update_plot_with_new_frame_intervals(self):
        self.qualisys_start_end_frames = self.frame_selector_widget.qualisys_start_end_frames_new
        self.freemocap_start_end_frames = self.frame_selector_widget.freemocap_start_end_frames_new 
        self.trajectory_view_widget.update_plot(self.marker_selector_widget.current_marker, freemocap_data= self.freemocap_data, qualisys_data= self.qualisys_data, freemocap_start_end_frames=self.freemocap_start_end_frames, qualisys_start_end_frames=self.qualisys_start_end_frames)

    def update_plot_with_clipped_data(self):
        self.qualisys_start_end_frames = self.data_clip_widget.qualisys_start_end_frames_new
        self.freemocap_start_end_frames = self.data_clip_widget.freemocap_start_end_frames_new
        self.trajectory_view_widget.update_plot(self.marker_selector_widget.current_marker, freemocap_data= self.freemocap_data, qualisys_data= self.qualisys_data, freemocap_start_end_frames=self.freemocap_start_end_frames, qualisys_start_end_frames=self.qualisys_start_end_frames)
        self.frame_selector_widget.set_start_and_end_frames(freemocap_start_end_frames=self.freemocap_start_end_frames, qualisys_start_end_frames=self.qualisys_start_end_frames)

if __name__ == "__main__":

    
    path_to_qualisys_session_folder = Path(r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\qualisys_MDN_NIH_Trial4")
    #path_to_qualisys_session_folder = Path(r"D:\ValidationStudy2022\FreeMocap_Data\qualisys_sesh_2022-05-24_16_02_53_JSM_T1_BOS")
    #path_to_qualisys_session_folder = Path(r"D:\ValidationStudy2022\FreeMocap_Data\qualisys_sesh_2022-05-24_16_02_53_JSM_T1_NIH")

    path_to_freemocap_session_folder = Path(r'D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_15_03_20_MDN_NIH_Trial4')
    #path_to_freemocap_session_folder = Path(r'D:\ValidationStudy2022\FreeMocap_Data\sesh_2022-05-24_15_55_40_JSM_T1_BOS')
    #path_to_freemocap_session_folder = Path(r'D:\ValidationStudy2022\FreeMocap_Data\sesh_2022-05-24_16_02_53_JSM_T1_NIH')

    app = QApplication([])
    win = MainWindow(path_to_freemocap_session_folder=path_to_freemocap_session_folder, path_to_qualisys_session_folder=path_to_qualisys_session_folder)
    win.show()
    app.exec()
