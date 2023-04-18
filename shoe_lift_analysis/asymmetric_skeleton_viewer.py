
from pathlib import Path
import numpy as np

from PyQt6.QtWidgets import QMainWindow, QApplication, QWidget, QVBoxLayout, QGroupBox, QPushButton, QHBoxLayout
from PyQt6.QtCore import QTimer

from freemocap_utils.shoe_lift_widgets.slider_widget import FrameCountSlider
from freemocap_utils.shoe_lift_widgets.skeleton_viewers_container import SkeletonViewersContainer

class FileManager:
    def __init__(self, path_to_recording: str):
        self.path_to_recording = path_to_recording
        self.data_array_path = self.path_to_recording
    
    def load_skeleton_data(self, file_names: list[str]) -> list[np.ndarray]:
        skeleton_data_list = []
        for file_name in file_names:
            skeleton_data = np.load(self.data_array_path / file_name/'output_data'/'average_step_3d.npy')
            skeleton_data = skeleton_data[:, 0:33, :]
            skeleton_data_list.append(skeleton_data)
        return skeleton_data_list

    def save_skeleton_data(self, skeleton_data, skeleton_file_name):
        np.save(self.data_array_path/skeleton_file_name,skeleton_data)

class MultiSkeletonViewer(QWidget):
    def __init__(self,path_to_data_folder:Path,  file_names: list[str]):
        super().__init__()

        layout = QVBoxLayout()

        self.file_manager = FileManager(path_to_recording=path_to_data_folder)

        self.skeletons_data = self.file_manager.load_skeleton_data(file_names)

        skeleton_viewer_groupbox = self.create_skeleton_viewer_groupbox()
        layout.addWidget(skeleton_viewer_groupbox)

        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.update_animation)


        self.skeleton_viewers_container.plot_raw_skeleton(self.skeletons_data)

        self.setLayout(layout)

        self.connect_signals_to_slots()

    def start_animation(self):
        frame_rate = 1000 / 10  # Set the frame rate in milliseconds (e.g., 1000 ms / 30 frames = 33.33 ms per frame)
        self.animation_timer.start(frame_rate)

    def stop_animation(self):
        self.animation_timer.stop()

    def update_animation(self):
        current_frame = self.frame_count_slider.slider.value()
        num_frames = self.skeletons_data[0].shape[0]

        next_frame = (current_frame + 1) % num_frames
        self.frame_count_slider.slider.setValue(next_frame)

    def connect_signals_to_slots(self):
        self.frame_count_slider.slider.valueChanged.connect(lambda: self.update_viewer_plots(self.frame_count_slider.slider.value()))
        self.play_button.clicked.connect(self.start_animation)
        self.stop_button.clicked.connect(self.stop_animation)


    def update_viewer_plots(self, frame_to_plot):
        self.skeleton_viewers_container.update_raw_viewer_plot(frame_to_plot)
   
    def create_skeleton_viewer_groupbox(self):
        groupbox = QGroupBox('View your raw and processed mocap data')
        viewer_layout = QVBoxLayout()

        control_buttons_layout = QHBoxLayout()
        self.play_button = QPushButton('Play')
        control_buttons_layout.addWidget(self.play_button)

        self.stop_button = QPushButton('Stop')
        control_buttons_layout.addWidget(self.stop_button)

        viewer_layout.addLayout(control_buttons_layout)

        self.frame_count_slider = FrameCountSlider(num_frames=self.skeletons_data[0].shape[0])
        viewer_layout.addWidget(self.frame_count_slider)
        self.skeleton_viewers_container = SkeletonViewersContainer()
        viewer_layout.addWidget(self.skeleton_viewers_container)
        groupbox.setLayout(viewer_layout)
        return groupbox



class MainWindow(QMainWindow):
    def __init__(self,path_to_data_folder:Path,  file_names: list[str]):
        super().__init__()

        layout = QVBoxLayout()

        widget = QWidget()
        postprocessing_window = MultiSkeletonViewer(path_to_data_folder,file_names)

        layout.addWidget(postprocessing_window)

        widget.setLayout(layout)
        self.setCentralWidget(widget)


if __name__ == "__main__":
    
    path_to_freemocap_session_folder = Path(r'D:\ValidationStudy2022\FreeMocap_Data\sesh_2022-05-24_16_10_46_JSM_T1_WalkRun')
    #path_to_freemocap_session_folder = Path(r'C:\Users\aaron\FreeMocap_Data\recording_sessions\recording_15_20_51_gmt-4__brit_half_inch')
    #path_to_freemocap_session_folder = Path(r'C:\Users\aaron\FreeMocap_Data\recording_sessions\recording_15_22_56_gmt-4__brit_one_inch')

    # path_to_freemocap_session_folder = Path(r'C:\Users\aaron\FreeMocap_Data\recording_sessions\recording_15_19_00_gmt-4__brit_baseline')
    # freemocap_raw_data = np.load(path_to_freemocap_session_folder/'output_data'/'raw_data'/'mediapipe3dData_numFrames_numTrackedPoints_spatialXYZ.npy')

    # freemocap_raw_data = np.load(path_to_freemocap_session_folder/'DataArrays'/'mediaPipeSkel_3d.npy')
    # freemocap_raw_data = freemocap_raw_data[:,0:33,:]


    # path_to_data_folder = Path(r'D:\ValidationStudy2022\FreeMocap_Data\sesh_2022-05-24_16_10_46_JSM_T1_WalkRun')
    path_to_data_folder = Path(r'C:\Users\aaron\FreeMocap_Data\recording_sessions')
    file_names = ['recording_15_19_00_gmt-4__brit_baseline','recording_15_20_51_gmt-4__brit_half_inch', 'recording_15_22_56_gmt-4__brit_one_inch','recording_15_24_58_gmt-4__brit_two_inch']
    #file_names = ['recording_15_19_00_gmt-4__brit_baseline','recording_15_20_51_gmt-4__brit_half_inch']

    app = QApplication([])
    win = MainWindow(path_to_data_folder, file_names)
    win.show()
    app.exec()