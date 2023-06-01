
from PyQt6.QtWidgets import QMainWindow, QWidget, QApplication, QHBoxLayout,QVBoxLayout, QPushButton, QFileDialog, QRadioButton

from freemocap_utils.GUI_widgets.skeleton_view_widget import SkeletonViewWidget
from freemocap_utils.GUI_widgets.slider_widget import FrameCountSlider
from freemocap_utils.GUI_widgets.video_capture_widget import VideoCapture
from freemocap_utils.GUI_widgets.NIH_widgets.frame_marking_widget import FrameMarker
from freemocap_utils.GUI_widgets.NIH_widgets.saving_data_analysis_widget import SavingDataAnalysisWidget
from freemocap_utils.GUI_widgets.NIH_widgets.balance_assessment_widget import BalanceAssessmentWidget
from freemocap_utils.mediapipe_skeleton_builder import build_skeleton, mediapipe_connections, mediapipe_indices, qualisys_indices

from pathlib import Path

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import numpy as np

class FileManager:
    def __init__(self):
        self.data_options = {
            "freemocap": {
                "marker_data_array_name": "mediapipe_body_3d_xyz.npy",
                "markers_to_use": mediapipe_indices
            },
            "qualisys": {
                "marker_data_array_name": "clipped_qualisys_skel_3d.npy",
                "markers_to_use": qualisys_indices
            }
        }
        
    def get_existing_directory(self, dialog_title="Choose a session"):
        folder_diag = QFileDialog()
        session_folder_path = QFileDialog.getExistingDirectory(None, dialog_title)
        return Path(session_folder_path) if session_folder_path else None

    def load_skeleton_data(self, session_folder_path, marker_data_array_name):
        skeleton_data_folder_path = session_folder_path / 'output_data' / marker_data_array_name
        return np.load(skeleton_data_folder_path)
    
    def get_data_option(self, option_name):
        return self.data_options.get(option_name)
    
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("My App")

        layout = QVBoxLayout()

        widget = QWidget()

        self.file_manager = FileManager()

        slider_and_skeleton_layout = QVBoxLayout()

        self.frame_count_slider = FrameCountSlider()
        slider_and_skeleton_layout.addWidget(self.frame_count_slider)

        self.folder_open_button = QPushButton('Load a session folder',self)
        slider_and_skeleton_layout.addWidget(self.folder_open_button)
        self.folder_open_button.clicked.connect(self.open_folder_dialog)

        self.freemocap_radio = QRadioButton('Load FreeMoCap Data')
        self.freemocap_radio.setChecked(True)
        slider_and_skeleton_layout.addWidget(self.freemocap_radio)
        self.qualisys_radio = QRadioButton('Load Qualisys Data')
        slider_and_skeleton_layout.addWidget(self.qualisys_radio)

        self.skeleton_view_widget = SkeletonViewWidget()
        self.skeleton_view_widget.setFixedSize(self.skeleton_view_widget.size())
        slider_and_skeleton_layout.addWidget(self.skeleton_view_widget)
        
        self.camera_view_widget = VideoCapture()

        self.camera_view_widget.setFixedSize(self.skeleton_view_widget.size())

        skeleton_plot_and_video_layout = QHBoxLayout()
        skeleton_plot_and_video_layout.addLayout(slider_and_skeleton_layout)
        skeleton_plot_and_video_layout.addWidget(self.camera_view_widget)

        layout.addLayout(skeleton_plot_and_video_layout)

        self.frame_marking_widget = FrameMarker()
        layout.addWidget(self.frame_marking_widget)
        self.frame_marking_widget.setFixedSize(640,200)

        self.balance_assessment_widget = BalanceAssessmentWidget()
        layout.addWidget(self.balance_assessment_widget)

        self.saving_data_widget = SavingDataAnalysisWidget()
        layout.addWidget(self.saving_data_widget)

        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.connect_signals_to_slots()

        # self.setFixedSize(layout.sizeHint())

    def connect_signals_to_slots(self):
        self.frame_count_slider.slider.valueChanged.connect(lambda: self.skeleton_view_widget.replot(self.frame_count_slider.slider.value()))

        self.frame_marking_widget.conditions_dict_updated_signal.connect(lambda: self.saving_data_widget.set_conditions_frames_dictionary(self.frame_marking_widget.condition_widget_dictionary))
        self.frame_marking_widget.conditions_dict_updated_signal.connect(lambda: self.balance_assessment_widget.set_conditions_frames_dictionary(self.frame_marking_widget.condition_widget_dictionary))

        self.frame_count_slider.slider.valueChanged.connect(lambda: self.camera_view_widget.set_frame(self.frame_count_slider.slider.value()) if (self.camera_view_widget.video_loader.video_is_loaded) else NotImplemented)
        
        self.balance_assessment_widget.run_button_clicked_signal.connect(self.show_histograms)
        self.balance_assessment_widget.run_button_clicked_signal.connect(lambda: self.saving_data_widget.set_conditions_path_length_dictionary(self.balance_assessment_widget.path_length_dictionary))
        self.balance_assessment_widget.run_button_clicked_signal.connect(lambda: self.saving_data_widget.set_histogram_figure(self.window.histogram_plots.figure))
        self.balance_assessment_widget.run_button_clicked_signal.connect(lambda: self.saving_data_widget.set_velocity_dictionary(self.balance_assessment_widget.velocity_dictionary))
    
    def _handle_session_folder_loaded(self):
        self.frame_count_slider.set_slider_range(self.num_frames)
        self.enable_buttons()
        self.set_session_folder_path()
        
    def open_folder_dialog(self):
        self.session_folder_path = self.file_manager.get_existing_directory("Choose a session")

        if self.session_folder_path:

            if self.freemocap_radio.isChecked():
                marker_data_array_name = 'mediapipe_body_3d_xyz.npy'
                markers_to_use = mediapipe_indices
            elif self.qualisys_radio.isChecked():
                marker_data_array_name = 'clipped_qualisys_skel_3d.npy'
                markers_to_use = qualisys_indices

            self.skel3d_data = self.file_manager.load_skeleton_data(self.session_folder_path, marker_data_array_name)
            self.build_mediapipe_skeleton(markers_to_use)

    def build_mediapipe_skeleton(self, markers_to_use:list):

        #self.mediapipe_skeleton = build_skeleton(self.skel3d_data,mediapipe_indices,mediapipe_connections)
        self.mediapipe_skeleton = build_skeleton(self.skel3d_data,markers_to_use,mediapipe_connections)

        self.num_frames = self.skel3d_data.shape[0]
        # self.reset_slider()
        self.skeleton_view_widget.reset_skeleton_3d_plot(self.skel3d_data, self.mediapipe_skeleton)
        #self.reset_skeleton_3d_plot()
        #self.session_folder_loaded_signal.emit()
        self._handle_session_folder_loaded()
    
    def set_session_folder_path(self):
        self.camera_view_widget.video_loader.set_session_folder_path(self.session_folder_path)
        self.saving_data_widget.set_session_folder_path(self.session_folder_path)
        self.balance_assessment_widget.set_session_folder_path(self.session_folder_path)
    
    def enable_buttons(self):
        self.balance_assessment_widget.run_path_length_analysis_button.setEnabled(True)
        self.camera_view_widget.video_loader.videoLoadButton.setEnabled(True)
        self.frame_marking_widget.save_condition_button.setEnabled(True)
        self.frame_marking_widget.load_conditions_button.setEnabled(True)
        self.saving_data_widget.save_data_button.setEnabled(True)

    
    def set_condition_frames_dictionary(self, condition_frames_dictionary:dict):
        self.condition_frames_dictionary = condition_frames_dictionary

    def show_histograms(self):
        self.window = HistogramWindow(self.balance_assessment_widget.velocity_dictionary)
        self.window.show()

class HistogramWindow(QMainWindow):
    def __init__(self, velocities_dict:dict, parent = None):
        super().__init__()
        self.layout = QVBoxLayout()
        
        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.setCentralWidget(self.widget)

        x_array_list = [item[0] for item in velocities_dict.values()] #grab the X dimension of each condition to plot 
        conditions_list = list(velocities_dict.keys())

        velocities_dict = dict(zip(conditions_list,x_array_list))

        self.velocities_dict = velocities_dict
        self.setWindowTitle("Window22222")
        self.create_subplots()
        


    def create_subplots(self):
        self.histogram_plots = Mpl3DPlotCanvas()

        ylimit = 120
        hist_range = (-.5,.5)
        num_bins = 75
        alpha_val = .75

        num_rows = len(self.velocities_dict)
        for count, condition in enumerate(self.velocities_dict, start=1):
            self.ax = self.histogram_plots.figure.add_subplot(num_rows,1,count)
            self.ax.set_title(condition)
            self.ax.set_ylim(0,ylimit)

            # self.ax.plot(self.velocities_dict[condition])

            self.ax.hist(self.velocities_dict[condition], bins = num_bins, range = hist_range,label = condition, alpha = alpha_val)
            self.histogram_plots.figure.canvas.draw_idle()

        
        # self.histogram_plots.figure.savefig('temp.png')
        self.layout.addWidget(self.histogram_plots)

        
class Mpl3DPlotCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=4, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        super(Mpl3DPlotCanvas, self).__init__(fig)

            

    

        
if __name__ == "__main__":

    app = QApplication([])
    win = MainWindow()

    win.show()
    app.exec()
