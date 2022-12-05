
from PyQt6.QtWidgets import QMainWindow, QGridLayout, QWidget, QApplication, QHBoxLayout

from freemocap_utils.GUI_widgets.skeleton_view_widget import SkeletonViewWidget
from freemocap_utils.GUI_widgets.slider_widget import FrameCountSlider
from freemocap_utils.GUI_widgets.video_capture_widget import VideoCapture


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("My App")

        layout = QHBoxLayout()

        widget = QWidget()

        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.frame_count_slider = FrameCountSlider()
        layout.addWidget(self.frame_count_slider)

        self.skeleton_view_widget = SkeletonViewWidget()
        layout.addWidget(self.skeleton_view_widget)

        self.camera_view_widget = VideoCapture()
        layout.addWidget(self.camera_view_widget)

        self.connect_signals_to_slots()

    def connect_signals_to_slots(self):
        self.frame_count_slider.slider.valueChanged.connect(lambda: self.skeleton_view_widget.replot(self.frame_count_slider.slider.value()))

        self.skeleton_view_widget.session_folder_loaded_signal.connect(lambda: self.frame_count_slider.set_slider_range(self.skeleton_view_widget.num_frames))
        self.skeleton_view_widget.session_folder_loaded_signal.connect(lambda: self.camera_view_widget.video_loader.videoLoadButton.setEnabled(True))
        self.skeleton_view_widget.session_folder_loaded_signal.connect(lambda: self.camera_view_widget.video_loader.set_session_folder_path(self.skeleton_view_widget.session_folder_path))

 
        self.frame_count_slider.slider.valueChanged.connect(lambda: self.camera_view_widget.set_frame(self.frame_count_slider.slider.value()) if (self.camera_view_widget.video_loader.video_is_loaded) else NotImplemented)


        
if __name__ == "__main__":

    app = QApplication([])
    win = MainWindow()
    win.show()
    app.exec()
