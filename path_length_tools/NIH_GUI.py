
from PyQt6.QtWidgets import QMainWindow, QGridLayout, QWidget, QApplication, QHBoxLayout,QVBoxLayout

from freemocap_utils.GUI_widgets.skeleton_view_widget import SkeletonViewWidget
from freemocap_utils.GUI_widgets.slider_widget import FrameCountSlider
from freemocap_utils.GUI_widgets.video_capture_widget import VideoCapture
from freemocap_utils.GUI_widgets.NIH_widgets.frame_marking_widget import FrameMarker

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("My App")

        layout = QVBoxLayout()

        widget = QWidget()


        slider_and_skeleton_layout = QVBoxLayout()

        self.frame_count_slider = FrameCountSlider()
        slider_and_skeleton_layout.addWidget(self.frame_count_slider)

        self.skeleton_view_widget = SkeletonViewWidget()
        self.skeleton_view_widget.setFixedSize(self.skeleton_view_widget.size())
        slider_and_skeleton_layout.addWidget(self.skeleton_view_widget)
        
        # layout.addLayout(slider_and_skeleton_layout)

        self.camera_view_widget = VideoCapture()
        # layout.addWidget(self.camera_view_widget)
        self.camera_view_widget.setFixedSize(self.skeleton_view_widget.size())

        skeleton_plot_and_video_layout = QHBoxLayout()
        skeleton_plot_and_video_layout.addLayout(slider_and_skeleton_layout)
        skeleton_plot_and_video_layout.addWidget(self.camera_view_widget)

        layout.addLayout(skeleton_plot_and_video_layout)

        self.frame_marking_widget = FrameMarker()
        layout.addWidget(self.frame_marking_widget)
        self.frame_marking_widget.setFixedSize(640,200)

        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.connect_signals_to_slots()

        self.setFixedSize(layout.sizeHint())

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
