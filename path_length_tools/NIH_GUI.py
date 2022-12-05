
from PyQt6.QtWidgets import QMainWindow, QGridLayout, QWidget, QApplication

from freemocap_utils.GUI_widgets.skeleton_view_widget import SkeletonViewWidget
from freemocap_utils.GUI_widgets.slider_widget import FrameCountSlider
from freemocap_utils.GUI_widgets.video_capture_widget import VideoCapture


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("My App")

        layout = QGridLayout()

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


if __name__ == "__main__":

    app = QApplication([])
    win = MainWindow()
    win.show()
    app.exec()
