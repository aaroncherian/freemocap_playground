from PyQt6.QtWidgets import QWidget, QHBoxLayout
from freemocap_utils.shoe_lift_widgets.skeleton_view_widget import SkeletonViewWidget


class SkeletonViewersContainer(QWidget):
    def __init__(self):
        super().__init__()

        layout = QHBoxLayout()

        self.raw_skeleton_viewer = SkeletonViewWidget('Raw data')
        layout.addWidget(self.raw_skeleton_viewer)

        self.setLayout(layout)

    def plot_raw_skeleton(self, skeletons_3d_data_list):
        colors = ["blue", "red", 'pink', 'green']
        self.raw_skeleton_viewer.load_skeletons(skeletons_3d_data_list =skeletons_3d_data_list,colors=colors)

    def update_raw_viewer_plot(self, frame_number):
        self.raw_skeleton_viewer.replot(frame_number)
