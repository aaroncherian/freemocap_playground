from skellyforge.freemocap_utils.postprocessing_widgets.visualization_widgets.timeseries_view_widget import TimeSeriesPlotterWidget
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton
from pathlib import Path
import numpy as np

path_to_recording_folder = Path(r'D:\2023-06-07_JH\1.0_recordings\treadmill_calib\sesh_2023-06-07_12_38_16_JH_leg_length_neg_5_trial_1')
path_to_data = path_to_recording_folder/'output_data'/'raw_data'/'mediapipe_deeplabcut_3dData_spliced.npy'
path_to_data = path_to_recording_folder/'output_data'/'mediapipe_body_3d_xyz.npy'
data = np.load(path_to_data)

class TimeSeriesApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Time Series Plotter")
        self.setGeometry(100, 100, 1200, 800)

        self.plotter_widget = TimeSeriesPlotterWidget()
        self.setCentralWidget(self.plotter_widget)

        # Dummy data for demonstration (Replace with actual data)
        self.original_data = data
        self.processed_data = data

        # A button to trigger the plot update (for demonstration purposes)
        self.plotter_widget.update_plot('left_heel', self.original_data, self.processed_data)


if __name__ == "__main__":
    app = QApplication([])
    window = TimeSeriesApp()
    window.show()
    app.exec()

