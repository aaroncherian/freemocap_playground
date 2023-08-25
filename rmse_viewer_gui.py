
from pathlib import Path
import numpy as np

from PyQt6.QtWidgets import QMainWindow, QWidget, QApplication, QHBoxLayout,QVBoxLayout, QLabel

from freemocap_utils.GUI_widgets.rmse_widgets.timeseries_plot_widget import TimeSeriesViewWidget
from freemocap_utils.GUI_widgets.rmse_widgets.rmse_plot_widget import RMSEViewWidget
from freemocap_utils.GUI_widgets.time_syncing_widgets.marker_selector_widget import MarkerSelectorWidget
from freemocap_utils.GUI_widgets.rmse_widgets.frame_selector_widget import FrameSelectorWidget
from freemocap_utils.GUI_widgets.rmse_widgets.RMSE_calculator import calculate_rmse_dataframe
from freemocap_utils.GUI_widgets.rmse_widgets.reference_frame_widget import ReferenceFrameWidget

import matplotlib.pyplot as plt

from pathlib import Path
import numpy as np

from PyQt6.QtWidgets import QMainWindow, QWidget, QApplication, QHBoxLayout,QVBoxLayout

from freemocap_utils.mediapipe_skeleton_builder import mediapipe_indices
from freemocap_utils.qualisys_indices import qualisys_indices


class RMSEViewerGUI(QMainWindow):
    def __init__(self, qualisys_data_original:np.ndarray, freemocap_data_original:np.ndarray):
        super().__init__()
        layout = QVBoxLayout()

        widget = QWidget()

        self.freemocap_data_original = freemocap_data_original
        self.qualisys_data_original = qualisys_data_original

        self.freemocap_data = freemocap_data_original.copy()
        self.qualisys_data = qualisys_data_original.copy()

        self.marker_selector_widget = MarkerSelectorWidget()
        layout.addWidget(self.marker_selector_widget)

        plots_layout = QHBoxLayout()
        layout.addLayout(plots_layout)

        self.time_series_viewer_widget = TimeSeriesViewWidget()
        plots_layout.addWidget(self.time_series_viewer_widget)

        self.rmse_viewer_widget = RMSEViewWidget()
        plots_layout.addWidget(self.rmse_viewer_widget)

        self.reference_frame_widget = ReferenceFrameWidget()
        layout.addWidget(self.reference_frame_widget)

        self.data_size_label = QLabel(f'Original Data Size = {freemocap_data_original.shape[0]}')
        layout.addWidget(self.data_size_label)

        self.frame_selector_widget = FrameSelectorWidget(qualisys_data=qualisys_data_original, freemocap_data=freemocap_data_original)
        layout.addWidget(self.frame_selector_widget)

        widget.setLayout(layout)
        self.setCentralWidget(widget)
        self.connect_signals_to_slots()
        self.run_initial_plots()

    def connect_signals_to_slots(self):
        self.marker_selector_widget.marker_to_plot_updated_signal.connect(self.handle_plotting)
        self.frame_selector_widget.frame_intervals_updated_signal.connect(self.handle_updated_frames)
        self.reference_frame_widget.reference_frame_updated_signal.connect(self.zero_data)
        self.reference_frame_widget.reset_signal.connect(self.run_initial_plots)

    def run_initial_plots(self):
        # self.zero_data()
        self.time_series_viewer_widget.update_plot(self.marker_selector_widget.current_marker,freemocap_data=self.freemocap_data_original, qualisys_data=self.qualisys_data_original            )
        self.rmse_viewer_widget.update_plot(qualisys_data=self.qualisys_data, freemocap_data=self.freemocap_data)

    def handle_plotting(self):
        # self.zero_data()
        self.time_series_viewer_widget.update_plot(self.marker_selector_widget.current_marker,freemocap_data=self.freemocap_data, qualisys_data=self.qualisys_data)
        self.rmse_viewer_widget.update_plot(qualisys_data=self.qualisys_data, freemocap_data=self.freemocap_data)

        def calculate_rmse_per_timepoint_per_dimension(qualisys_data, freemocap_data, qualisys_indices, mediapipe_indices):
            num_timepoints, _, _ = qualisys_data.shape
            rmse_per_timepoint_per_dimension = np.zeros((num_timepoints, 3))

            for marker_name in mediapipe_indices:
                if marker_name in qualisys_indices:
                    mediapipe_marker_index = mediapipe_indices.index(marker_name)
                    qualisys_marker_index = qualisys_indices.index(marker_name)

                    for dimension in range(3):
                        for timepoint in range(num_timepoints):
                            predictions = qualisys_data[timepoint, qualisys_marker_index, dimension]
                            targets = freemocap_data[timepoint, mediapipe_marker_index, dimension]

                            squared_error = (predictions - targets) ** 2
                            rmse_per_timepoint_per_dimension[timepoint, dimension] += np.sqrt(squared_error)

            shared_markers_count = len(set(qualisys_indices) & set(mediapipe_indices))
            rmse_per_timepoint_per_dimension /= shared_markers_count

            return rmse_per_timepoint_per_dimension

        # rmse_per_timepoint_per_dimension = calculate_rmse_per_timepoint_per_dimension(
        #     self.qualisys_data, self.freemocap_data, qualisys_indices, mediapipe_indices
        # )

        # plt.plot(rmse_per_timepoint_per_dimension[:, 0], label='X')
        # plt.plot(rmse_per_timepoint_per_dimension[:, 1], label='Y')
        # plt.plot(rmse_per_timepoint_per_dimension[:, 2], label='Z')

        # plt.xlabel('Time Point')
        # plt.ylabel('RMSE (Average Over Markers)')
        # plt.legend()
        # plt.title('Average RMSE Over Time for Each Dimension of Shared Markers')
        # plt.show()


    def handle_updated_frames(self):
        self.qualisys_data = self.frame_selector_widget.get_sliced_qualisys_data()
        self.freemocap_data = self.frame_selector_widget.get_sliced_freemocap_data()
        self.handle_plotting()

        # Calculate RMSE dataframe for the specified frames

    def zero_data(self, reference_frame):
        self.freemocap_data = self.freemocap_data_original[:,:,:] - self.freemocap_data_original[reference_frame,:,:]
        self.qualisys_data = self.qualisys_data_original[:,:,:] - self.qualisys_data_original[reference_frame,:,:]
        self.handle_plotting()
    
        # difference = abs(np.nanmean(self.freemocap_data[:,:,:]) - np.nanmean(self.qualisys_data[:,:,:]))
        # self.freemocap_data = self.freemocap_data[:,:,:] -difference
        # self.qualisys_data = self.qualisys_data[:,:,:] - difference

path_to_freemocap_session_folder = Path(r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_14_53_48_MDN_NIH_Trial3"
)
freemocap_data = np.load(path_to_freemocap_session_folder/'output_data'/'mediapipe_body_3d_xyz.npy')

path_to_qualisys_session_folder = Path(r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\qualisys_MDN_NIH_Trial3")
qualisys_data = np.load(path_to_qualisys_session_folder/'output_data'/'clipped_qualisys_skel_3d.npy')


# freemocap_sliced_data = freemocap_data[1162:6621,:,:]


app = QApplication([])
win = RMSEViewerGUI(qualisys_data_original=qualisys_data, freemocap_data_original=freemocap_data)

win.show()
app.exec()