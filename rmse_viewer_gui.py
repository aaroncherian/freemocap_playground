
from pathlib import Path
import numpy as np

from PyQt6.QtWidgets import QMainWindow, QWidget, QApplication, QHBoxLayout,QVBoxLayout, QLabel

from freemocap_utils.GUI_widgets.rmse_widgets.timeseries_plot_widget import TimeSeriesViewWidget
from freemocap_utils.GUI_widgets.rmse_widgets.rmse_plot_widget import RMSEViewWidget
from freemocap_utils.GUI_widgets.time_syncing_widgets.marker_selector_widget import MarkerSelectorWidget
from freemocap_utils.GUI_widgets.rmse_widgets.frame_selector_widget import FrameSelectorWidget


from pathlib import Path
import numpy as np

from PyQt6.QtWidgets import QMainWindow, QWidget, QApplication, QHBoxLayout,QVBoxLayout



class RMSEViewerGUI(QMainWindow):
    def __init__(self, qualisys_data_original:np.ndarray, freemocap_data_original:np.ndarray):
        super().__init__()
        layout = QVBoxLayout()

        widget = QWidget()

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

    def run_initial_plots(self):
        self.zero_data()
        self.time_series_viewer_widget.update_plot(self.marker_selector_widget.current_marker,freemocap_data=self.freemocap_data, qualisys_data=self.qualisys_data)
        self.rmse_viewer_widget.update_plot(qualisys_data=self.qualisys_data, freemocap_data=self.freemocap_data)

    def handle_plotting(self):
        self.zero_data()
        self.time_series_viewer_widget.update_plot(self.marker_selector_widget.current_marker,freemocap_data=self.freemocap_data, qualisys_data=self.qualisys_data)
        self.rmse_viewer_widget.update_plot(qualisys_data=self.qualisys_data, freemocap_data=self.freemocap_data)

    def handle_updated_frames(self):
        self.qualisys_data = self.frame_selector_widget.get_sliced_qualisys_data()
        self.freemocap_data = self.frame_selector_widget.get_sliced_freemocap_data()
        self.handle_plotting()

    def zero_data(self):
        self.freemocap_data = self.freemocap_data[:,:,:] - self.freemocap_data[0,:,:]
        self.qualisys_data = self.qualisys_data[:,:,:] - self.qualisys_data[0,:,:]

path_to_freemocap_session_folder = Path(r'D:\ValidationStudy_numCams\FreeMoCap_Data\sesh_2022-05-24_16_10_46_JSM_T1_WalkRun')
freemocap_data = np.load(path_to_freemocap_session_folder/'DataArrays'/'mediaPipeSkel_3d_origin_aligned.npy')

path_to_qualisys_session_folder = Path(r"D:\ValidationStudy2022\FreeMocap_Data\qualisys_sesh_2022-05-24_16_02_53_JSM_T1_WalkRun")
qualisys_data = np.load(path_to_qualisys_session_folder/'DataArrays'/'qualisys_marker_data_29Hz.npy')


freemocap_sliced_data = freemocap_data[1162:6621,:,:]


app = QApplication([])
win = RMSEViewerGUI(qualisys_data_original=qualisys_data, freemocap_data_original=freemocap_sliced_data)

win.show()
app.exec()