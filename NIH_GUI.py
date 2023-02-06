
from PyQt6.QtWidgets import QMainWindow, QWidget, QApplication, QHBoxLayout,QVBoxLayout

from freemocap_utils.GUI_widgets.skeleton_view_widget import SkeletonViewWidget
from freemocap_utils.GUI_widgets.slider_widget import FrameCountSlider
from freemocap_utils.GUI_widgets.video_capture_widget import VideoCapture
from freemocap_utils.GUI_widgets.NIH_widgets.frame_marking_widget import FrameMarker
from freemocap_utils.GUI_widgets.NIH_widgets.saving_data_analysis_widget import SavingDataAnalysisWidget
from freemocap_utils.GUI_widgets.NIH_widgets.balance_assessment_widget import BalanceAssessmentWidget

from pathlib import Path

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


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

        self.skeleton_view_widget.session_folder_loaded_signal.connect(lambda: self.frame_count_slider.set_slider_range(self.skeleton_view_widget.num_frames))
        self.skeleton_view_widget.session_folder_loaded_signal.connect(self.enable_buttons)
        self.skeleton_view_widget.session_folder_loaded_signal.connect(lambda: self.set_session_folder_path(self.skeleton_view_widget.session_folder_path))

        self.frame_marking_widget.conditions_dict_updated_signal.connect(lambda: self.saving_data_widget.set_conditions_frames_dictionary(self.frame_marking_widget.condition_widget_dictionary))
        self.frame_marking_widget.conditions_dict_updated_signal.connect(lambda: self.balance_assessment_widget.set_conditions_frames_dictionary(self.frame_marking_widget.condition_widget_dictionary))

        self.frame_count_slider.slider.valueChanged.connect(lambda: self.camera_view_widget.set_frame(self.frame_count_slider.slider.value()) if (self.camera_view_widget.video_loader.video_is_loaded) else NotImplemented)
        
        self.balance_assessment_widget.run_button_clicked_signal.connect(self.show_histograms)
        self.balance_assessment_widget.run_button_clicked_signal.connect(lambda: self.saving_data_widget.set_conditions_path_length_dictionary(self.balance_assessment_widget.path_length_dictionary))
        self.balance_assessment_widget.run_button_clicked_signal.connect(lambda: self.saving_data_widget.set_histogram_figure(self.window.histogram_plots.figure))
        self.balance_assessment_widget.run_button_clicked_signal.connect(lambda: self.saving_data_widget.set_velocity_dictionary(self.balance_assessment_widget.velocity_dictionary))
    
    def set_session_folder_path(self,session_folder_path:Path):
        self.session_folder_path = session_folder_path
        self.camera_view_widget.video_loader.set_session_folder_path(self.skeleton_view_widget.session_folder_path)
        self.saving_data_widget.set_session_folder_path(self.skeleton_view_widget.session_folder_path)
        self.balance_assessment_widget.set_session_folder_path(self.skeleton_view_widget.session_folder_path)
    
    def enable_buttons(self):
        self.balance_assessment_widget.run_path_length_analysis_button.setEnabled(True)
        self.camera_view_widget.video_loader.videoLoadButton.setEnabled(True)
        self.frame_marking_widget.save_condition_button.setEnabled(True)
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
