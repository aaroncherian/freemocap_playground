from freemocap_utils.freemocap_data_loader import FreeMoCapDataLoader
from PyQt6.QtWidgets import QWidget,QVBoxLayout, QLineEdit, QPushButton ,QFormLayout, QLabel
from path_length_tools import PathLengthCalculator

from pathlib import Path
import json



class BalanceAssessmentWidget(QWidget):
    def __init__(self):
        super().__init__()

        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        run_path_length_analysis_button = QPushButton('Run balance assessment')
        run_path_length_analysis_button.clicked.connect(self.run_COM_analysis)
        self._layout.addWidget(run_path_length_analysis_button)

        self.path_length_results = QLabel()
        self._layout.addWidget(self.path_length_results)



        # run_and_save_path_length_button.clicked.connect.run_COM_analysis()

    def set_session_folder_path(self, path_to_session_folder:Path):
        self.path_to_session_folder = path_to_session_folder

    def set_data_analysis_folder_path(self, path_to_data_analysis_folder:Path):
        self.path_to_data_analysis_folder = path_to_data_analysis_folder
    
    def set_conditions_frames_dictionary(self, condition_frames_dictionary:dict):
        self.condition_frames_dictionary = condition_frames_dictionary

    def run_COM_analysis(self):
        self.total_body_COM_data = self.load_COM_data(self.path_to_session_folder)
        self.path_length_calculator = PathLengthCalculator.PathLengthCalculator(self.total_body_COM_data)
        path_length_dictionary = self.calculate_path_lengths(self.condition_frames_dictionary)
        # self.save_condition_path_lengths(path_length_dictionary, self.path_to_data_analysis_folder)
        self.path_length_results.setText(str(path_length_dictionary))

    def load_COM_data(self, path_to_session_folder:Path):
        loaded_freemocap_data = FreeMoCapDataLoader(path_to_session_folder)
        total_body_COM_data = loaded_freemocap_data.load_total_body_COM_data()

        return total_body_COM_data

    def calculate_path_lengths(self, condition_dictionary):
        path_length_dictionary = {}
        for condition in condition_dictionary:
            this_condition_frame_range = range(condition_dictionary[condition][0],condition_dictionary[condition][1])
            path_length_dictionary[condition] = self.path_length_calculator.get_path_length(this_condition_frame_range)

        return path_length_dictionary

    def save_condition_path_lengths(self,path_length_dictionary,path_to_data_analysis_folder:Path):
        json_file_name = path_to_data_analysis_folder/'condition_path_lengths.json'
        out_file = open(json_file_name,'w')

        json.dump(path_length_dictionary, out_file)

            

    
