

from PyQt6.QtWidgets import QWidget,QVBoxLayout, QLineEdit, QPushButton ,QFormLayout, QLabel

from pathlib import Path
import json

import datetime

import pandas as pd

class SavingDataAnalysisWidget(QWidget):
    def __init__(self):
        super().__init__()

        self._layout = QVBoxLayout()
        self.setLayout(self._layout)


        self.saved_folder_name_entry = QLineEdit()
        self.saved_folder_name_entry.setMaximumWidth(200)
        self.saved_folder_name_entry.setText(datetime.datetime.now().strftime("analysis_%Y-%m-%d_%H_%M_%S"))
        
        saved_folder_name_form = QFormLayout()
        saved_folder_name_form.addRow(QLabel('ID for data analysis folder'), self.saved_folder_name_entry)
        self._layout.addLayout(saved_folder_name_form)  



        self.save_data_button = QPushButton('Save out data analysis results')
        self.save_data_button.clicked.connect(self.save_data_out)
        self.save_data_button.setEnabled(False)
        self._layout.addWidget(self.save_data_button)

    def set_session_folder_path(self, session_folder_path):
        self.session_folder_path = session_folder_path

    def set_conditions_frames_dictionary(self, conditions_dict:dict):
        self.condition_frame_intervals_dictionary = conditions_dict

    def set_conditions_path_length_dictionary(self, conditions_path_length:dict):
        self.conditions_path_length_dictionary = conditions_path_length

    def set_histogram_figure(self,histogram_figure):
        self.histogram_figure = histogram_figure

    def set_velocity_dictionary(self, velocity_dictionary):
        self.velocity_dictionary = velocity_dictionary

    def format_data_json(self,condition_frame_intervals_dictionary, path_length_dictionary):
        dict_to_save = {'Frame Intervals':condition_frame_intervals_dictionary, 'Path Lengths:': path_length_dictionary}
        return dict_to_save

    def save_data_out(self):
        saved_folder_name = self.saved_folder_name_entry.text()
        self.saved_data_analysis_path = self.create_folder_to_save_data(saved_folder_name)
        formatted_condition_data_dict = self.format_data_json(self.condition_frame_intervals_dictionary,self.conditions_path_length_dictionary)
        self.save_data_json(formatted_condition_data_dict, 'condition_data.json', self.saved_data_analysis_path)
        self.save_velocity_dict_as_csv(self.velocity_dictionary,self.saved_data_analysis_path)
        self.save_plot(self.histogram_figure,self.saved_data_analysis_path)


    # def save_conditions_dict(self, conditions_dictionary:dict):
    #     saved_folder_name = self.saved_folder_name_entry.text()
    #     self.saved_data_analysis_path = self.create_folder_to_save_data(saved_folder_name)

    #     self.create_frame_interval_json(conditions_dictionary,self.saved_data_analysis_path)


    def create_folder_to_save_data(self, saved_folder_name:str):

        saved_data_analysis_path = self.session_folder_path/'data_analysis'/saved_folder_name
        saved_data_analysis_path.mkdir(parents = True, exist_ok=True)

        return saved_data_analysis_path


    def save_data_json(self,dict_for_json:dict, json_name:str, save_folder_path:Path):
        json_file_name = save_folder_path/json_name
        out_file = open(json_file_name,'w')

        json.dump(dict_for_json,out_file, indent=1)


    def save_velocity_dict_as_csv(self, velocity_dict:dict, save_folder_path:Path):
        
        for count,dimension in enumerate(['x','y', 'z']):
            this_dimension_array_list = [item[count] for item in velocity_dict.values()] #grab the X dimension of each condition to plot 
            conditions_list = list(velocity_dict.keys())
            
            this_dimension_velocity_dict = dict(zip(conditions_list,this_dimension_array_list))
            velocity_dataframe = pd.DataFrame({ key:pd.Series(value) for key, value in this_dimension_velocity_dict.items() })
            velocity_dataframe.to_csv(save_folder_path/f'condition_velocities_{dimension}.csv')

        f = 2

    def save_plot(self,figure,save_folder_path:Path):
        figure.savefig(str(save_folder_path/'velocity_histogram.png'))


