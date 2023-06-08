import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path 
from freemocap_utils.mediapipe_skeleton_builder import mediapipe_indices
import seaborn as sns

class FileManager:
    def __init__(self, path_to_session_folder, path_to_data_analysis):
        self.path_to_session_folder = path_to_session_folder
        self.path_to_data_analysis = path_to_data_analysis

        self.output_data_folder = 'output_data'
        self.marker_data_array_name = 'mediapipe_body_3d_xyz.npy'
        self.com_data_array_name = 'total_body_center_of_mass_xyz.npy'

    def load_skeleton_data(self, path_to_session_folder):
        skeleton_data_folder_path = path_to_session_folder / self.output_data_folder  / self.marker_data_array_name
        return np.load(skeleton_data_folder_path)
    
    def load_com_data(self,path_to_session_folder):
        com_folder_path = path_to_session_folder/self.output_data_folder/'center_of_mass'/self.com_data_array_name
        return np.load(com_folder_path)
    
    def load_condition_slices(self, path_to_data_analysis):

        self.marker_data_3d = self.load_skeleton_data(path_to_session_folder=self.path_to_session_folder)
        self.com_data_3d = self.load_com_data(path_to_session_folder=self.path_to_session_folder)

        condition_json_path = path_to_data_analysis/'condition_data.json'
        condition_slices = json.load(open(condition_json_path))

        conditions_names = list(condition_slices['Frame Intervals'].keys())
        conditions_slices = list(condition_slices['Frame Intervals'].values())

        marker_data_sliced = {}
        for condition_name, condition_slice in zip(conditions_names, conditions_slices):
            marker_data_sliced[condition_name] = self.marker_data_3d[condition_slice[0]:condition_slice[1]]

        com_data_sliced = {}
        for condition_name, condition_slice in zip(conditions_names, conditions_slices):
            com_data_sliced[condition_name] = self.com_data_3d[condition_slice[0]:condition_slice[1]]

        return marker_data_sliced,com_data_sliced
        f = 2
    
    def save_figure(self, figure, filename):
        figure.savefig(self.path_to_data_analysis / filename)

class COM_2D_Plotter:
    def __init__(self, path_to_session_folder, path_to_data_analysis):
        self.path_to_session_folder = path_to_session_folder

        self.file_manager = FileManager(path_to_session_folder=path_to_session_folder, path_to_data_analysis=path_to_data_analysis)

        self.condition_slices = self.file_manager.load_condition_slices(path_to_data_analysis)
        self.marker_data_sliced, self.com_data_sliced = self.file_manager.load_condition_slices(path_to_data_analysis)

    def plot_COM_1D_histograms(self, ax_range=100):
        fig, axs = plt.subplots(2, 4, figsize=(20, 10), sharex=False, sharey='row',
                                gridspec_kw={'hspace': 0.5, 'wspace': 0.3})
        
        conditions = list(self.com_data_sliced.keys())
        num_bins = 100

        for i, condition in enumerate(conditions):
            # Get the COM data for this condition
            com_data = self.com_data_sliced[condition]

            # 1D histograms
            x_data = com_data[:, 0]
            y_data = com_data[:, 1]
            
            # Calculate mean values
            com_mean_x = np.mean(x_data)
            com_mean_y = np.mean(y_data)
            
            # Plot histogram for x direction
            sns.histplot(x_data, ax=axs[0, i], bins=np.linspace(com_mean_x - ax_range, com_mean_x + ax_range, num_bins), color='blue')
            axs[0, i].set_title(condition)
            axs[0, i].set_xlabel('X-Axis (mm)')
            axs[0, i].set_ylabel('Frequency')
            # axs[0, i].set_xlim(40, 200)
            
            # Plot histogram for y direction
            sns.histplot(y_data, ax=axs[1, i], bins=np.linspace(com_mean_y - ax_range, com_mean_y + ax_range, num_bins), color='red')
            axs[1, i].set_xlabel('Y-Axis (mm)')
            axs[1, i].set_ylabel('Frequency')
            axs[1, i].set_xlim(550, 700)

            fig.suptitle('COM X and Y Positions')


        plt.show()
        # self.file_manager.save_figure(fig, "com_dispersion_plots.png")


if __name__ == "__main__":

    path_to_session_folder = Path(r'D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_14_53_48_MDN_NIH_Trial3')
    path_to_data_analysis = Path(r'D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_14_53_48_MDN_NIH_Trial3\data_analysis\analysis_2023-06-01_10_12_24')
    
    com_2d_plotter = COM_2D_Plotter(path_to_session_folder=path_to_session_folder, path_to_data_analysis=path_to_data_analysis)
    com_2d_plotter.plot_COM_1D_histograms()




