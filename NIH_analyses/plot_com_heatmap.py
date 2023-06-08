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

    def plot_COM_2D(self, ax_range=200):
        fig, axs = plt.subplots(1, 4, figsize=(20, 5), sharex='col', sharey='row',
                                gridspec_kw={'hspace': 0, 'wspace': 0})

        conditions = list(self.com_data_sliced.keys())
        num_bins = 75

        # Mediapipe marker indices for heels and foot_index
        left_heel_idx = 29
        right_heel_idx = 30
        left_foot_index_idx = 31
        right_foot_index_idx = 32
        
        for ax, condition in zip(axs, conditions):
            # Get the COM data for this condition
            com_data = self.com_data_sliced[condition]
            marker_data = self.marker_data_sliced[condition]

            # Create 2D histogram
            com_mean_x = np.mean(com_data[:, 0])
            com_mean_y = np.mean(com_data[:, 1])
            H, xedges, yedges = np.histogram2d(com_data[:, 0], com_data[:, 1], bins=[np.linspace(com_mean_x - ax_range, com_mean_x + ax_range, num_bins), np.linspace(com_mean_y - ax_range, com_mean_y + ax_range, num_bins)])
            
            # Draw heatmap using seaborn
            sns.heatmap(H, ax=ax, cmap='hot', cbar=False)

            # Plot feet markers with lines between them
            left_heel = np.mean(marker_data[:, left_heel_idx, :2], axis=0)
            right_heel = np.mean(marker_data[:, right_heel_idx, :2], axis=0)
            left_foot_index = np.mean(marker_data[:, left_foot_index_idx, :2], axis=0)
            right_foot_index = np.mean(marker_data[:, right_foot_index_idx, :2], axis=0)

            def coords_to_index(coords, xedges, yedges):
                x_idx = np.digitize(coords[0], xedges) - 1
                y_idx = np.digitize(coords[1], yedges) - 1
                return x_idx, y_idx

            ax.plot(*zip(coords_to_index(left_heel, xedges, yedges), coords_to_index(left_foot_index, xedges, yedges)), color='blue')
            ax.plot(*zip(coords_to_index(right_heel, xedges, yedges), coords_to_index(right_foot_index, xedges, yedges)), color='red')

            # Set title and labels
            ax.set_title(condition)
            ax.set_xlabel('X-Axis (mm)')
            ax.set_ylabel('Y-Axis (mm)')
            
            # Set tick labels to represent actual data range
            ax.set_xticks([0, num_bins-1])
            ax.set_yticks([0, num_bins-1])
            ax.set_xticklabels([int(com_mean_x - ax_range), int(com_mean_x + ax_range)])
            ax.set_yticklabels([int(com_mean_y - ax_range), int(com_mean_y + ax_range)])


        plt.show()
        # self.file_manager.save_figure(fig, "com_dispersion_plots.png")


if __name__ == "__main__":

    path_to_session_folder = Path(r'D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_14_53_48_MDN_NIH_Trial3')
    path_to_data_analysis = Path(r'D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_14_53_48_MDN_NIH_Trial3\data_analysis\analysis_2023-06-01_10_12_24')
    
    com_2d_plotter = COM_2D_Plotter(path_to_session_folder=path_to_session_folder, path_to_data_analysis=path_to_data_analysis)
    com_2d_plotter.plot_COM_2D()




