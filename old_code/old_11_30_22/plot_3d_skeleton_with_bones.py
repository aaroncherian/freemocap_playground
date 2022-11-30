import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import matplotlib.ticker as mticker
from pathlib import Path
from datetime import datetime

class SkeletonAnimation():
    def __init__(self,freemocap_marker_data_array:np.ndarray,skeleton_dict_with_connections,path_to_save_video:Path,saved_video_file_name:str):
        self.freemocap_marker_data_array = freemocap_marker_data_array
        self.path_to_save_video = path_to_save_video
        self.saved_video_file_name = saved_video_file_name
        self.num_frames_to_plot = freemocap_marker_data_array.shape[0]
        self.skeleton_dict_with_connections = skeleton_dict_with_connections
        self.azimuth = -70

    def __create_figure(self, freemocap_marker_data_array):

        figure = plt.figure(figsize = (5,5))
        ax = figure.add_subplot(111, projection = '3d')    
        self.ax_range = 900
        self.mx,self.my,self.mz = self.__get_axes_means(freemocap_marker_data_array)
        return figure, ax

    def __get_axes_means(self,skeleton_data):
        

        mx = np.nanmean(skeleton_data[int(self.num_frames_to_plot/2),:,0])
        my = np.nanmean(skeleton_data[int(self.num_frames_to_plot/2),:,1])
        mz = np.nanmean(skeleton_data[int(self.num_frames_to_plot/2),:,2])
        
        return mx,my,mz
    
    def __set_axes_ranges(self, plot_ax, ax_range):
        plot_ax.set_xlim(self.mx-ax_range,self.mx+ax_range)
        plot_ax.set_ylim(self.my-ax_range,self.my+ax_range)
        plot_ax.set_zlim(self.mz-ax_range,self.mz+ax_range)

    def __label_axes(self,plot_ax):
        plot_ax.set_xlabel('X')
        plot_ax.set_ylabel('Y')
        plot_ax.set_zlabel('Z')

    def __create_plot(self):
        output_video_fps = 30
        print('Starting Frame Animation') 
        ani = FuncAnimation(self.figure, self.__animate, frames = self.num_frames_to_plot, interval=.1, repeat=False, init_func= self.__animation_init)
        writervideo = animation.FFMpegWriter(fps= output_video_fps)
        ani.save(self.path_to_save_video/self.saved_video_file_name, writer=writervideo, dpi = 300)
        print('Animation has been saved to {}'.format(self.path_to_save_video))

    
    
    def __plot_skeleton_bones(self,ax,skeleton_connection_data,frame):
        this_frame_skeleton_data = skeleton_connection_data[frame]
        for segment in this_frame_skeleton_data.keys():
            prox_joint = this_frame_skeleton_data[segment][0] 
            dist_joint = this_frame_skeleton_data[segment][1]
            
            bone_x,bone_y,bone_z = [prox_joint[0],dist_joint[0]],[prox_joint[1],dist_joint[1]],[prox_joint[2],dist_joint[2]] 

            ax.plot(bone_x,bone_y,bone_z,color = 'b')

    def __create_segment_connection(self,skeleton_connection_data,frame,segment):
        left_segment_name = 'left_' + segment
        right_segment_name = 'right_' + segment

        left_joint = skeleton_connection_data[frame][left_segment_name][0]
        right_joint = skeleton_connection_data[frame][right_segment_name][0]
        segment_connection_x, segment_connection_y, segment_connection_z = [[left_joint[0],right_joint[0]],[left_joint[1],right_joint[1]],[left_joint[2],right_joint[2]]]

        return segment_connection_x, segment_connection_y, segment_connection_z

    def __animation_init(self):
        #the FuncAnimation needs an initial function that it will run, otherwise it will run animate() twice for frame 0 
        pass

    def __animate(self,frame):
        ax = self.ax 
        ax.cla()

        if frame % 100 == 0:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("Currently on frame: {} at {}".format(frame,current_time))

        hip_connection_x,hip_connection_y,hip_connection_z = self.__create_segment_connection(self.skeleton_dict_with_connections,frame,'thigh')
        shoulder_connection_x,shoulder_connection_y,shoulder_connection_z = self.__create_segment_connection(self.skeleton_dict_with_connections,frame,'upper_arm')
        

        ax.scatter(self.freemocap_marker_data_array[frame,:,0], self.freemocap_marker_data_array[frame,:,1], self.freemocap_marker_data_array[frame,:,2], c='r', marker='o')
        self.__plot_skeleton_bones(ax,self.skeleton_dict_with_connections,frame)
        ax.plot(hip_connection_x,hip_connection_y,hip_connection_z,color = 'b')
        ax.plot(shoulder_connection_x,shoulder_connection_y,shoulder_connection_z, color = 'b')
 
        self.azimuth = self.azimuth + .25
    
        ax.view_init(elev = 30, azim = self.azimuth)
        self.__set_axes_ranges(ax, self.ax_range)
        self.__label_axes(ax)     
        #ax.legend()

        ax.xaxis.set_major_locator(mticker.MultipleLocator(400))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(200))

        ax.yaxis.set_major_locator(mticker.MultipleLocator(400))
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(200))

        ax.zaxis.set_major_locator(mticker.MultipleLocator(400))
        ax.zaxis.set_minor_locator(mticker.MultipleLocator(200))


    def run(self):
        self.figure, self.ax = self.__create_figure(self.freemocap_marker_data_array)
        self.__create_plot()
    




if __name__ == '__main__':

    from pathlib import Path
    from fmc_core_toolbox.anthropometry_data_tables import segments, joint_connections, segment_COM_lengths, segment_COM_percentages, build_anthropometric_dataframe
    from fmc_core_toolbox.mediapipe_skeleton_builder import mediapipe_indices,build_mediapipe_skeleton
    from fmc_core_toolbox.good_frame_finder import find_good_frame
    from fmc_core_toolbox.skeleton_origin_alignment import align_skeleton_with_origin
    
    freemocap_data_folder_path = Path(r'D:\freemocap2022\FreeMocap_Data')
    #freemocap_data_folder_path = Path(r'D:\ValidationStudy2022\FreeMocap_Data')

    sessionID = 'sesh_2022-09-29_17_29_31'
    data_array_path = freemocap_data_folder_path/sessionID/'DataArrays'
    freemocap_marker_data_array = np.load(data_array_path/'mediaPipeSkel_3d_origin_aligned.npy')
    #freemocap_marker_data_array = np.load(data_array_path/'mediaPipeSkel_3d.npy')
    
    freemocap_body_marker_data_array = freemocap_marker_data_array[:,0:33,:]

    good_frame = find_good_frame(freemocap_body_marker_data_array,mediapipe_indices,.3)
    freemocap_alignment_marker_data_tuple = align_skeleton_with_origin(freemocap_body_marker_data_array, mediapipe_indices, good_frame)
    origin_aligned_freemocap_marker_data = freemocap_alignment_marker_data_tuple[0]

    anthro_df = build_anthropometric_dataframe(segments,joint_connections,segment_COM_lengths,segment_COM_percentages)
    skelcoordinates_frame_segment_joint_XYZ  = build_mediapipe_skeleton(origin_aligned_freemocap_marker_data,anthro_df,mediapipe_indices)

    skeleton_scatter_plot_vid = SkeletonAnimation(origin_aligned_freemocap_marker_data,skelcoordinates_frame_segment_joint_XYZ,freemocap_data_folder_path/sessionID,'skeleton_3d_scatter.mp4')
    skeleton_scatter_plot_vid.run()

    

    # freemocap_body_marker_data_array = freemocap_marker_data_array[:,0:33,:]
    # anthro_df = build_anthropometric_dataframe(segments,joint_connections,segment_COM_lengths,segment_COM_percentages)
    # skelcoordinates_frame_segment_joint_XYZ  = build_mediapipe_skeleton(freemocap_body_marker_data_array,anthro_df,mediapipe_indices)

    # skeleton_scatter_plot_vid = SkeletonAnimation(freemocap_body_marker_data_array,skelcoordinates_frame_segment_joint_XYZ,freemocap_data_folder_path/sessionID,'og_skeleton_3d_scatter.mp4')
    # skeleton_scatter_plot_vid.run()