import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import matplotlib.ticker as mticker
from pathlib import Path
from datetime import datetime

class SkeletonAnimation():
    def __init__(self,freemocap_marker_data_array:np.ndarray,path_to_save_video:Path,saved_video_file_name:str):
        self.freemocap_marker_data_array = freemocap_marker_data_array
        self.path_to_save_video = path_to_save_video
        self.saved_video_file_name = saved_video_file_name
        self.num_frames_to_plot = freemocap_marker_data_array.shape[0]
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

        ax.scatter(self.freemocap_marker_data_array[frame,:,0], self.freemocap_marker_data_array[frame,:,1], self.freemocap_marker_data_array[frame,:,2], c='r', marker='o')
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

    freemocap_data_folder_path = Path(r'D:\freemocap2022\FreeMocap_Data')
    #sessionID = 'sesh_2022-05-24_15_55_40_JSM_T1_BOS' #name of the sessionID folder
    sessionID = 'sesh_2022-09-19_13_56_34'
    data_array_path = freemocap_data_folder_path/sessionID/'DataArrays'
    freemocap_marker_data_array = np.load(data_array_path/'mediaPipeSkel_3d_origin_aligned.npy')

    skeleton_scatter_plot_vid = SkeletonAnimation(freemocap_marker_data_array[:,0:33,:],freemocap_data_folder_path/sessionID,'skeleton_3d_scatter.mp4')
    skeleton_scatter_plot_vid.run()