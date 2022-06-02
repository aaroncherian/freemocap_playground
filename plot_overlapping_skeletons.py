from distutils.log import debug
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path 
import socket
import pickle
from rich.progress import track
import cv2
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import matplotlib.ticker as mticker
import sys 
from datetime import datetime

from mediapipe_skeleton_builder import mediapipe_indices, build_mediapipe_skeleton, slice_mediapipe_data
from skeleton_data_holder import SkeletonDataHolder
import scipy.io as sio


this_computer_name = socket.gethostname()

if this_computer_name == 'DESKTOP-V3D343U':
    freemocap_data_path = Path(r"I:\My Drive\HuMoN_Research_Lab\FreeMoCap_Stuff\FreeMoCap_Balance_Validation\data")
elif this_computer_name == 'DESKTOP-F5LCT4Q':
    #freemocap_validation_data_path = Path(r"C:\Users\aaron\Documents\HumonLab\Spring2022\ValidationStudy\FreeMocap_Data")
    #freemocap_validation_data_path = Path(r'D:\freemocap2022\FreeMocap_Data')
    freemocap_data_path = Path(r'D:\ValidationStudy2022\FreeMocap_Data')
else:
    #freemocap_validation_data_path = Path(r"C:\Users\kiley\Documents\HumonLab\SampleFMC_Data\FreeMocap_Data-20220216T173514Z-001\FreeMocap_Data")
    freemocap_data_path = Path(r"C:\Users\Rontc\Documents\HumonLab\ValidationStudy")


class OverlappingSkeletonPlotter:
    def __init__(self, sessionID, mediapipe_data, qualisys_data, mediapipe_num_frame_range, qualisys_num_frame_range):
        self.sessionID = sessionID
        self.mediapipe_data = mediapipe_data
        self.qualisys_data = qualisys_data
        self.mediapipe_num_frame_range = mediapipe_num_frame_range
        self.qualisys_num_frame_range = qualisys_num_frame_range

    def get_session_path(self,sessionID):
        this_freemocap_session_path = freemocap_data_path/sessionID
        return this_freemocap_session_path

    def slice_data_to_plot(self, mediapipe_num_frame_range, qualisys_num_frame_range):

        mediapipe_data_to_plot = self.mediapipe_data[mediapipe_num_frame_range[0]:mediapipe_num_frame_range[-1],:,:]
        qualisys_data_to_plot = self.qualisys_data[qualisys_num_frame_range[0]:qualisys_num_frame_range[-1]:5,:,:]

        num_frames_to_plot = range(self.mediapipe_num_frame_range[-1] - self.mediapipe_num_frame_range[0])

        return mediapipe_data_to_plot, qualisys_data_to_plot, num_frames_to_plot
    def create_figure(self):

        figure = plt.figure(figsize = (5,5))
        ax1 = figure.add_subplot(111, projection = '3d')    
        self.ax_range = 900
        self.get_axes_means(self.mediapipe_data_to_plot)
        return figure, ax1

    def label_axes(self,plot_ax):
        plot_ax.set_xlabel('X')
        plot_ax.set_ylabel('Y')
        plot_ax.set_zlabel('Z')
    def set_axes_ranges(self, plot_ax, ax_range):
    
        plot_ax.set_xlim(self.mx-ax_range,self.mx+ax_range)
        plot_ax.set_ylim(self.my-ax_range,self.my+ax_range)
        plot_ax.set_zlim(self.mz-ax_range,self.mz+ax_range)

    def get_axes_means(self,skeleton_data):
        
        num_frame_length = len(self.num_frames_to_plot)

        self.mx = -1*np.nanmean(skeleton_data[int(num_frame_length/2),:,0])
        self.my = -1*np.nanmean(skeleton_data[int(num_frame_length/2),:,1])
        self.mz = np.nanmean(skeleton_data[int(num_frame_length/2),:,2])


    def create_plot(self):

        output_video_fps = 30
        print('Starting Frame Animation') 
        ani = FuncAnimation(self.figure, self.animate, frames = self.num_frames_to_plot, interval=.1, repeat=False, init_func= self.animation_init)
        writervideo = animation.FFMpegWriter(fps= output_video_fps)
        ani.save(self.session_path/'overlapping_skeletons.mp4', writer=writervideo, dpi = 300)
        print('Animation has been saved to {}'.format(self.session_path))


    def animation_init(self):
        #the FuncAnimation needs an initial function that it will run, otherwise it will run animate() twice for frame 0 
        pass
    
    def animate(self,frame):
        ax1 = self.ax1 

        ax1.cla()
        if frame % 100 == 0:
            now = datetime.now()

            current_time = now.strftime("%H:%M:%S")
            print("Currently on frame: {} at {}".format(frame,current_time))

        ax1.scatter(-1*self.mediapipe_data_to_plot[frame,:,0], -1*self.mediapipe_data_to_plot[frame,:,1], self.mediapipe_data_to_plot[frame,:,2], c='r', marker='o', label = 'GoPro/Mediapipe')
        ax1.scatter(-1*self.qualisys_data_to_plot[frame,:,0], -1*self.qualisys_data_to_plot[frame,:,1], self.qualisys_data_to_plot[frame,:,2], c='b', marker='o', label = 'Qualisys')
        ax1.view_init(elev = 0, azim =-70)
        self.set_axes_ranges(ax1, self.ax_range)
        self.label_axes(ax1)     
        ax1.legend()

        ax1.xaxis.set_major_locator(mticker.MultipleLocator(400))
        ax1.xaxis.set_minor_locator(mticker.MultipleLocator(200))

        ax1.yaxis.set_major_locator(mticker.MultipleLocator(400))
        ax1.yaxis.set_minor_locator(mticker.MultipleLocator(200))

        ax1.zaxis.set_major_locator(mticker.MultipleLocator(400))
        ax1.zaxis.set_minor_locator(mticker.MultipleLocator(200))

    def run(self):
        self.session_path = self.get_session_path(self.sessionID)
        self.mediapipe_data_to_plot, self.qualisys_data_to_plot, self.num_frames_to_plot = self.slice_data_to_plot(self.mediapipe_num_frame_range, self.qualisys_num_frame_range)
        self.figure, self.ax1 = self.create_figure()
        self.create_plot()

        f = 2
sessionID = 'session_SER_1_20_22' #name of the sessionID folder

qualisys_data_array_name = 'skeleton_fr_mar_dim_rotated.mat'

mediapipe_data_array_name = 'mediapipe_skel_data_aligned_to_qualisys.npy'

this_freemocap_session_path = freemocap_data_path / sessionID
this_freemocap_data_path = this_freemocap_session_path/'DataArrays'

qualisys_data_path = this_freemocap_data_path/qualisys_data_array_name
mediapipe_data_path = this_freemocap_data_path/mediapipe_data_array_name

qualysis_mat_file = sio.loadmat(qualisys_data_path)
qualisys_pose_data = qualysis_mat_file['skeleton_fr_mar_dim_rotated']

num_mediapipe_pose_joints = 33
mediapipeSkel_fr_mar_dim = np.load(mediapipe_data_path)
mediapipe_pose_data = slice_mediapipe_data(mediapipeSkel_fr_mar_dim, num_mediapipe_pose_joints)

#mediapipe_num_frame_range = range(9500,12000)
mediapipe_num_frame_range = range(13000,15180)
#qualisys_num_frame_range = range(58355,70855,5)
qualisys_num_frame_range = range(75855,86755,5)

assert len(qualisys_num_frame_range) == len(mediapipe_num_frame_range), "The number of frames in the two data arrays must be the same. Num qualisys frames is {} and num mediapipe frames is {}".format(len(qualisys_num_frame_range), len(mediapipe_num_frame_range))

overlapping_skeleton_plotter = OverlappingSkeletonPlotter(sessionID,mediapipe_pose_data, qualisys_pose_data, mediapipe_num_frame_range, qualisys_num_frame_range)
overlapping_skeleton_plotter.run()

f = 2

