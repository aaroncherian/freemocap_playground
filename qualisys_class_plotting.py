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
import scipy.io as sio

from io import BytesIO


#you can skip every 10th frame 
class animateSkeleton:
    def __init__(self, freemocap_validation_data_path, sessionID, num_frame_range, step_interval, camera_fps, output_video_fps, tail_length):
        self.num_frame_range = num_frame_range

        self.sessionID = sessionID

        self.validation_data_path = freemocap_validation_data_path

        self.step_interval = step_interval

        
        self.camera_fps = camera_fps 

        self.output_video_fps = output_video_fps

        self.tail_length = tail_length

    def create_paths_to_data_files(self, validation_data_path, sessionID):
        
        this_freemocap_session_path = validation_data_path / sessionID
        this_freemocap_data_path = this_freemocap_session_path/'DataArrays'

        totalCOM_data_path = this_freemocap_data_path / 'qualisys_totalBodyCOM_frame_XYZ.npy'
        segmentedCOM_data_path = this_freemocap_data_path / 'qualisys_segmentedCOM_frame_joint_XYZ.npy'
        #mediapipe_data_path = this_freemocap_data_path/'mediaPipeSkel_3d_smoothed.npy'
        mediapipe_data_path = this_freemocap_data_path/'qualisysData_3d.mat'
        mediapipeSkeleton_file_name = this_freemocap_data_path/'qualisysSkelcoordinates_frame_segment_joint_XYZ.pkl'

        syncedVideoName = sessionID + '_Cam1_synced.mp4'

        syncedVideoPath = this_freemocap_session_path/'SyncedVideos'/syncedVideoName

        self.this_freemocap_session_path = this_freemocap_session_path #needed when saving out the plot video 

        return mediapipe_data_path, mediapipeSkeleton_file_name, totalCOM_data_path, segmentedCOM_data_path, syncedVideoPath

    
    def load_data_from_paths(self,mediapipe_data_path,mediapipeSkeleton_file_name,totalCOM_data_path, segmentedCOM_data_path):

        totalCOM_frame_XYZ = np.load(totalCOM_data_path) #loads in the data as a numpy array

        segmentedCOM_frame_joint_XYZ = np.load(segmentedCOM_data_path)

        qualysis_mat_file = sio.loadmat(mediapipe_data_path)
        mediapipe_pose_data = qualysis_mat_file['skeleton_fr_mar_dim_reorg']

        open_file = open(mediapipeSkeleton_file_name, "rb")
        mediapipeSkelcoordinates_frame_segment_joint_XYZ = pickle.load(open_file)
        open_file.close()

        return mediapipe_pose_data, mediapipeSkelcoordinates_frame_segment_joint_XYZ, segmentedCOM_frame_joint_XYZ, totalCOM_frame_XYZ

    def get_mediapipe_pose_data(self,  mediapipeSkel_fr_mar_dim):

        num_pose_joints = 33 #yes, this is hardcoded. but if mediapipe updates to use a different skeleton we need to update a lot of things anyway 

        mediapipe_pose_data =  mediapipeSkel_fr_mar_dim[:,0:num_pose_joints,:]

        return mediapipe_pose_data

    def slice_data_arrays_by_range(self,num_frame_range,mediapipe_pose_data, mediapipeSkelcoordinates_frame_segment_joint_XYZ, segmentedCOM_frame_joint_XYZ, totalCOM_frame_XYZ):

            
        start_frame = num_frame_range[0]
        end_frame = num_frame_range[-1]
        
        
        this_range_mp_pose_XYZ = mediapipe_pose_data[start_frame:end_frame:self.step_interval,:,:]

        this_range_mp_skeleton_segment_XYZ = mediapipeSkelcoordinates_frame_segment_joint_XYZ[start_frame:end_frame:self.step_interval]

        this_range_segmentCOM_fr_joint_XYZ = segmentedCOM_frame_joint_XYZ[start_frame:end_frame:self.step_interval,:,:]

        this_range_totalCOM_frame_XYZ = totalCOM_frame_XYZ[start_frame:end_frame:self.step_interval,:]

        return this_range_mp_pose_XYZ,this_range_mp_skeleton_segment_XYZ,this_range_segmentCOM_fr_joint_XYZ,this_range_totalCOM_frame_XYZ

    def load_video_capture_object(self,syncedVideoPath):
        
        cap = cv2.VideoCapture(str(syncedVideoPath))

        return cap

    def get_video_frames_to_plot(self,cap,num_frame_range):

        video_frames_to_plot = []

        # for frame in track(num_frame_range):
        #         cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        #         success, image = cap.read()
        #         video_frames_to_plot.append(image)
        # cap.release()
        start_frame = num_frame_range[0]
        end_frame = num_frame_range[-1]
        current_frame = start_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        while current_frame < end_frame:
            success, image = cap.read()
            video_frames_to_plot.append(image)
            current_frame += 1

        # for frame in track(num_frame_range):
        #         cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        #         success, image = cap.read()
        #         video_frames_to_plot.append(image)
        cap.release()
        #print('finished getting video frames')
        return video_frames_to_plot

    def animation_init(self):
        #the FuncAnimation needs an initial function that it will run, otherwise it will run animate() twice for frame 0 
        pass  
    # def animate(self,frame,video_frames_to_plot):

    #     video_frames_to_plot = []
    #     ax = self.ax #NOTE - will redefining self.ax as ax at the start of this function have a screwy effect down the line?
    #     ax2 = self.ax2
    #     # ax3 = self.ax3
    #     ax4 = self.ax4
    #     ax5 = self.ax5
    #     # for frame in track(num_frame_range):
    #     #         cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    #     #         success, image = cap.read()
    #     #         video_frames_to_plot.append(image)
    #     # cap.release()
    #     # start_frame = num_frame_range[0]
    #     # end_frame = num_frame_range[-1]
    #     # current_frame = start_frame
    #     # cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    #     # while current_frame < end_frame:
    #     #     success, image = cap.read()
    #     #     video_frames_to_plot.append(image)
    #     #     current_frame += 1

    #     # for frame in track(num_frame_range):
    #     #         cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    #     #         success, image = cap.read()
    #     #         video_frames_to_plot.append(image)
    #     cap.release()
    #     #print('finished getting video frames')
    #     return video_frames_to_plot

   
    def animate(self,frame,video_frames_to_plot):

        ax = self.ax #NOTE - will redefining self.ax as ax at the start of this function have a screwy effect down the line?
        ax2 = self.ax2
        #ax3 = self.ax3
        ax4 = self.ax4
        ax5 = self.ax5

        if frame % 100 == 0:
            now = datetime.now()

            current_time = now.strftime("%H:%M:%S")
            print("Currently on frame: {} at {}".format(frame,current_time))

        # now = datetime.now()

        # current_time = now.strftime("%H:%M:%S")
        # print("Currently on frame: {} at {}".format(frame,current_time))

        this_frame_skel_x = self.skel_x[frame,:]
        this_frame_skel_y = self.skel_y[frame,:]
        this_frame_skel_z = self.skel_z[frame,:]

        left_heel_x = this_frame_skel_x[18]
        left_heel_y = this_frame_skel_y[18]        
        left_heel_z = this_frame_skel_y[18]

        left_toe_x = this_frame_skel_x[19]
        left_toe_y = this_frame_skel_y[19]
        left_toe_z = this_frame_skel_y[19]

        right_heel_x = this_frame_skel_x[22]
        right_heel_y = this_frame_skel_y[22]
        right_heel_z = this_frame_skel_y[22]

        right_toe_x = this_frame_skel_x[23]
        right_toe_y = this_frame_skel_y[23]
        right_toe_z = this_frame_skel_y[23]

        left_foot_x,left_foot_y, left_foot_z = [left_heel_x,left_toe_x], [left_heel_y,left_toe_y], [left_heel_z,left_toe_z]
        right_foot_x,right_foot_y, right_foot_z = [right_heel_x,right_toe_x], [right_heel_y,right_toe_y], [right_heel_z,right_toe_z]

        left_foot_average_position_x = (left_heel_x + left_toe_x)/2
        right_foot_average_position_x = (right_heel_x + right_toe_x)/2
        
        left_foot_average_position_y = (left_heel_y + left_toe_y)/2
        right_foot_average_position_y = (right_heel_y + right_toe_y)/2

        back_foot_average_position_x = (left_heel_x + right_heel_x)/2
        front_foot_average_position_x = (left_toe_x + right_toe_x)/2

        back_foot_average_position_y = (left_heel_y + right_heel_y)/2
        front_foot_average_position_y = (left_toe_y + right_toe_y)/2

        self.medial_bound_max_array.append(left_foot_average_position_x)
        self.medial_bound_min_array.append(right_foot_average_position_x)

        self.anterior_bound_array.append(front_foot_average_position_y)
        self.posterior_bound_array.append(back_foot_average_position_y)
        

        #Get the mediapipe segment COM data for this frame
        this_frame_segment_COM_x = self.this_range_segmentCOM_fr_joint_XYZ[frame,:,0]
        this_frame_segment_COM_y = self.this_range_segmentCOM_fr_joint_XYZ[frame,:,1]
        this_frame_segment_COM_z = self.this_range_segmentCOM_fr_joint_XYZ[frame,:,2]


        this_frame_total_COM_x = self.this_range_totalCOM_frame_XYZ[frame,0]
        this_frame_total_COM_y = self.this_range_totalCOM_frame_XYZ[frame,1]
        this_frame_total_COM_z = self.this_range_totalCOM_frame_XYZ[frame,2]

        plot_frame_bones_XYZ = self.this_range_mp_skeleton_segment_XYZ[frame]
        ax.cla()
        ax2.cla()
        ax4.cla()
        ax5.cla()
        
        ax.set_title('Frame# {}'.format(str(self.num_frame_range[frame])),pad = -20, y = 1.)

        ax.set_xlim([self.mx-self.ax_range, self.mx+self.ax_range]) #maybe set ax limits before the function? if we're using cla() they probably don't need to be redefined every time 
        ax.set_ylim([self.my-self.ax_range, self.my+self.ax_range])
        ax.set_zlim([self.mz-self.ax_range, self.mz+self.ax_range])
       
        ax.xaxis.set_major_locator(mticker.MultipleLocator(400))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(200))

        ax.yaxis.set_major_locator(mticker.MultipleLocator(400))
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(200))

        ax.zaxis.set_major_locator(mticker.MultipleLocator(400))
        ax.zaxis.set_minor_locator(mticker.MultipleLocator(200))
        #ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.tick_params(axis='y', labelrotation = -9)
        #ax.yaxis._axinfo['label']['space_factor'] = 20

        ax2.set_title('Total Body Center of Mass Trajectory')
        ax2.set_xlim([self.mx_com-400, self.mx_com+400])
        ax2.set_ylim([self.my_com-400, self.my_com+400])


        ax4.set_title('Max/Min Lateral COM Position vs. Time') 
        ax4.set_ylim([self.mx_com-300, self.mx_com+300])

        ax4.set_xlim(self.time_array[0],self.time_array[-1])
        ax5.set_xlim(self.time_array[0],self.time_array[-1])
  
        ax4.axes.xaxis.set_ticks([])

        ax5.set_title('Anterior/Posterior COM Position vs. Time')
        ax5.set_ylim([self.my_com-250, self.my_com+250])
 
        # ax4.set_ylim([self.mz-200, self.mz+200])
        # ax4.set_xlim([self.num_frame_range[0],self.num_frame_range[-1]])


        #ax.view_init(elev=-70., azim=-60)
        ax.view_init(elev = 0, azim =-70)
        for segment in plot_frame_bones_XYZ.keys():
            if segment == 'head':
                head_point = plot_frame_bones_XYZ[segment][0]
                bone_x,bone_y,bone_z = [head_point[0],head_point[1],head_point[2]]
            else:
                prox_joint = plot_frame_bones_XYZ[segment][0] 
                dist_joint = plot_frame_bones_XYZ[segment][1]
   
                dist_joint = plot_frame_bones_XYZ[segment][1]              
                bone_x,bone_y,bone_z = [prox_joint[0],dist_joint[0]],[prox_joint[1],dist_joint[1]],[prox_joint[2],dist_joint[2]] 

            ax.plot(bone_x,bone_y,bone_z,color = 'black')
            ax2.plot(bone_x,bone_y, color = 'grey', alpha = .4)

        ax.scatter(this_frame_segment_COM_x,this_frame_segment_COM_y,this_frame_segment_COM_z, color = 'orange', label = 'Segment Center of Mass')    
        ax.scatter(this_frame_total_COM_x,this_frame_total_COM_y,this_frame_total_COM_z, color = 'purple', label = 'Total Body Center of Mass', marker = '*')
        ax.scatter(this_frame_skel_x, this_frame_skel_y,this_frame_skel_z, color = 'grey')
            
        #ax.scatter(left_foot_x,left_foot_y,left_foot_z, color = 'blue')
        #ax.scatter(right_foot_x,right_foot_y,right_foot_z, color = 'red')
        ax.legend(bbox_to_anchor=(.5, 0.14))

        plot_fade_frame = frame - self.tail_length
        if plot_fade_frame < 0:
                ax2.plot(self.this_range_totalCOM_frame_XYZ[0:frame,0],self.this_range_totalCOM_frame_XYZ[0:frame,1], color = 'grey', label = 'Center of Mass Path')
        else:
                ax2.plot(self.this_range_totalCOM_frame_XYZ[plot_fade_frame:frame,0],self.this_range_totalCOM_frame_XYZ[plot_fade_frame:frame,1], color = 'grey')
                
        ax2.plot(this_frame_total_COM_x,this_frame_total_COM_y, marker = '*', color = 'purple', ms = 4)

        ax2.plot(left_foot_x,left_foot_y, color = 'blue', label= 'Max Lateral Bound (Left Foot)')
        ax2.plot(right_foot_x,right_foot_y, color = 'red', label = 'Min Lateral Bound (Right Foot)')
        
        ax2.plot(left_foot_average_position_x, left_foot_average_position_y, color = 'blue', ms = 3, marker = 'o')
        ax2.plot(right_foot_average_position_x, right_foot_average_position_y, color = 'red', ms = 3, marker = 'o')
        
        ax2.plot(back_foot_average_position_x,back_foot_average_position_y, color = 'coral', marker = 'o', ms = 3, alpha = .3)
        ax2.plot(front_foot_average_position_x,front_foot_average_position_y, color = 'forestgreen', marker = 'o', ms = 3, alpha = .3)

        ax2.plot([left_heel_x,right_heel_x],[left_heel_y,right_heel_y], color = 'coral', label = 'Posterior Bound', alpha = .3, linestyle = '--')
        ax2.plot([left_toe_x,right_toe_x],[left_toe_y,right_toe_y], color = 'forestgreen', label = 'Anterior Bound', alpha = .3, linestyle = '--')

        ax2.legend(fontsize = 10, bbox_to_anchor=(1.15, -.2),ncol = 2)

        ax4.set_ylabel('X-Axis (mm)')

        ax4.plot(self.time_array[0:frame+1],self.medial_bound_max_array, color = 'blue', alpha = .5)
        ax4.plot(self.time_array[0:frame+1],self.medial_bound_min_array, color = 'red', alpha = .5)
        ax4.plot(self.time_array,self.this_range_totalCOM_frame_XYZ[:,0], color = 'lightgrey')
        ax4.plot(self.time_array[0:frame+1],self.this_range_totalCOM_frame_XYZ[0:frame+1,0], color = 'grey')
        ax4.plot(self.time_array[frame],this_frame_total_COM_x, '*', color = 'purple', ms = 4)
        
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Y-Axis (mm)')
        #ax5.axhline(back_foot_average_position_y, color = 'forestgreen', alpha = .5, label = 'Mean Back Foot Position')
        #ax5.axhline(front_foot_average_position_y, color = 'coral', alpha = .5, label = 'Mean Front Foot Position')
        ax5.plot(self.time_array[0:frame+1],self.anterior_bound_array, color = 'forestgreen', alpha = .5)
        ax5.plot(self.time_array[0:frame+1],self.posterior_bound_array, color = 'coral', alpha = .5)
        ax5.plot(self.time_array,self.this_range_totalCOM_frame_XYZ[:,1], color = 'lightgrey')
        ax5.plot(self.time_array[0:frame+1],self.this_range_totalCOM_frame_XYZ[0:frame+1,1], color = 'grey')
        ax5.plot(self.time_array[frame],this_frame_total_COM_y,'*', color = 'purple', ms = 4)
        #ax5.legend(bbox_to_anchor=(1.12, -.65), ncol = 2)

        
        ax2.set_xlabel('X-Axis (mm)')
        ax2.set_ylabel('Y-Axis (mm)')
        
        ax.set_xlabel('X-Axis (mm)')
        ax.set_ylabel('Y-Axis (mm)')
        ax.set_zlabel('Z-Axis (mm)')
       
        


    def set_up_data(self):

        print('Loading skeleton and COM data from file paths')
        mediapipe_data_path,mediapipeSkeleton_file_name,totalCOM_data_path, segmentedCOM_data_path, syncedVideoPath =  self.create_paths_to_data_files(self.validation_data_path,self.sessionID)

        mediapipe_pose_data, mediapipeSkelcoordinates_frame_segment_joint_XYZ, segmentedCOM_frame_joint_XYZ, totalCOM_frame_XYZ = self.load_data_from_paths(mediapipe_data_path,mediapipeSkeleton_file_name,totalCOM_data_path, segmentedCOM_data_path)

        #mediapipe_pose_data = self.get_mediapipe_pose_data(mediapipeSkel_fr_mar_dim)

        if self.num_frame_range == 0:
            self.num_frame_range = range(0,mediapipe_pose_data.shape[0], self.step_interval) #if no range was specified, use the whole video 

        print('Slicing skeleton and COM data from frames {} to {}'.format(self.num_frame_range[0],self.num_frame_range[-1]))
        this_range_mp_pose_XYZ,this_range_mp_skeleton_segment_XYZ,this_range_segmentCOM_fr_joint_XYZ,this_range_totalCOM_frame_XYZ = self.slice_data_arrays_by_range(
            self.num_frame_range,mediapipe_pose_data, mediapipeSkelcoordinates_frame_segment_joint_XYZ, segmentedCOM_frame_joint_XYZ, totalCOM_frame_XYZ)



        #print('Loading video frames to plot from {}'.format(syncedVideoPath))
        #cap = self.load_video_capture_object(syncedVideoPath)

        #video_frames_to_plot = self.get_video_frames_to_plot(cap,self.num_frame_range)
        video_frames_to_plot = []

        #print('Video frames loaded')
        return this_range_mp_pose_XYZ,this_range_mp_skeleton_segment_XYZ,this_range_segmentCOM_fr_joint_XYZ,this_range_totalCOM_frame_XYZ, video_frames_to_plot

        f= 2


    def generate_plot(self,this_range_mp_pose_XYZ,this_range_mp_skeleton_segment_XYZ,this_range_segmentCOM_fr_joint_XYZ,this_range_totalCOM_frame_XYZ, video_frames_to_plot):
      

       
        figure = plt.figure(figsize= (10,10))
        figure.suptitle('Qualisys Center of Mass', fontsize = 16, y = .94, color = 'royalblue')
 
        # gs = figure.add_gridspec(3,2)
        self.ax_range = 900
        self.ax_com_range= 500

        # plt.tight_layout()
        ax = figure.add_subplot(3,2,(1,4), projection = '3d')
        ax2 = figure.add_subplot(325)
        #ax3 = figure.add_subplot(322)
        ax4 = figure.add_subplot(326)
        ax5 = figure.add_subplot(326)
        
        l1, b1, w1, h1 = ax.get_position().bounds
        ax.set_position([l1-l1*.72,b1+b1*.1,w1,h1])
        l2, b2, w2, h2 = ax2.get_position().bounds
        ax2.set_position([l2,b2+b2*.5,w2,h2])
        #l3, b3, w3, h3 = ax3.get_position().bounds
        #ax3.set_position([l3,b3-b3*.15,w3,h3])
        l4, b4, w4, h4 = ax4.get_position().bounds
        ax4.set_position([l4+l4*.1,b4+b4*2,w4,h4*.5])
        l5, b5, w5, h5 = ax5.get_position().bounds
        ax5.set_position([l5+l5*.1,b5+b5*.5,w5,h5*.5])
   


        # ax = figure.add_subplot(gs[0:,0:1], projection = '3d')
        # ax2 = figure.add_subplot(gs[2,0])
        # ax3 = figure.add_subplot(gs[1,1])
        # ax5 = figure.add_subplot(gs[2,1])


        self.ax = ax
        self.ax2 = ax2
        #self.ax3 = ax3
        self.ax4 = ax4
        self.ax5 = ax5

        skel_x = this_range_mp_pose_XYZ[:,:,0]
        skel_y = this_range_mp_pose_XYZ[:,:,1]
        skel_z = this_range_mp_pose_XYZ[:,:,2]

        num_frames_to_plot = int(np.ceil(((self.num_frame_range[-1] - self.num_frame_range[0])/self.step_interval)))
        num_frames_to_plot = range(num_frames_to_plot) #create a range for the number of frames to plot based on the interval (and round up)
        num_frame_length = (len(num_frames_to_plot)/5)-1 #5 because this is the interval to make the 300fps qualisys system = to our 60fps go pros
        self.mx = np.nanmean(skel_x[int(num_frame_length/2),:])
        self.my = np.nanmean(skel_y[int(num_frame_length/2),:])
        self.mz = np.nanmean(skel_z[int(num_frame_length/2),:])


        self.mx_com = np.nanmean(this_range_totalCOM_frame_XYZ[:,0])
        self.my_com = np.nanmean(this_range_totalCOM_frame_XYZ[:,1])

        
        self.skel_x = skel_x
        self.skel_y = skel_y
        self.skel_z = skel_z

        time_array = []
        for frame_num in num_frames_to_plot:
            time_array.append(frame_num/self.camera_fps)
        self.time_array = time_array


        self.this_range_mp_pose_XYZ = this_range_mp_pose_XYZ
        self.this_range_mp_skeleton_segment_XYZ = this_range_mp_skeleton_segment_XYZ
        self.this_range_segmentCOM_fr_joint_XYZ = this_range_segmentCOM_fr_joint_XYZ
        self.this_range_totalCOM_frame_XYZ = this_range_totalCOM_frame_XYZ
        
        self.img_artist = None #used to speed up the video plotting

        self.medial_bound_max_array = []
        self.medial_bound_min_array = []
        self.anterior_bound_array = []
        self.posterior_bound_array = []

        # time_array = []
        # for frame_num in self.num_frame_range:
        #     time_array.append(frame_num/300) #I think? have to figure out qualisys time conversion
        # self.time_array = time_array


        print('Starting Frame Animation') 
        ani = FuncAnimation(figure, self.animate, frames= num_frames_to_plot, interval=.1, repeat=False, fargs = (video_frames_to_plot,), init_func= self.animation_init)
        writervideo = animation.FFMpegWriter(fps=self.output_video_fps)
        ani.save(self.this_freemocap_session_path/'qualysis_test_plots.mp4', writer=writervideo)
        print('Animation has been saved to {}'.format(self.this_freemocap_session_path))
        f=2

this_computer_name = socket.gethostname()
print(this_computer_name)


if this_computer_name == 'DESKTOP-V3D343U':
    freemocap_validation_data_path = Path(r"I:\My Drive\HuMoN_Research_Lab\FreeMoCap_Stuff\FreeMoCap_Balance_Validation\data")
elif this_computer_name == 'DESKTOP-F5LCT4Q':
    freemocap_validation_data_path = Path(r"C:\Users\aaron\Documents\HumonLab\Spring2022\ValidationStudy\FreeMocap_Data")
    #freemocap_validation_data_path = Path(r'D:\freemocap2022\FreeMocap_Data')
else:
    #freemocap_validation_data_path = Path(r"C:\Users\kiley\Documents\HumonLab\SampleFMC_Data\FreeMocap_Data-20220216T173514Z-001\FreeMocap_Data")
    freemocap_validation_data_path = Path(r"C:\Users\Rontc\Documents\HumonLab\ValidationStudy")
sessionID = 'session_SER_1_20_22' #name of the sessionID folder

step_interval = 5
num_frame_range = range(60255,72255, step_interval)
#num_frame_range = range(65000,78000, step_interval)
#num_frame_range = 0
camera_fps = 300
output_video_fps = 60
tail_length = 120 #number of frames to keep the COM trajectory tail 
#num_frame_range = 0

COM_plot = animateSkeleton(freemocap_validation_data_path,sessionID,num_frame_range, step_interval, camera_fps, output_video_fps, tail_length)

this_range_mp_pose_XYZ,this_range_mp_skeleton_segment_XYZ,this_range_segmentCOM_fr_joint_XYZ,this_range_totalCOM_frame_XYZ, video_frames_to_plot = COM_plot.set_up_data()
COM_plot.generate_plot(this_range_mp_pose_XYZ,this_range_mp_skeleton_segment_XYZ,this_range_segmentCOM_fr_joint_XYZ,this_range_totalCOM_frame_XYZ, video_frames_to_plot)


# this_freemocap_session_path = freemocap_validation_data_path / sessionID
# this_freemocap_data_path = this_freemocap_session_path/'DataArrays'

# syncedVideoName = sessionID + '_Cam1_synced.mp4'

# syncedVideoPath = this_freemocap_session_path/'SyncedVideos'/syncedVideoName

# totalCOM_data_path = this_freemocap_data_path / 'totalBodyCOM_frame_XYZ.npy'
# segmentedCOM_data_path = this_freemocap_data_path / 'segmentedCOM_frame_joint_XYZ.npy'
# #mediapipe_data_path = this_freemocap_data_path/'mediaPipeSkel_3d_smoothed.npy'
# mediapipe_data_path = this_freemocap_data_path/'rotated_mediaPipeSkel_3d_smoothed.npy'
# mediapipeSkeleton_file_name = this_freemocap_data_path/'mediapipeSkelcoordinates_frame_segment_joint_XYZ.pkl'


# totalCOM_frame_XYZ = np.load(totalCOM_data_path) #loads in the data as a numpy array

# segmentedCOM_frame_joint_XYZ = np.load(segmentedCOM_data_path)

# mediapipeSkel_fr_mar_dim = np.load(mediapipe_data_path) #loads in the data as a numpy array

# open_file = open(mediapipeSkeleton_file_name, "rb")
# mediapipeSkelcoordinates_frame_segment_joint_XYZ = pickle.load(open_file)
# open_file.close()


# cap = cv2.VideoCapture(str(syncedVideoPath))

# fps = 60

# num_pose_joints = 33 #number of pose joints tracked by mediapipe 
# pose_joint_range = range(num_pose_joints)
# #frame_range = range(first_frame,last_frame)
# mediapipe_pose_data = mediapipeSkel_fr_mar_dim[:,0:num_pose_joints,:] #load just the pose joints into a data array, removing hands and face data 
# num_frames = len(mediapipe_pose_data)
# #num_frame_range = range(2900,3500)
# num_frame_range = range(num_frames)

# num_frame_range = range(9900,12200)

# skel_x = mediapipe_pose_data[:,:,0]
# skel_y = mediapipe_pose_data[:,:,1]
# skel_z = mediapipe_pose_data[:,:,2]



# mx = np.nanmean(skel_x[int(num_frames/2),:])
# my = np.nanmean(skel_y[int(num_frames/2),:])
# mz = np.nanmean(skel_z[int(num_frames/2),:])


# mx_com = np.nanmean(totalCOM_frame_XYZ[int(num_frames/2),0])
# my_com = np.nanmean(totalCOM_frame_XYZ[int(num_frames/2),1])


# figure = plt.figure(figsize= (10,10))
# ax_range = 1000
# ax_com_range= 500


 
# # plt.tight_layout()
# ax = figure.add_subplot(222, projection = '3d')
# ax2 = figure.add_subplot(224)
# ax3 = figure.add_subplot(221)









# def animate(frame,num_frames, video_frames_to_plot):


#     if frame % 100 == 0:
#         now = datetime.now()

#         current_time = now.strftime("%H:%M:%S")
#         print("Currently on frame: {} at {}".format(frame,current_time))

#     this_frame_skel_x = skel_x[frame,:]
#     this_frame_skel_y = skel_y[frame,:]
#     this_frame_skel_z = skel_z[frame,:]

#     left_heel_x = this_frame_skel_x[30]
#     left_heel_z = this_frame_skel_y[30]

#     left_toe_x = this_frame_skel_x[32]
#     left_toe_z = this_frame_skel_y[32]

#     right_heel_x = this_frame_skel_x[29]
#     right_heel_z = this_frame_skel_y[29]

#     right_toe_x = this_frame_skel_x[31]
#     right_toe_z = this_frame_skel_y[31]

#     left_foot_x,left_foot_z = [left_heel_x,left_toe_x], [left_heel_z,left_toe_z]
#     right_foot_x,right_foot_z = [right_heel_x,right_toe_x], [right_heel_z,right_toe_z]

#     segment_COM_x = segmentedCOM_frame_joint_XYZ[frame,:,0]
#     segment_COM_y = segmentedCOM_frame_joint_XYZ[frame,:,1]
#     segment_COM_z = segmentedCOM_frame_joint_XYZ[frame,:,2]

#     total_COM_x = totalCOM_frame_XYZ[frame,0]
#     total_COM_y = totalCOM_frame_XYZ[frame,1]
#     total_COM_z = totalCOM_frame_XYZ[frame,2]

#     plot_frame_bones_XYZ = mediapipeSkelcoordinates_frame_segment_joint_XYZ[frame]
#     ax.clear()
#     #ax2.clear()
#     ax2.set_xlim([mx_com-ax_com_range, mx_com+ax_range])
#     ax2.set_ylim([my_com-ax_com_range, my_com+ax_range])


#     ax.set_xlim([mx-ax_range, mx+ax_range])
#     ax.set_ylim([my-ax_range, my+ax_range])
#     ax.set_zlim([mz-ax_range, mz+ax_range])
#     #ax.view_init(elev=-70., azim=-60)
#     ax.view_init(elev = 0, azim =-70)
#     for segment in plot_frame_bones_XYZ.keys():
#         prox_joint = plot_frame_bones_XYZ[segment][0] 
#         dist_joint = plot_frame_bones_XYZ[segment][1]
        
#         bone_x,bone_y,bone_z = [prox_joint[0],dist_joint[0]],[prox_joint[1],dist_joint[1]],[prox_joint[2],dist_joint[2]] 

#         ax.plot(bone_x,bone_y,bone_z,color = 'black')
    
#     ax.scatter(segment_COM_x,segment_COM_y,segment_COM_z, color = 'orange')    

#     ax.scatter(total_COM_x,total_COM_y,total_COM_z, color = 'purple')
   

#     ax.scatter(this_frame_skel_x, this_frame_skel_y,this_frame_skel_z)

#     #plt.show()
#     #plt.pause()

#     ax2.scatter(total_COM_x,total_COM_y, marker = '.', color = 'black', s = 5)
#     ax2.plot(left_foot_x,left_foot_z, color = 'blue')
#     ax2.plot(right_foot_x,right_foot_z, color = 'red')

#     cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
#     success, frame = cap.read()
#     if success:
#         ax3.imshow(frame)
#     # f=2

# # video_frames_to_plot = []
# # for frame_array in tqdm(num_frame_range):
# #     cap.set(cv2.CAP_PROP_POS_FRAMES, frame_array)
# #     success, frame = cap.read()

# #     video_frames_to_plot.append(frame)

# # cap.release()

# #ani = FuncAnimation(figure, animate, frames= num_frame_range, interval=(1/fps)*100, repeat=False, fargs = (num_frames,video_frames_to_plot,))

# ani = FuncAnimation(figure, animate, frames= num_frame_range, interval=(1/fps)*100, repeat=False, fargs = (num_frames,[]))


# writervideo = animation.FFMpegWriter(fps=fps)
# ani.save(this_freemocap_session_path/'ytest_with_trajectory_rotated_2.mp4', writer=writervideo)

# #plt.show()



# f=2