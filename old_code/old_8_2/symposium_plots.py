
from matplotlib.markers import MarkerStyle
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

from io import BytesIO



#you can skip every 10th frame 
class skeleton_COM_Plot:
    def __init__(self, freemocap_validation_data_path, sessionID, num_frame_range, camera_fps, output_video_fps, tail_length, stance, static_plot = False):
        self.num_frame_range = num_frame_range

        self.sessionID = sessionID

        self.validation_data_path = freemocap_validation_data_path

        self.camera_fps = camera_fps 

        self.output_video_fps = output_video_fps

        self.tail_length = tail_length

        self.static_plot = static_plot
        self.stance = stance

    def create_paths_to_data_files(self, validation_data_path, sessionID):
        
        this_freemocap_session_path = validation_data_path / sessionID
        this_freemocap_data_path = this_freemocap_session_path/'output_data'





        totalCOM_data_path = this_freemocap_data_path /'center_of_mass'/ 'total_body_center_of_mass_xyz.npy'
        segmentedCOM_data_path = this_freemocap_data_path /'center_of_mass'/'segmentCOM_frame_joint_xyz.npy'
        #mediapipe_data_path = this_freemocap_data_path/'mediaPipeSkel_3d_smoothed.npy'
        #mediapipe_data_path = this_freemocap_data_path/'rotated_mediaPipeSkel_3d_smoothed.npy'
        mediapipe_data_path = this_freemocap_data_path/'mediapipe_body_3d_xyz.npy'
        #mediapipe_data_path = this_freemocap_data_path/'mediapipe_origin_corrected_and_rotated.npy' 
        # mediapipeSkeleton_file_name = this_freemocap_data_path/'mediapipe_skeleton_segments_dict.pkl'
        mediapipeSkeleton_file_name = this_freemocap_data_path/'center_of_mass'/'origin_aligned_mediapipe_Skelcoordinates_frame_segment_joint_XYZ.pkl'



        syncedVideoName = sessionID + '_synced_Cam1.mp4'
        #syncedVideoName = sessionID + '_mediapipe_overlayed_video.mp4'


        syncedVideoPath = this_freemocap_session_path/'synchronized_videos'/syncedVideoName

        self.this_freemocap_session_path = this_freemocap_session_path #needed when saving out the plot video 

        return mediapipe_data_path, mediapipeSkeleton_file_name, totalCOM_data_path, segmentedCOM_data_path, syncedVideoPath

    
    def load_data_from_paths(self,mediapipe_data_path,mediapipeSkeleton_file_name,totalCOM_data_path, segmentedCOM_data_path):

        totalCOM_frame_XYZ = np.load(totalCOM_data_path) #loads in the data as a numpy array

        segmentedCOM_frame_joint_XYZ = np.load(segmentedCOM_data_path)

        mediapipeSkel_fr_mar_dim = np.load(mediapipe_data_path) #loads in the data as a numpy array

        open_file = open(mediapipeSkeleton_file_name, "rb")
        mediapipeSkelcoordinates_frame_segment_joint_XYZ = pickle.load(open_file)
        open_file.close()

        return mediapipeSkel_fr_mar_dim, mediapipeSkelcoordinates_frame_segment_joint_XYZ, segmentedCOM_frame_joint_XYZ, totalCOM_frame_XYZ

    def get_mediapipe_pose_data(self,  mediapipeSkel_fr_mar_dim):

        num_pose_joints = 33 #yes, this is hardcoded. but if mediapipe updates to use a different skeleton we need to update a lot of things anyway 

        mediapipe_pose_data =  mediapipeSkel_fr_mar_dim[:,0:num_pose_joints,:]

        return mediapipe_pose_data

    def slice_data_arrays_by_range(self,num_frame_range,mediapipe_pose_data, mediapipeSkelcoordinates_frame_segment_joint_XYZ, segmentedCOM_frame_joint_XYZ, totalCOM_frame_XYZ):

            
        start_frame = num_frame_range[0]
        end_frame = num_frame_range[-1]

        
        this_range_mp_pose_XYZ = mediapipe_pose_data[start_frame:end_frame,:,:]

        this_range_mp_skeleton_segment_XYZ = mediapipeSkelcoordinates_frame_segment_joint_XYZ[start_frame:end_frame]

        this_range_segmentCOM_fr_joint_XYZ = segmentedCOM_frame_joint_XYZ[start_frame:end_frame,:,:]

        this_range_totalCOM_frame_XYZ = totalCOM_frame_XYZ[start_frame:end_frame,:]

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
       
        frame = start_frame
       
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
       
        while frame < end_frame:
            success, image = cap.read()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            video_frames_to_plot.append(image)
            frame += 1

        # for frame in track(num_frame_range):
        #         cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        #         success, image = cap.read()
        #         video_frames_to_plot.append(image)
        cap.release()
        #print('finished getting video frames')
        return video_frames_to_plot

    def get_foot_data(self, skeleton_data_x, skeleton_data_y, skeleton_data_z, heel_index, toe_index) :
        
        toe_data_x = skeleton_data_x[:,toe_index]
        toe_data_y = skeleton_data_y[:,toe_index]
        toe_data_z = skeleton_data_z[:,toe_index]

        heel_data_x = skeleton_data_x[:,heel_index]
        heel_data_y = skeleton_data_y[:,heel_index]
        heel_data_z = skeleton_data_z[:,heel_index]


        foot_data_x = [heel_data_x,toe_data_x]
        foot_data_y = [heel_data_y,toe_data_y]
        foot_data_z = [heel_data_z,toe_data_z]

        foot_data_XYZ = [foot_data_x,foot_data_y,foot_data_z]

        avg_foot_position_x = np.mean(foot_data_x,axis=0)
        avg_foot_position_y = np.mean(foot_data_y,axis=0)
        avg_foot_position_z = np.mean(foot_data_z,axis=0)

        avg_foot_data_XYZ = [avg_foot_position_x,avg_foot_position_y,avg_foot_position_z]

        f =2 
        return foot_data_XYZ, avg_foot_data_XYZ
    def get_anterior_posterior_bounds(self,left_foot_data,right_foot_data):
        
        left_foot_x = left_foot_data[0]
        left_foot_y = left_foot_data[1]
        left_foot_z = left_foot_data[2]

        right_foot_x = right_foot_data[0]
        right_foot_y = right_foot_data[1]
        right_foot_z = right_foot_data[2]

        front_foot_avg_position_x = np.mean([left_foot_x[1],right_foot_x[1]],axis = 0) # 0 = toe data for each foot
        front_foot_avg_position_y = np.mean([left_foot_y[1],right_foot_y[1]],axis = 0)
        front_foot_avg_position_z = np.mean([left_foot_z[1],right_foot_z[1]],axis = 0)

        back_foot_avg_position_x = np.mean([left_foot_x[0],right_foot_x[0]],axis = 0)
        back_foot_avg_position_y = np.mean([left_foot_y[0],right_foot_y[0]],axis = 0)
        back_foot_avg_position_z = np.mean([left_foot_z[0],right_foot_z[0]],axis = 0)

        
        front_foot_avg_XYZ = [front_foot_avg_position_x, front_foot_avg_position_y, front_foot_avg_position_z]
        back_foot_avg_XYZ = [back_foot_avg_position_x, back_foot_avg_position_y, back_foot_avg_position_z]

        return front_foot_avg_XYZ, back_foot_avg_XYZ


    def animation_init(self):
        #the FuncAnimation needs an initial function that it will run, otherwise it will run animate() twice for frame 0 
        pass
    def animate(self,frame,video_frames_to_plot):

        ax = self.ax #NOTE - will redefining self.ax as ax at the start of this function have a screwy effect down the line?
        ax2 = self.ax2
        # ax3 = self.ax3
        # ax4 = self.ax4
        # ax5 = self.ax5


        if frame % 100 == 0:
            now = datetime.now()

            current_time = now.strftime("%H:%M:%S")
            print("Currently on frame: {} at {}".format(frame,current_time))

        # now = datetime.now()

        # current_time = now.strftime("%H:%M:%S")
        # print("Currently on frame: {} at {}".format(frame,current_time))

        #Get the skeleton data for this frame
        this_frame_skel_x = -1*self.skel_x[frame,:]
        this_frame_skel_y = -1*self.skel_y[frame,:] #*-1 to flip the y axis display in the plot
        this_frame_skel_z = self.skel_z[frame,:]

        #Get the foot data for this frame

        # #Get the mediapipe feet data for this frame
        # left_heel_x = this_frame_skel_x[29]
        # left_heel_y = this_frame_skel_y[29]
        # left_heel_z = this_frame_skel_z[29]

        # left_toe_x = this_frame_skel_x[31]
        # left_toe_y = this_frame_skel_y[31]
        # left_toe_z = this_frame_skel_z[31]

        # right_heel_x =this_frame_skel_x[30]
        # right_heel_y = this_frame_skel_y[30]
        # right_heel_z = this_frame_skel_z[30]

        # right_toe_x =this_frame_skel_x[32]
        # right_toe_y = this_frame_skel_y[32]
        # right_toe_z = this_frame_skel_z[32]

        # left_foot_x,left_foot_y, left_foot_z = [left_heel_x,left_toe_x], [left_heel_y,left_toe_y], [left_heel_z,left_toe_z]
        # right_foot_x,right_foot_y, right_foot_z = [right_heel_x,right_toe_x], [right_heel_y,right_toe_y], [right_heel_z,right_toe_z]

        # left_foot_average_position_x = (left_heel_x + left_toe_x)/2
        # right_foot_average_position_x = (right_heel_x + right_toe_x)/2
        
        # left_foot_average_position_y = (left_heel_y + left_toe_y)/2
        # right_foot_average_position_y = (right_heel_y + right_toe_y)/2

        # back_foot_average_position_x = (left_heel_x + right_heel_x)/2
        # front_foot_average_position_x = (left_toe_x + right_toe_x)/2

        # back_foot_average_position_y = (left_heel_y + right_heel_y)/2
        # front_foot_average_position_y = (left_toe_y + right_toe_y)/2

        # self.medial_bound_max_array.append(left_foot_average_position_x)
        # self.medial_bound_min_array.append(right_foot_average_position_x)

        # self.anterior_bound_array.append(front_foot_average_position_y)
        # self.posterior_bound_array.append(back_foot_average_position_y)


    
        left_foot_x = self.left_foot_position_XYZ[0]
        left_foot_y = self.left_foot_position_XYZ[1]
        left_foot_z = self.left_foot_position_XYZ[2]

        right_foot_x = self.right_foot_position_XYZ[0]
        right_foot_y = self.right_foot_position_XYZ[1]
        right_foot_z = self.right_foot_position_XYZ[2]

        this_frame_left_foot_x = [-1*left_foot_x[joint][frame] for joint in range(2)]
        this_frame_left_foot_y = [-1*left_foot_y[joint][frame] for joint in range(2)]
        this_frame_left_foot_z = [left_foot_z[joint][frame] for joint in range(2)]

        this_frame_right_foot_x = [-1*right_foot_x[joint][frame] for joint in range(2)]
        this_frame_right_foot_y = [-1*right_foot_y[joint][frame] for joint in range(2)]
        this_frame_right_foot_z = [right_foot_z[joint][frame] for joint in range(2)]

        left_foot_avg_x = -1*self.left_foot_avg_position_XYZ[0]
        left_foot_avg_y = -1*self.left_foot_avg_position_XYZ[1]
        left_foot_avg_z = self.left_foot_avg_position_XYZ[2]

        right_foot_avg_x = -1*self.right_foot_avg_position_XYZ[0]
        right_foot_avg_y = -1*self.right_foot_avg_position_XYZ[1]
        right_foot_avg_z = self.right_foot_avg_position_XYZ[2]

        this_frame_left_foot_avg_x = left_foot_avg_x[frame]
        this_frame_left_foot_avg_y = left_foot_avg_y[frame]
        this_frame_left_foot_avg_z = left_foot_avg_z[frame]

        this_frame_right_foot_avg_x = right_foot_avg_x[frame]
        this_frame_right_foot_avg_y = right_foot_avg_y[frame]
        this_frame_right_foot_avg_z = right_foot_avg_z[frame]

        back_foot_avg_x = -1*self.back_foot_avg_position_XYZ[0]
        back_foot_avg_y = -1*self.back_foot_avg_position_XYZ[1]
        back_foot_avg_z = self.back_foot_avg_position_XYZ[2]

        front_foot_avg_x = -1*self.front_foot_avg_position_XYZ[0]
        front_foot_avg_y = -1*self.front_foot_avg_position_XYZ[1]
        front_foot_avg_z = self.front_foot_avg_position_XYZ[2]

        this_frame_back_foot_avg_x = back_foot_avg_x[frame]
        this_frame_back_foot_avg_y = back_foot_avg_y[frame]
        this_frame_back_foot_avg_z = back_foot_avg_z[frame]

        this_frame_front_foot_avg_x = front_foot_avg_x[frame]
        this_frame_front_foot_avg_y = front_foot_avg_y[frame]
        this_frame_front_foot_avg_z = front_foot_avg_z[frame]



        #Get the mediapipe segment COM data for this frame
        this_frame_segment_COM_x = -1*self.this_range_segmentCOM_fr_joint_XYZ[frame,:,0]
        this_frame_segment_COM_y = -1*self.this_range_segmentCOM_fr_joint_XYZ[frame,:,1]
        this_frame_segment_COM_z = self.this_range_segmentCOM_fr_joint_XYZ[frame,:,2]


        #Get the mediapipe total COM data for this frame
        this_frame_total_COM_x = -1*self.this_range_totalCOM_frame_XYZ[frame,0]
        this_frame_total_COM_y = -1*self.this_range_totalCOM_frame_XYZ[frame,1]
        this_frame_total_COM_z = self.this_range_totalCOM_frame_XYZ[frame,2]

        #Get the bonez data for this frame
        this_frame_bones_XYZ = self.this_range_mp_skeleton_segment_XYZ[frame]
        
        #clear the necessary axes 
        ax.cla()
        ax2.cla()
        # ax4.cla()
        # ax5.cla()
        
        #axes titles and limits 
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
        
 
        #ax2.set_xlim([self.mx_com-self.ax_com_range, self.mx_com+self.ax_com_range])
        #ax2.set_ylim([self.my_com-self.ax_com_range, self.my_com+self.ax_com_range])
        ax2.set_title('Total Body COM Trajectory')
        ax2.set_xlim([self.mx_com-225, self.mx_com+225])
        ax2.set_ylim([self.my_com-225, self.my_com+225])
        
        ax2.set_aspect('equal', adjustable='box')

        # ax3.set_xlim([700,1300])
        # #ax3.set_ylim([1000,200])

        # ax4.set_title('Max/Min Lateral COM Position vs. Time')
        # ax4.set_ylim([self.mx_com-300, self.mx_com+300])

        # # if self.time_array[-1] - self.time_array[0] < 10: #if the data is shorter than a 10 second window 
        # ax4.set_xlim(self.time_array[0],self.time_array[-1])
        # ax5.set_xlim(self.time_array[0],self.time_array[-1])
        # # else: #if the 10 second window would go past 0 on the time axis, start from 0 instead 
        # #     if self.time_array[frame] - 10 < 0:
        # #         ax4.set_xlim([self.time_array[0],self.time_array[frame]+10])
        # #         ax5.set_xlim(self.time_array[0],self.time_array[-1])
        # #     else:
        # #         ax4.set_xlim([self.time_array[frame]-10,self.time_array[frame]+10])
        # #         ax5.set_xlim([self.time_array[frame]-10,self.time_array[frame]+10])

        # ax4.axes.xaxis.set_ticks([])

        # ax5.set_title('Anterior/Posterior COM Position vs. Time')
        # ax5.set_ylim([self.my_com-350, self.my_com+350])

        #ax.view_init(elev=-70., azim=-60)
        ax.view_init(elev = 0, azim =-70)
        
        #plotting a line across the shoulders 
        left_shoulder_joint = this_frame_bones_XYZ['left_upper_arm'][0]
        right_shoulder_joint = this_frame_bones_XYZ['right_upper_arm'][0]
        shoulder_connection_x, shoulder_connection_y, shoulder_connection_z = [left_shoulder_joint[0],right_shoulder_joint[0]],[left_shoulder_joint[1],right_shoulder_joint[1]],[left_shoulder_joint[2],right_shoulder_joint[2]]
        
        #plotting a line across the hips

        left_hip_joint = this_frame_bones_XYZ['left_thigh'][0]
        right_hip_joint = this_frame_bones_XYZ['right_thigh'][0]
        hip_connection_x, hip_connection_y, hip_connection_z = [left_hip_joint[0],right_hip_joint[0]],[left_hip_joint[1],right_hip_joint[1]],[left_hip_joint[2],right_hip_joint[2]]

        ## 3D skeleton, segment COM, and total body COM plot 
        for segment in this_frame_bones_XYZ.keys():
            prox_joint = this_frame_bones_XYZ[segment][0] 
            dist_joint = this_frame_bones_XYZ[segment][1]
            
            bone_x,bone_y,bone_z = [prox_joint[0],dist_joint[0]],[prox_joint[1],dist_joint[1]],[prox_joint[2],dist_joint[2]] 

            bone_x = [-x for x in bone_x]
            bone_y = [-x for x in bone_y]

            ax.plot(bone_x,bone_y,bone_z,color = 'black')

            ax2.plot(bone_x,bone_y, color = 'grey', alpha = .4) #plot the bones transparently on the trajectory graph while we're at it 

        ax.plot(-1*np.array(shoulder_connection_x),-1*np.array(shoulder_connection_y),shoulder_connection_z,color = 'black')
        ax2.plot(-1*np.array(shoulder_connection_x),-1*np.array(shoulder_connection_y),color = 'grey', alpha = .4)
        ax.plot(-1*np.array(hip_connection_x), -1*np.array(hip_connection_y), hip_connection_z, color = 'black')
        ax2.plot(-1*np.array(hip_connection_x),-1*np.array(hip_connection_y),color = 'grey', alpha = .4)

        ax.plot(this_frame_left_foot_x,this_frame_left_foot_y,this_frame_left_foot_z,color = 'blue')
        ax.plot(this_frame_right_foot_x,this_frame_right_foot_y,this_frame_right_foot_z,color = 'red')

        # this_frame_segment_COM_y = [-x for x in this_frame_segment_COM_y]
        # this_frame_total_COM_y = -this_frame_total_COM_y
        # this_frame_skel_y = -this_frame_skel_y

        # ax.scatter(this_frame_segment_COM_x,this_frame_segment_COM_y,this_frame_segment_COM_z, color = 'orange', label = 'Segment COM')    
        ax.scatter(this_frame_total_COM_x,this_frame_total_COM_y,this_frame_total_COM_z, color = 'magenta', label = 'Total Body COM', marker = '*', s = 70, edgecolor = 'purple')
        ax.scatter(this_frame_skel_x, this_frame_skel_y,this_frame_skel_z, color = 'grey')

        #ax.plot(left_foot_x,left_foot_y,left_foot_z, color = 'blue')
        #ax.plot(right_foot_x,right_foot_y,right_foot_z, color = 'red')
        ax.scatter(this_frame_left_foot_x,this_frame_left_foot_y,this_frame_left_foot_z, color = 'blue')
        ax.scatter(this_frame_right_foot_x,this_frame_right_foot_y,this_frame_right_foot_z, color = 'red')


        ax.legend()


      
        plot_fade_frame = frame - self.tail_length

        #plot the past trajectory of the total body center of mass, if the frame would go past the beginning of the trajectory, just plot the first frame
        if plot_fade_frame < 0:
            ax2.plot(-1*self.this_range_totalCOM_frame_XYZ[0:frame,0],-1*self.this_range_totalCOM_frame_XYZ[0:frame,1], color = 'grey')
        else:
            ax2.plot(-1*self.this_range_totalCOM_frame_XYZ[plot_fade_frame:frame,0],-1*self.this_range_totalCOM_frame_XYZ[plot_fade_frame:frame,1], color = 'grey')
        
        ax2.plot(this_frame_total_COM_x,this_frame_total_COM_y, marker = '*', color = 'magenta', markeredgecolor = 'purple', ms = 4)

        if self.stance == 'natural':
            ax2.plot(this_frame_left_foot_x,this_frame_left_foot_y, color = 'blue', label= 'Max Lateral Bound (Left Foot)')
            ax2.plot(this_frame_right_foot_x,this_frame_right_foot_y, color = 'red', label = 'Min Lateral Bound (Right Foot)')
            
            ax2.plot(this_frame_left_foot_avg_x, this_frame_left_foot_avg_y, color = 'blue', ms = 3, marker = 'o')
            ax2.plot(this_frame_right_foot_avg_x, this_frame_right_foot_avg_y, color = 'red', ms = 3, marker = 'o')
            
            # ax2.plot(this_frame_back_foot_avg_x,this_frame_back_foot_avg_y, color = 'coral', marker = 'o', ms = 3, alpha = .3)
            # ax2.plot(this_frame_front_foot_avg_x,this_frame_front_foot_avg_y, color = 'forestgreen', marker = 'o', ms = 3, alpha = .3)

            # #plot the a line between the heels, and then a line between the toes
            # ax2.plot([this_frame_left_foot_x[0],this_frame_right_foot_x[0]],[this_frame_left_foot_y[0],this_frame_right_foot_y[0]], color = 'coral', label = 'Posterior Bound', alpha = .3, linestyle = '--')
            # ax2.plot([this_frame_left_foot_x[1],this_frame_right_foot_x[1]],[this_frame_left_foot_y[1],this_frame_right_foot_y[1]], color = 'forestgreen', label = 'Anterior Bound', alpha = .3, linestyle = '--')
        
        elif self.stance == 'left_leg':
            ax2.plot(this_frame_left_foot_x,this_frame_left_foot_y, color = 'blue', label= 'Max Lateral Bound (Left Foot)')
            # ax2.plot(this_frame_right_foot_x,this_frame_right_foot_y, color = 'red', label = 'Min Lateral Bound (Right Foot)')
            
            ax2.plot(this_frame_left_foot_avg_x, this_frame_left_foot_avg_y, color = 'blue', ms = 3, marker = 'o')
            # ax2.plot(this_frame_right_foot_avg_x, this_frame_right_foot_avg_y, color = 'red', ms = 3, marker = 'o')
            

            ax2.plot(this_frame_left_foot_x[0],this_frame_left_foot_y[0], color = 'coral', marker = 'o', ms = 4, alpha = .7, label = 'Posterior Bound') #left leg one leg stance plotting
            ax2.plot(this_frame_left_foot_x[1],this_frame_left_foot_y[1], color = 'forestgreen', marker = 'o', ms = 4, alpha = .7, label = 'Anterior Bound') #left leg one leg stance plotting

        elif self.stance == 'right_leg':
            # ax2.plot(this_frame_left_foot_x,this_frame_left_foot_y, color = 'blue', label= 'Max Lateral Bound (Left Foot)')
            ax2.plot(this_frame_right_foot_x,this_frame_right_foot_y, color = 'red', label = 'Min Lateral Bound (Right Foot)')
            
            # ax2.plot(this_frame_left_foot_avg_x, this_frame_left_foot_avg_y, color = 'blue', ms = 3, marker = 'o')
            ax2.plot(this_frame_right_foot_avg_x, this_frame_right_foot_avg_y, color = 'red', ms = 3, marker = 'o')
            

            ax2.plot(this_frame_right_foot_x[0],this_frame_right_foot_y[0], color = 'coral', marker = 'o', ms = 4, alpha = .7, label = 'Posterior Bound') #left leg one leg stance plotting
            ax2.plot(this_frame_right_foot_x[1],this_frame_right_foot_y[1], color = 'forestgreen', marker = 'o', ms = 4, alpha = .7, label = 'Anterior Bound') #left leg one leg stance plotting


        # ax2.plot(left_toe_x,left_toe_y, color = 'coral', marker = 'o', ms = 4, alpha = .7, label = 'Anterior Bound') #left leg one leg stance plotting
        # ax2.plot(left_heel_x,left_heel_y, color = 'forestgreen', marker = 'o', ms = 4, alpha = .7, label = 'Posterior Bound') #left leg one leg stance plotting

        # ax2.legend(loc = 'lower left', bbox_to_anchor = (1,0))
      

        # video_frame = video_frames_to_plot[frame]

        # if self.img_artist is None:

        #     self.img_artist = ax3.imshow(video_frame)
        
        # else:
        #     self.img_artist.set_data(video_frame)


    
        # ax4.plot(self.num_frame_range[frame],total_COM_z, 'o', color = 'black', ms = 1.5)
        # ax4.plot(self.num_frame_range[0:frame],self.this_range_totalCOM_frame_XYZ[0:frame,2])
        # ax4.text(.5,.5,'Frame# {}'.format(str(self.num_frame_range[frame])))
        # ax4.set_title('COM Position vs. Frame')
        # ax4.set_ylabel('Vertical Position (units)')

        # ax4.set_ylabel('X Position (mm)')
        
        # ax5.set_xlabel('Time (s)')
        # ax5.set_ylabel('Y Position (mm)')
                
        ax2.set_xlabel('X Position (mm)')
        ax2.set_ylabel('Y Position (mm)')
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')

  
        # ax4.axhline(left_foot_average_position_x, color = 'blue', alpha = .5, label = 'Mean Left Foot Position')
        # ax4.axhline(right_foot_average_position_x, color = 'red', alpha = .5, label =  'Mean Right Foot Position')
        # if self.stance == 'natural':
            # Natural Stance Plot---------------------------------------------------------------------------------------
            #future points

            # ax4.plot(self.time_array, left_foot_avg_x, color = 'paleturquoise')
            # ax4.plot(self.time_array, right_foot_avg_x, color = 'lightpink')

            # ax4.plot(self.time_array,-1*self.this_range_totalCOM_frame_XYZ[:,0], color = 'lightgrey')

            # #current points
            # ax4.axvline(self.time_array[frame], color = 'black')

            # ax4.plot(self.time_array[0:frame+1],left_foot_avg_x[0:frame+1], color = 'blue', alpha = .5)
            # ax4.plot(self.time_array[0:frame+1],right_foot_avg_x[0:frame+1], color = 'red', alpha = .5)

            # ax4.plot(self.time_array[0:frame+1],-1*self.this_range_totalCOM_frame_XYZ[0:frame+1,0], color = 'grey')
            
        
            # ax4.plot(self.time_array[frame],this_frame_total_COM_x, '*', color = 'magenta', ms = 8, markeredgecolor = 'purple')

            
            #ax4.legend(bbox_to_anchor=(1.1, -1.85), ncol= 2)

            #ax5.axhline(back_foot_average_position_y, color = 'forestgreen', alpha = .5, label = 'Mean Back Foot Position')
            #ax5.axhline(front_foot_average_position_y, color = 'coral', alpha = .5, label = 'Mean Front Foot Position')

            #future points

            # ax5.plot(self.time_array, front_foot_avg_y, color = 'palegreen', alpha = .3)
            # ax5.plot(self.time_array, back_foot_avg_y, color = 'lightcoral', alpha = .3)
            # ax5.plot(self.time_array,-1*self.this_range_totalCOM_frame_XYZ[:,1], color = 'lightgrey')

            # ax5.axvline(self.time_array[frame], color = 'black')

            # ax5.plot(self.time_array[0:frame+1],front_foot_avg_y[0:frame+1], color = 'forestgreen', alpha = .5)
            # ax5.plot(self.time_array[0:frame+1],back_foot_avg_y[0:frame+1], color = 'coral', alpha = .5)
            # ax5.plot(self.time_array[0:frame+1],-1*self.this_range_totalCOM_frame_XYZ[0:frame+1,1], color = 'grey')

            # ax5.plot(self.time_array[0:frame+1],self.anterior_bound_array, color = 'forestgreen', alpha = .5)
            # ax5.plot(self.time_array[0:frame+1],self.posterior_bound_array, color = 'coral', alpha = .5)

            # ax5.plot(self.time_array[frame],this_frame_total_COM_y,'*', color = 'magenta', ms = 8, markeredgecolor = 'purple')
            #ax5.legend(bbox_to_anchor=(1.12, -.65), ncol = 2)

            
            # ax4.set_ylim([-460,140])
            # ax5.set_ylim([-340,260])

        # elif self.stance == 'left_leg':
        #     #Left Leg Stance Plot-----------------------------------------------------------------------------------------------------------
        #     ax4.plot(self.time_array, left_foot_avg_x, color = 'paleturquoise')
        #     ax4.plot(self.time_array,-1*self.this_range_totalCOM_frame_XYZ[:,0], color = 'lightgrey')
            
        #     ax4.axvline(self.time_array[frame], color = 'black')
        #     ax4.plot(self.time_array[0:frame+1],left_foot_avg_x[0:frame+1], color = 'blue', alpha = .5)
        #     ax4.plot(self.time_array[0:frame+1],-1*self.this_range_totalCOM_frame_XYZ[0:frame+1,0], color = 'grey')
        
        #     ax4.plot(self.time_array[frame],this_frame_total_COM_x, '*', color = 'magenta', ms = 8, markeredgecolor = 'purple')


        #     ax5.plot(self.time_array, -1*left_foot_y[1], color = 'palegreen', alpha = .3)
        #     ax5.plot(self.time_array, -1*left_foot_y[0], color = 'lightcoral', alpha = .3)
        #     ax5.plot(self.time_array,-1*self.this_range_totalCOM_frame_XYZ[:,1], color = 'lightgrey')

        #     ax5.axvline(self.time_array[frame], color = 'black')
        #     ax5.plot(self.time_array[0:frame+1],-1*left_foot_y[1][0:frame+1], color = 'forestgreen', alpha = .5)
        #     ax5.plot(self.time_array[0:frame+1],-1*left_foot_y[0][0:frame+1], color = 'coral', alpha = .5)
        #     ax5.plot(self.time_array[0:frame+1],-1*self.this_range_totalCOM_frame_XYZ[0:frame+1,1], color = 'grey')

            # ax4.set_ylim([-590, 510])
            # ax5.set_ylim([-690, 409])
        # elif self.stance == 'right_leg': 
        # #Right Leg Stance Plot-----------------------------------------------------------------------------------------------------------

        #     ax4.plot(self.time_array, right_foot_avg_x, color = 'lightpink')
        #     ax4.plot(self.time_array,-1*self.this_range_totalCOM_frame_XYZ[:,0], color = 'lightgrey')
            
        #     ax4.axvline(self.time_array[frame], color = 'black')
        #     ax4.plot(self.time_array[0:frame+1],right_foot_avg_x[0:frame+1], color = 'red', alpha = .5)
        #     ax4.plot(self.time_array[0:frame+1],-1*self.this_range_totalCOM_frame_XYZ[0:frame+1,0], color = 'grey')
        
        #     ax4.plot(self.time_array[frame],this_frame_total_COM_x, '*', color = 'magenta', ms = 8, markeredgecolor = 'purple')


        #     ax5.plot(self.time_array, -1*right_foot_y[1], color = 'palegreen', alpha = .3)
        #     ax5.plot(self.time_array, -1*right_foot_y[0], color = 'lightcoral', alpha = .3)
        #     ax5.plot(self.time_array,-1*self.this_range_totalCOM_frame_XYZ[:,1], color = 'lightgrey')

        #     ax5.axvline(self.time_array[frame], color = 'black')
        #     ax5.plot(self.time_array[0:frame+1],-1*right_foot_y[1][0:frame+1], color = 'forestgreen', alpha = .5)
        #     ax5.plot(self.time_array[0:frame+1],-1*right_foot_y[0][0:frame+1], color = 'coral', alpha = .5)
        #     ax5.plot(self.time_array[0:frame+1],-1*self.this_range_totalCOM_frame_XYZ[0:frame+1,1], color = 'grey')

            
            #ax4.set_ylim([-899, 401])
            #ax5.set_ylim([-739, 561])
            
    

    def set_up_data(self):

        print('Loading skeleton and COM data from file paths')
        mediapipe_data_path,mediapipeSkeleton_file_name,totalCOM_data_path, segmentedCOM_data_path, syncedVideoPath =  self.create_paths_to_data_files(self.validation_data_path,self.sessionID)

        mediapipeSkel_fr_mar_dim, mediapipeSkelcoordinates_frame_segment_joint_XYZ, segmentedCOM_frame_joint_XYZ, totalCOM_frame_XYZ = self.load_data_from_paths(mediapipe_data_path,mediapipeSkeleton_file_name,totalCOM_data_path, segmentedCOM_data_path)

        mediapipe_pose_data = self.get_mediapipe_pose_data(mediapipeSkel_fr_mar_dim)

        if self.num_frame_range == 0:
            self.num_frame_range = range(mediapipe_pose_data.shape[0]) #if no range was specified, use the whole video 

        print('Slicing skeleton and COM data from frames {} to {}'.format(self.num_frame_range[0],self.num_frame_range[-1]))
        this_range_mp_pose_XYZ,this_range_mp_skeleton_segment_XYZ,this_range_segmentCOM_fr_joint_XYZ,this_range_totalCOM_frame_XYZ = self.slice_data_arrays_by_range(
            self.num_frame_range,mediapipe_pose_data, mediapipeSkelcoordinates_frame_segment_joint_XYZ, segmentedCOM_frame_joint_XYZ, totalCOM_frame_XYZ)



        print('Loading video frames to plot from {}'.format(syncedVideoPath))
        cap = self.load_video_capture_object(syncedVideoPath)

        if self.static_plot:
            video_frames_to_plot = []
        else:            
            video_frames_to_plot = self.get_video_frames_to_plot(cap,self.num_frame_range)




        print('Video frames loaded')
        return this_range_mp_pose_XYZ,this_range_mp_skeleton_segment_XYZ,this_range_segmentCOM_fr_joint_XYZ,this_range_totalCOM_frame_XYZ, video_frames_to_plot


        f= 2


    def generate_plot(self,this_range_mp_pose_XYZ,this_range_mp_skeleton_segment_XYZ,this_range_segmentCOM_fr_joint_XYZ,this_range_totalCOM_frame_XYZ, video_frames_to_plot):
  

        figure = plt.figure(figsize= (5,10))
        # figure.suptitle('FreeMoCap/Mediapipe Center of Mass (COM)', fontsize = 16, y = .94, color = 'royalblue')
 
        self.ax_range = 900
        self.ax_com_range= 300


        ax = figure.add_subplot(3,2,(1,4), projection = '3d')
        ax2 = figure.add_subplot(325)
        # ax3 = figure.add_subplot(322)
        # ax4 = figure.add_subplot(326)
        # ax5 = figure.add_subplot(326)
        
        l1, b1, w1, h1 = ax.get_position().bounds
        ax.set_position([l1-l1*.72,b1+b1*.1,w1,h1])
        l2, b2, w2, h2 = ax2.get_position().bounds
        ax2.set_position([l2*1.8,b2+b2*.95,w2,h2])
        # l3, b3, w3, h3 = ax3.get_position().bounds
        # ax3.set_position([l3,b3-b3*.15,w3,h3])
        # l4, b4, w4, h4 = ax4.get_position().bounds
        # ax4.set_position([l4+l4*.1,b4+b4*2,w4,h4*.5])
        # l5, b5, w5, h5 = ax5.get_position().bounds
        # ax5.set_position([l5+l5*.1,b5+b5*.5,w5,h5*.5])
   

        self.ax = ax
        self.ax2 = ax2
        # self.ax3 = ax3
        # self.ax4 = ax4
        # self.ax5 = ax5

        # ax3.axis('off')


    
        skel_x = this_range_mp_pose_XYZ[:,:,0]
        skel_y = this_range_mp_pose_XYZ[:,:,1]
        skel_z = this_range_mp_pose_XYZ[:,:,2]

        self.left_foot_position_XYZ, self.left_foot_avg_position_XYZ = self.get_foot_data(skel_x,skel_y,skel_z,29,31)
        self.right_foot_position_XYZ, self.right_foot_avg_position_XYZ = self.get_foot_data(skel_x,skel_y,skel_z,30,32)

        self.front_foot_avg_position_XYZ,self.back_foot_avg_position_XYZ = self.get_anterior_posterior_bounds(self.left_foot_position_XYZ,self.right_foot_position_XYZ)

        num_frames_to_plot = range(self.num_frame_range[-1] - self.num_frame_range[0])
        num_frame_length = len(num_frames_to_plot)

        #axes range for the 3D skeleton plot 
        self.mx = -1*np.nanmean(skel_x[int(num_frame_length/2),:])
        self.my = -1*np.nanmean(skel_y[int(num_frame_length/2),:])
        self.mz = np.nanmean(skel_z[int(num_frame_length/2),:])

        #axes range for the COM trajectory plot
        self.mx_com = np.nanmean(-1*this_range_totalCOM_frame_XYZ[:,0])
        self.my_com = np.nanmean(-1*this_range_totalCOM_frame_XYZ[:,1]) #negative because we flip the display in the plot

        #to make x-axis range for the scrolling plot  
        time_array = []
        for frame_num in num_frames_to_plot:
            time_array.append(frame_num/self.camera_fps)
        self.time_array = time_array


        self.skel_x = skel_x
        self.skel_y = skel_y
        self.skel_z = skel_z

        self.this_range_mp_pose_XYZ = this_range_mp_pose_XYZ
        self.this_range_mp_skeleton_segment_XYZ = this_range_mp_skeleton_segment_XYZ
        self.this_range_segmentCOM_fr_joint_XYZ = this_range_segmentCOM_fr_joint_XYZ
        self.this_range_totalCOM_frame_XYZ = this_range_totalCOM_frame_XYZ
        
        self.img_artist = None #used to speed up the video plotting


        print('Starting Frame Animation') 
        ani = FuncAnimation(figure, self.animate, frames= num_frames_to_plot, interval=.1, repeat=False, fargs = (video_frames_to_plot,), init_func= self.animation_init)
        writervideo = animation.FFMpegWriter(fps=self.output_video_fps)
        ani.save(self.this_freemocap_session_path/'eo_sg_freemocap_{}.mp4'.format(self.stance), writer=writervideo)
        print('Animation has been saved to {}'.format(self.this_freemocap_session_path))
        f=2


        f = 2

if __name__ == '__main__':
    this_computer_name = socket.gethostname()
    print(this_computer_name)


    if this_computer_name == 'DESKTOP-V3D343U':
        freemocap_validation_data_path = Path(r"I:\My Drive\HuMoN_Research_Lab\FreeMoCap_Stuff\FreeMoCap_Balance_Validation\data")
    elif this_computer_name == 'DESKTOP-F5LCT4Q':
        #freemocap_validation_data_path = Path(r"C:\Users\aaron\Documents\HumonLab\Spring2022\ValidationStudy\FreeMocap_Data")
        #freemocap_validation_data_path = Path(r'D:\freemocap2022\FreeMocap_Data')
        freemocap_validation_data_path = Path(r'D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3')
    else:
        #freemocap_validation_data_path = Path(r"C:\Users\kiley\Documents\HumonLab\SampleFMC_Data\FreeMocap_Data-20220216T173514Z-001\FreeMocap_Data")
        freemocap_validation_data_path = Path(r"C:\Users\Rontc\Documents\HumonLab\ValidationStudy")
    #sessionID = 'sesh_2022-05-03_13_43_00_JSM_treadmill_day2_t0' #name of the sessionID folder
    sessionID = 'sesh_2023-05-17_14_40_56_MDN_NIH_Trial2'
    #sessionID = 'gopro_sesh_2022-05-24_16_02_53_JSM_T1_BOS'

    stance = 'natural'

    if stance == 'natural':
        #num_frame_range = range(9500,12000)
        #num_frame_range = range(500,1600) #for BOS
        #num_frame_range = range(450, 1300) #for go pro natural 
        #num_frame_range = range(503,1353) #for webcam natural
        num_frame_range = range(1000,2500)
        #num_frame_range = range(4500,6800)

    elif stance == 'left_leg':
        num_frame_range = range(13000,15180)
        num_frame_range = range(5500,6670)

    elif stance == 'right_leg':
        #num_frame_range = range(16000,17450)
        #num_frame_range = range(5400,6620) #gopro
        num_frame_range = range(5453,6673)

    #num_frame_range = range(10000,10100)

    #num_frame_range = range(16680,17740)
    #sessionID_list = ['sesh_2022-05-24_16_02_53_JSM_T1_NIH', 'gopro_sesh_2022-05-24_16_02_53_JSM_T1_NIH']
    # sessionID_list = [gopro_sesh_2022-05-24_16_02_53_JSM_T1_WalkRun']
    # num_frame_range_list = [range(0,1550),range(53,2003)]

    # for sessionID,num_frame_range in zip(sessionID_list,num_frame_range_list):

    camera_fps = 30
    output_video_fps = 90
    tail_length = 120 #number of frames to keep the COM trajectory tail 
    #num_frame_range = 0

    COM_plot = skeleton_COM_Plot(freemocap_validation_data_path,sessionID,num_frame_range, camera_fps, output_video_fps, tail_length,stance,static_plot=False)

    this_range_mp_pose_XYZ,this_range_mp_skeleton_segment_XYZ,this_range_segmentCOM_fr_joint_XYZ,this_range_totalCOM_frame_XYZ, video_frames_to_plot = COM_plot.set_up_data()
    COM_plot.generate_plot(this_range_mp_pose_XYZ,this_range_mp_skeleton_segment_XYZ,this_range_segmentCOM_fr_joint_XYZ,this_range_totalCOM_frame_XYZ, video_frames_to_plot)

