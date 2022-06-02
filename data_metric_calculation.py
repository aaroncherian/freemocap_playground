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

class pathLengthDataHolder:
    def __init__(self,skeleton_data,num_frame_range):
        self.skeleton_data = skeleton_data
        self.num_frame_range = num_frame_range

        self.sliced_skeleton_data = self.slice_skeleton_data(self.skeleton_data, self.num_frame_range)
        self.path_length = self.calculate_path_length(self.sliced_skeleton_data)

    def slice_skeleton_data(self, skeleton_data, num_frame_range):
        sliced_skeleton_data = skeleton_data[num_frame_range[0]:num_frame_range[-1],:]
        return sliced_skeleton_data

    def calculate_path_length(self, skeleton_data):
        path_length = 0
        for i in range(1,len(skeleton_data)):
            path_length += self.calculate_distance(skeleton_data[i-1],skeleton_data[i])
        return path_length

    def calculate_distance(self, point1, point2):
        return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2 + (point1[2]-point2[2])**2)

    def update_frame_range(self,num_frame_range):
        self.num_frame_range = num_frame_range
        self.sliced_skeleton_data = self.slice_skeleton_data(self.skeleton_data, self.num_frame_range)
        #self.path_length = self.calculate_path_length(self.sliced_skeleton_data)

    def get_path_length(self):
        self.path_length = self.calculate_path_length(self.sliced_skeleton_data)
        return self.path_length 

this_computer_name = socket.gethostname()
print(this_computer_name)

if this_computer_name == 'DESKTOP-V3D343U':
    freemocap_validation_data_path = Path(r"I:\My Drive\HuMoN_Research_Lab\FreeMoCap_Stuff\FreeMoCap_Balance_Validation\data")
elif this_computer_name == 'DESKTOP-F5LCT4Q':
    #freemocap_validation_data_path = Path(r"C:\Users\aaron\Documents\HumonLab\Spring2022\ValidationStudy\FreeMocap_Data")
    freemocap_validation_data_path = Path(r'D:\ValidationStudy2022\FreeMocap_Data')
else:
    #freemocap_validation_data_path = Path(r"C:\Users\kiley\Documents\HumonLab\SampleFMC_Data\FreeMocap_Data-20220216T173514Z-001\FreeMocap_Data")
    freemocap_validation_data_path = Path(r"C:\Users\Rontc\Documents\HumonLab\ValidationStudy")

go_pro_sessionID = 'gopro_sesh_2022-05-24_16_02_53_JSM_T1_NIH' #name of the sessionID folder

webcam_sessionID = 'sesh_2022-05-24_16_02_53_JSM_T1_NIH'


this_go_pro_session_path = freemocap_validation_data_path / go_pro_sessionID

this_webcam_session_path = freemocap_validation_data_path / webcam_sessionID

this_go_pro_data_path = this_go_pro_session_path /'DataArrays'
this_webcam_data_path = this_webcam_session_path/'DataArrays'

go_pro_data = np.load(this_go_pro_data_path/'origin_aligned_totalBodyCOM_frame_XYZ.npy')
webcam_data = np.load(this_webcam_data_path/'origin_aligned_totalBodyCOM_frame_XYZ.npy')


go_pro_num_frame_range_eyesopen_flatground = range(650,2320)
webcam_num_frame_range_eyesopen_flatground = range(685,2335)

go_pro_num_frame_range_eyesclosed_flatground = range(2690,4490)
webcam_num_frame_range_eyesclosed_flatground = range(2725,4525)

go_pro_num_frame_range_eyesopen_foam = range(5925,7715)
webcam_num_frame_range_eyesopen_foam = range(5960,7760)

go_pro_num_frame_range_eyesclosed_foam =  range(8315,10050)
webcam_num_frame_range_eyesclosed_foam = range(8340,10085)

go_pro_skeletonCOM_holder = pathLengthDataHolder(go_pro_data,go_pro_num_frame_range_eyesopen_flatground)
go_pro_COM_path_length_eyesopen_flatground  = go_pro_skeletonCOM_holder.get_path_length()

webcam_skeletonCOM_holder = pathLengthDataHolder(webcam_data,go_pro_num_frame_range_eyesopen_flatground)
webcam_COM_path_length_eyesopen_flatground = webcam_skeletonCOM_holder.get_path_length()

print('Eyes Open/Flat Ground:\n', 'GoPro: ', go_pro_COM_path_length_eyesopen_flatground, 'Webcam:', webcam_COM_path_length_eyesopen_flatground)

webcam_skeletonCOM_holder.update_frame_range(webcam_num_frame_range_eyesclosed_flatground)
webcam_COM_path_length_eyesclosed_flatground = webcam_skeletonCOM_holder.get_path_length()

go_pro_skeletonCOM_holder.update_frame_range(go_pro_num_frame_range_eyesclosed_flatground)
go_pro_COM_path_length_eyesclosed_flatground = go_pro_skeletonCOM_holder.get_path_length()

print('Eyes Closed/Flat Ground:\n', 'GoPro: ', go_pro_COM_path_length_eyesclosed_flatground, 'Webcam:', webcam_COM_path_length_eyesclosed_flatground)

webcam_skeletonCOM_holder.update_frame_range(webcam_num_frame_range_eyesopen_foam)
webcam_COM_path_length_eyesopen_foam = webcam_skeletonCOM_holder.get_path_length()

go_pro_skeletonCOM_holder.update_frame_range(go_pro_num_frame_range_eyesopen_foam)
go_pro_COM_path_length_eyesopen_foam = go_pro_skeletonCOM_holder.get_path_length()

print('Eyes Open/Foam:\n', 'GoPro: ', go_pro_COM_path_length_eyesopen_foam, 'Webcam:', webcam_COM_path_length_eyesopen_foam)

webcam_skeletonCOM_holder.update_frame_range(webcam_num_frame_range_eyesclosed_foam)
webcam_COM_path_length_eyesclosed_foam = webcam_skeletonCOM_holder.get_path_length()

go_pro_skeletonCOM_holder.update_frame_range(go_pro_num_frame_range_eyesclosed_foam)
go_pro_COM_path_length_eyesclosed_foam = go_pro_skeletonCOM_holder.get_path_length()

print('Eyes Closed/Foam:\n', 'GoPro: ', go_pro_COM_path_length_eyesclosed_foam, 'Webcam:', webcam_COM_path_length_eyesclosed_foam)



figure = plt.figure()

ax1 = figure.add_subplot()


webcam_frame_range = range(0,len(webcam_data))
gopro_frame_range = range(0,len(go_pro_data)+0)

ax1.plot(np.diff(go_pro_data[:,0]),'b')
ax1.plot(np.diff(webcam_data[:,0]),'r')
plt.show()

f = 2
#mediapipe_COM_data = np.load(this_freemocap_data_path/'totalBodyCOM_frame_XYZ.npy')

#qualisys_COM_data = np.load(this_freemocap_data_path/'qualisys_totalBodyCOM_frame_XYZ.npy')

#calculate stdev of X,Y,Z for mediapipe and qualisys data

# med_start_frame = 16680
# med_end_frame = 17740

# qual_start_frame = 94290
# qual_end_frame = 99685


# mediapipe_COM_XYZ = mediapipe_COM_data[med_start_frame:med_end_frame,:]
# mediapipe_COM_x = mediapipe_COM_data[med_start_frame:med_end_frame,0]
# mediapipe_COM_y = mediapipe_COM_data[med_start_frame:med_end_frame,1]
# mediapipe_COM_z = mediapipe_COM_data[med_start_frame:med_end_frame,2]

# qualisys_COM_XYZ = qualisys_COM_data[qual_start_frame:qual_end_frame,:]
# qualisys_COM_data_x = qualisys_COM_data[qual_start_frame:qual_end_frame,0]
# qualisys_COM_data_y = qualisys_COM_data[qual_start_frame:qual_end_frame,1]
# qualisys_COM_data_z = qualisys_COM_data[qual_start_frame:qual_end_frame,2]



# mediapipe_COM_x_stdev = np.std(mediapipe_COM_x)
# mediapipe_COM_y_stdev = np.std(mediapipe_COM_y)
# mediapipe_COM_z_stdev = np.std(mediapipe_COM_z)

# qualisys_COM_x_stdev = np.std(qualisys_COM_data_x)
# qualisys_COM_y_stdev = np.std(qualisys_COM_data_y)
# qualisys_COM_z_stdev = np.std(qualisys_COM_data_z)

# figure = plt.figure()

# ax1 = figure.add_subplot(1,2,1)
# ax2 = figure.add_subplot(1,2,2)

# ax1.plot(mediapipe_COM_x)
# ax1.set_title('Mediapipe COM Y')

# ax2.plot(qualisys_COM_data_x)
# ax2.set_title('Qualisys COM Y')

# plt.show()


# def calculate_path_length(data):
#     path_length = 0
#     for i in range(1,len(data)):
#         path_length += calculate_distance(data[i-1],data[i])
#     return path_length

# def calculate_distance(point1, point2):
#     return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2 + (point1[2]-point2[2])**2)


# mp_path_length = calculate_path_length(mediapipe_COM_XYZ)

# q_path_length = calculate_path_length(qualisys_COM_XYZ)


# f=2

#calculate distance between 3D COM point at each frame and add those distances together

