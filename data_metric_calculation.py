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
this_freemocap_session_path = freemocap_validation_data_path / sessionID
this_freemocap_data_path = this_freemocap_session_path/'DataArrays'

mediapipe_COM_data = np.load(this_freemocap_data_path/'totalBodyCOM_frame_XYZ.npy')

qualisys_COM_data = np.load(this_freemocap_data_path/'qualisys_totalBodyCOM_frame_XYZ.npy')

#calculate stdev of X,Y,Z for mediapipe and qualisys data

med_start_frame = 2600
med_end_frame = 17400

qual_start_frame = 24400
qual_end_frame = 103000


mediapipe_COM_XYZ = mediapipe_COM_data[med_start_frame:med_end_frame,:]
mediapipe_COM_x = mediapipe_COM_data[med_start_frame:med_end_frame,0]
mediapipe_COM_y = mediapipe_COM_data[med_start_frame:med_end_frame,1]
mediapipe_COM_z = mediapipe_COM_data[med_start_frame:med_end_frame,2]

qualisys_COM_XYZ = qualisys_COM_data[qual_start_frame:qual_end_frame,:]
qualisys_COM_data_x = qualisys_COM_data[qual_start_frame:qual_end_frame,0]
qualisys_COM_data_y = qualisys_COM_data[qual_start_frame:qual_end_frame,1]
qualisys_COM_data_z = qualisys_COM_data[qual_start_frame:qual_end_frame,2]



mediapipe_COM_x_stdev = np.std(mediapipe_COM_x)
mediapipe_COM_y_stdev = np.std(mediapipe_COM_y)
mediapipe_COM_z_stdev = np.std(mediapipe_COM_z)

qualisys_COM_x_stdev = np.std(qualisys_COM_data_x)
qualisys_COM_y_stdev = np.std(qualisys_COM_data_y)
qualisys_COM_z_stdev = np.std(qualisys_COM_data_z)

figure = plt.figure()

ax1 = figure.add_subplot(1,2,1)
ax2 = figure.add_subplot(1,2,2)

ax1.plot(mediapipe_COM_x)
ax1.set_title('Mediapipe COM Y')

ax2.plot(qualisys_COM_data_x)
ax2.set_title('Qualisys COM Y')

#plt.show()


def calculate_path_length(data):
    path_length = 0
    for i in range(1,len(data)):
        path_length += calculate_distance(data[i-1],data[i])
    return path_length

def calculate_distance(point1, point2):
    return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2 + (point1[2]-point2[2])**2)


mp_path_length = calculate_path_length(mediapipe_COM_XYZ)

q_path_length = calculate_path_length(qualisys_COM_XYZ)


f=2

#calculate distance between 3D COM point at each frame and add those distances together