from scipy import signal
from numpy.random import default_rng
import numpy as np
import matplotlib.pyplot as plt

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

from mediapipe_skeleton_builder import mediapipe_indices

from scipy.signal import find_peaks, argrelextrema, savgol_filter
from scipy.fft import fft, fftfreq


this_computer_name = socket.gethostname()
print(this_computer_name)







def get_time_sync_lag(session_one_info, session_two_info, freemocap_data_folder_path):

    sessionID_one = session_one_info['sessionID']
    sessionID_two = session_two_info['sessionID']
    
    session_one_data_path = freemocap_data_folder_path / sessionID_one / 'DataArrays'/ 'mediaPipeSkel_3d.npy'
    session_two_data_path = freemocap_data_folder_path / sessionID_two / 'DataArrays'/ 'qualisys_origin_aligned_skeleton_3D.npy'
    #session_two_data_path = freemocap_data_folder_path / sessionID_two / 'DataArrays'/'mediapipe_origin_aligned_skeleton_3D.npy'


    session_one_mediapipe_data = np.load(session_one_data_path)
    session_two_mediapipe_data = np.load(session_two_data_path)

    session_two_mediapipe_data = session_two_mediapipe_data[0:len(session_two_mediapipe_data):5,:,:]

    if session_one_mediapipe_data.shape[0] > session_two_mediapipe_data.shape[0]:
        frame_length_to_equalize = session_two_mediapipe_data.shape[0]

    elif session_two_mediapipe_data.shape[0] > session_one_mediapipe_data.shape[0]:
        frame_length_to_equalize = session_one_mediapipe_data.shape[0]

    else:
        frame_length_to_equalize = session_one_mediapipe_data.shape[0]

    left_shoulder_index = mediapipe_indices.index('left_shoulder')

    session_one_left_shoulder = savgol_filter(session_one_mediapipe_data[0:frame_length_to_equalize,left_shoulder_index,0], 51, 3)
    session_two_left_shoulder = savgol_filter(session_two_mediapipe_data[0:frame_length_to_equalize:,left_shoulder_index,0],51,3)

    session_one_left_shoulder = session_one_left_shoulder/np.max(session_one_left_shoulder)
    session_two_left_shoulder = session_two_left_shoulder/np.max(session_two_left_shoulder)



    correlation = signal.correlate(session_one_left_shoulder, session_two_left_shoulder, mode="full")
    lags = signal.correlation_lags(session_one_left_shoulder.size,session_two_left_shoulder.size, mode="full")
    lag = lags[np.argmax(correlation)]

    #max_qual_index = np.where(session_two_left_shoulder == np.max(session_two_left_shoulder))[0][0]
    max_qual_index = 8170

    matching_mp_index = max_qual_index + lag

    return lag 

# figure = plt.figure(figsize= (10,10))

# ax1 = figure.add_subplot(221)
# ax2 = figure.add_subplot(222)
# ax3 = figure.add_subplot(223)

# ax1.axvline(matching_mp_index, color='r', linestyle='--')
# ax1.plot(session_one_left_shoulder, label = 'session one')


# ax2.axvline(max_qual_index, color = 'r', linestyle = '--')
# ax2.plot(session_two_left_shoulder, label = 'session two')

# ax3.plot(correlation, label = 'correlation')
# plt.show()

# rng = default_rng()
# x = rng.standard_normal(1000)
# y = np.concatenate([rng.standard_normal(100), x])
# correlation = signal.correlate(x, y, mode="full")
# correlation = correlation/np.max(correlation)
# lags = signal.correlation_lags(x.size, y.size, mode="full")
# lag = lags[np.argmax(correlation)]


# figure = plt.figure(figsize= (10,10))

# ax1 = figure.add_subplot(221)
# ax2 = figure.add_subplot(222)
# ax3 = figure.add_subplot(223)
# ax4 = figure.add_subplot(224)


# ax1.plot(x)
# ax2.plot(y)
# ax3.plot(correlation)

if __name__ == '__main__':

    if this_computer_name == 'DESKTOP-V3D343U':
        freemocap_validation_data_path = Path(r"I:\My Drive\HuMoN_Research_Lab\FreeMoCap_Stuff\FreeMoCap_Balance_Validation\data")
    elif this_computer_name == 'DESKTOP-F5LCT4Q':
        #freemocap_validation_data_path = Path(r"C:\Users\aaron\Documents\HumonLab\Spring2022\ValidationStudy\FreeMocap_Data")
        #freemocap_validation_data_path = Path(r'D:\freemocap2022\FreeMocap_Data')
        freemocap_data_folder_path = Path(r'D:\ValidationStudy2022\FreeMocap_Data')
    else:
        #freemocap_validation_data_path = Path(r"C:\Users\kiley\Documents\HumonLab\SampleFMC_Data\FreeMocap_Data-20220216T173514Z-001\FreeMocap_Data")
        freemocap_validation_data_path = Path(r"C:\Users\Rontc\Documents\HumonLab\ValidationStudy")

    
    session_one_info = {'sessionID': 'session_SER_1_20_22', 'skeleton_type':'mediapipe'} #name of the sessionID folder
    
    session_two_info = {'sessionID': 'session_SER_1_20_22', 'skeleton_type': 'qualisys'}

    lag = get_time_sync_lag(session_one_info, session_two_info, freemocap_data_folder_path)

    f = 2
    #sessionID_one = 'sesh_2022-05-24_16_02_53_JSM_T1_NIH'
    #sessionID_one = 'sesh_2022-05-24_15_55_40_JSM_T1_BOS' 

    #sessionID_two = 'gopro_sesh_2022-05-24_16_02_53_JSM_T1_NIH'
    #sessionID_two = 'gopro_sesh_2022-05-24_16_02_53_JSM_T1_BOS'




#plt.show()
f=2