# from plot_with_classes import skeleton_COM_Plot
# from qualisys_class_plotting import skeleton_COM_Plot



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


from numpy import NaN, Inf, arange, isscalar, asarray, array

def peakdet(v, delta, x = None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    
    Returns two arrays
    
    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %      
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.
    
    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.
    
    """
    maxtab = []
    mintab = []
       
    if x is None:
        x = arange(len(v))
    
    v = asarray(v)
    
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    
    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    
    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN
    
    lookformax = True
    
    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return array(maxtab), array(mintab)


f=2

this_computer_name = socket.gethostname()
print(this_computer_name)


if this_computer_name == 'DESKTOP-V3D343U':
    freemocap_validation_data_path = Path(r"I:\My Drive\HuMoN_Research_Lab\FreeMoCap_Stuff\FreeMoCap_Balance_Validation\data")
elif this_computer_name == 'DESKTOP-F5LCT4Q':
    #freemocap_validation_data_path = Path(r"C:\Users\aaron\Documents\HumonLab\Spring2022\ValidationStudy\FreeMocap_Data")
    #freemocap_validation_data_path = Path(r'D:\freemocap2022\FreeMocap_Data')
    freemocap_validation_data_path = Path(r'D:\ValidationStudy2022\FreeMocap_Data')
else:
    #freemocap_validation_data_path = Path(r"C:\Users\kiley\Documents\HumonLab\SampleFMC_Data\FreeMocap_Data-20220216T173514Z-001\FreeMocap_Data")
    freemocap_validation_data_path = Path(r"C:\Users\Rontc\Documents\HumonLab\ValidationStudy")

#sessionID = 'session_SER_1_20_22' #name of the sessionID folder
sessionID_one = 'gopro_sesh_2022-05-24_16_02_53_JSM_T1_BOS'

sessionID_two = 'sesh_2022-05-24_15_55_40_JSM_T1_BOS'

session_one_data_path = freemocap_validation_data_path / sessionID_one / 'DataArrays'/ 'mediapipe_origin_aligned_skeleton_3D.npy'
session_two_data_path = freemocap_validation_data_path / sessionID_two / 'DataArrays'/'mediapipe_origin_aligned_skeleton_3D.npy'


session_one_medipipe_data = np.load(session_one_data_path)
session_two_medipipe_data = np.load(session_two_data_path)


left_shoulder_index = mediapipe_indices.index('left_shoulder')

session_one_left_shoulder = savgol_filter(session_one_medipipe_data[0:6620,left_shoulder_index,0], 51, 3)
session_two_left_shoulder = savgol_filter(session_two_medipipe_data[0:6620:,left_shoulder_index,0],51,3)

#peaks_one, _ = find_peaks(abs(session_one_left_shoulder[0]), height=200)

#mini,maxi = peakdet(session_one_left_shoulder[0], 1)

maximums_one = argrelextrema(session_one_left_shoulder, np.greater)

maximums_two = argrelextrema(session_two_left_shoulder, np.greater)

frame_window = range(1556,2223)

maximums_one_window = [x for x in maximums_one[0] if x in frame_window]

maximums_two_window = [x for x in maximums_two[0] if x in frame_window]

#distance, path = fastdtw(session_one_left_shoulder, session_two_left_shoulder, dist = euclidean)



maximums_difference = np.sort([abs(y -x) for x,y in zip(maximums_one_window, maximums_two_window)])
# A = fft(session_two_left_shoulder[0:6620])

median = np.median(maximums_difference)



x_range = range(0,len(session_one_left_shoulder))
x_range_altered = [x + median for x in x_range]

maximums_one_altered = [int(x + median) for x in maximums_one[0]]

# N = 6620
# freq = fftfreq(N)

#B = fftpack.fft(session_two_left_shoulder[0:6620])

#c = scipy.ifft(A * scipy.conj(B))
#time_shift = np.argmax(abs(c))
f = 2


figure = plt.figure(figsize= (10,10))


#figure.suptitle(title_text, fontsize = 16, y = .94)



ax1 = figure.add_subplot(211)
ax2 = figure.add_subplot(212)

ax1.set_title('Unsynced Webcam/GoPro Data')
ax1.plot(x_range,session_one_left_shoulder, 'r', label = 'webcam data')
ax1.plot(maximums_one[0], session_one_left_shoulder[maximums_one], 'ro')
ax1.plot(x_range,session_two_left_shoulder, 'b',)
ax1.plot(maximums_two[0], session_two_left_shoulder[maximums_two], 'bo', label = 'go pro data')

ax1.legend()

ax2.set_title('Synced Webcam/GoPro Data, shift = {}'.format(median))
ax2.plot(x_range_altered,session_one_left_shoulder, 'r',)
ax2.plot(x_range,session_two_left_shoulder, 'b',)
ax2.plot(maximums_two[0], session_two_left_shoulder[maximums_two], 'bo')
ax2.plot(maximums_one[0] + median, session_one_left_shoulder[maximums_one], 'ro')
#ax2.plot(freq,A)

plt.show()



f = 2




#sessionID = 'sesh_2022-05-03_13_43_00_JSM_treadmill_day2_t0'





