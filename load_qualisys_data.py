from pathlib import Path
import numpy as np

from freemocap_utils.freemocap_data_loader import FreeMoCapDataLoader
from freemocap_utils.mediapipe_skeleton_builder import mediapipe_indices

import matplotlib.pyplot as plt
from scipy import signal

import pandas as pd

qualisys_indices = [
'head',
'left_ear',
'right_ear',
'cspine',
'left_shoulder',
'right_shoulder',
'left_elbow',
'right_elbow',
'left_wrist',
'right_wrist',
'left_index',
'right_index',
'left_hip',
'right_hip',
'left_knee',
'right_knee',
'left_ankle',
'right_ankle',
'left_heel',
'right_heel',
'left_foot_index',
'right_foot_index',
]


joint_to_plot = 'left_shoulder'
qualisys_joint_index = qualisys_indices.index(joint_to_plot)
mediapipe_joint_index = mediapipe_indices.index(joint_to_plot)


#path_to_qualisys_session_folder = Path(r"D:\ValidationStudy_numCams\FreeMoCap_Data\qualisys_sesh_2022-05-24_16_02_53_JSM_T1_WalkRun")
# path_to_qualisys_session_folder = Path(r"D:\ValidationStudy2022\FreeMocap_Data\qualisys_sesh_2022-05-24_16_02_53_JSM_T1_BOS")
path_to_qualisys_session_folder = Path(r"D:\ValidationStudy2022\FreeMocap_Data\qualisys_sesh_2022-05-24_16_02_53_JSM_T1_NIH")
qualisys_data = np.load(path_to_qualisys_session_folder/'DataArrays'/'downsampled_qualisys_3D.npy')

#path_to_freemocap_session_folder = Path(r'D:\ValidationStudy_numCams\FreeMoCap_Data\sesh_2022-05-24_16_10_46_JSM_T1_WalkRun')
# path_to_freemocap_session_folder = Path(r'D:\ValidationStudy2022\FreeMocap_Data\sesh_2022-05-24_15_55_40_JSM_T1_BOS')
path_to_freemocap_session_folder = Path(r'D:\ValidationStudy2022\FreeMocap_Data\sesh_2022-05-24_16_02_53_JSM_T1_NIH')
freemocap_data = np.load(path_to_freemocap_session_folder/'DataArrays'/'mediaPipeSkel_3d_origin_aligned.npy')


#samples = qualisys_data.shape[0]
q = 10
#samples_decimated = int(samples/q)


#this_joint_qual = qualisys_data[:,qualisys_joint_index,0]

#qualisys_timeseries = signal.decimate(this_joint_qual,10)
qualisys_timeseries = qualisys_data[:,qualisys_joint_index,0]
freemocap_timeseries = freemocap_data[:,mediapipe_joint_index,0]

if qualisys_timeseries.shape[0] > freemocap_timeseries.shape[0]:
    frame_length_to_equalize = freemocap_timeseries.shape[0]

elif freemocap_timeseries.shape[0] > qualisys_timeseries.shape[0]:
    frame_length_to_equalize = qualisys_timeseries.shape[0]

else:
    frame_length_to_equalize = qualisys_timeseries.shape[0]

#qualisys_timeseries = qualisys_timeseries/np.max(qualisys_timeseries)
#freemocap_timeseries = freemocap_timeseries/np.max(freemocap_timeseries)

#https://medium.com/@dreamferus/how-to-synchronize-time-series-using-cross-correlation-in-python-4c1fd5668c7a

def shift_for_maximum_correlation(x, y):
    correlation = signal.correlate(x, y, mode="full")
    lags = signal.correlation_lags(x.size, y.size, mode="full")
    lag = lags[np.argmax(correlation)]
    print(f"Best lag: {lag}")
    if lag < 0:
        y = y.iloc[abs(lag):].reset_index(drop=True)
        print(f'Using FreeMoCap from {abs(lag)} onwards')
    else:
        x = x.iloc[lag:].reset_index(drop=True)
        print(f'Using Qualisys from {abs(lag)} onwards')
    x = x - x.iloc[0]
    y = y - y.iloc[0]
    return x, y
    
def correlation(x, y):
    shortest = min(x.shape[0], y.shape[0])
    return np.corrcoef(x.iloc[:shortest].values, y.iloc[:shortest].values)[0, 1]

def plot_correlation(x, y, text):
    # plot 
    fig,ax = plt.subplots(figsize=(10, 6))
    ax.plot(x,label = 'qualisys')
    ax.plot(y,label = 'freemocap')

    #plt.title(f"Correlation {text}: {correlation(x, y)}")
    plt.legend(loc="best")
    plt.show()


shifted_x, shifted_y = shift_for_maximum_correlation(pd.DataFrame(qualisys_timeseries), pd.DataFrame(freemocap_timeseries))
plot_correlation(shifted_x, shifted_y, text="after shifting")
f = 2

##---scipy time correlation
# correlation = signal.correlate(qualisys_timeseries, freemocap_timeseries, mode="full")
# lags = signal.correlation_lags(qualisys_timeseries.size,freemocap_timeseries.size, mode="full")
# lag = lags[np.argmax(correlation)]

# max_qual_index = np.where(qualisys_timeseries == np.max(qualisys_timeseries))[0][0]
# matching_mp_index = max_qual_index + lag

# figure = plt.figure(figsize= (10,10))

# ax1 = figure.add_subplot(221)
# ax2 = figure.add_subplot(222)
# ax3 = figure.add_subplot(223)

# ax1.axvline(matching_mp_index, color='r', linestyle='--')
# ax1.plot(freemocap_timeseries, label = 'freemocap')


# ax2.axvline(max_qual_index, color = 'r', linestyle = '--')
# ax2.plot(qualisys_timeseries, label = 'qualisys')

# ax3.plot(correlation, label = 'correlation')
# plt.show()



figure = plt.figure()

freemocap_ax = figure.add_subplot(211)
qualisys_ax = figure.add_subplot(212)

freemocap_ax.plot(freemocap_timeseries[1157:6400])
freemocap_ax.plot(qualisys_timeseries[0:5500], color = 'orange')

# freemocap_ax.set_ylim([-300,100])
# qualisys_ax.set_ylim([-300,100])


plt.show()

f = 2