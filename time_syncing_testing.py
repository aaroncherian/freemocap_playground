
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


path_to_qualisys_session_folder = Path(r"D:\ValidationStudy_numCams\FreeMoCap_Data\qualisys_sesh_2022-05-24_16_02_53_JSM_T1_WalkRun")
#qualisys_data = np.load(path_to_qualisys_session_folder/'DataArrays'/'downsampled_qualisys_3D.npy')
qualisys_data = np.load(path_to_qualisys_session_folder/'DataArrays'/'qualisysSkel_3d.npy')


path_to_freemocap_session_folder = Path(r'D:\ValidationStudy_numCams\FreeMoCap_Data\sesh_2022-05-24_16_10_46_JSM_T1_WalkRun')
freemocap_data = np.load(path_to_freemocap_session_folder/'DataArrays'/'mediaPipeSkel_3d_origin_aligned.npy')

freemocap_data = freemocap_data[1162:-1]



sec = 180

t_qual = np.linspace(0,sec,300*sec)
t_fmc = np.linspace(0,sec,29.970857503650052*sec)


# t_qual = np.linspace(0,qualisys_data.shape[0]/300,qualisys_data.shape[0])

# t_fmc = t_qual[0:-1:freemocap_data.shape[0]]
# np.linspace(0,freemocap_data.shape[0]/30,freemocap_data.shape[0])

# Fs = 300
# Ts = 1/Fs

fig = plt.figure()

ax1 = fig.add_subplot(111)

ax1.plot(t_qual,qualisys_data[0:sec*300,4,0] - qualisys_data[0,4,0])
ax1.plot(t_fmc,freemocap_data[0:sec*30,11,0] - freemocap_data[0,11,0])


plt.show()