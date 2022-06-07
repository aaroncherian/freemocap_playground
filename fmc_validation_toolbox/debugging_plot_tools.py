import numpy as np
import matplotlib.pyplot as plt
import socket

from pathlib import Path

from requests import session
import scipy.io as sio

def set_axes_ranges(plot_ax,skeleton_data,ax_range):

    mx = np.nanmean(skeleton_data[:,0])
    my = np.nanmean(skeleton_data[:,1])
    mz = np.nanmean(skeleton_data[:,2])

    plot_ax.set_xlim(mx-ax_range,mx+ax_range)
    plot_ax.set_ylim(my-ax_range,my+ax_range)
    plot_ax.set_zlim(mz-ax_range,mz+ax_range)


def plot_3D_skeleton(skeleton_data, frame_to_plot):
    this_frame_skeleton_data = skeleton_data[frame_to_plot,:,:]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(this_frame_skeleton_data[:,0],this_frame_skeleton_data[:,1],this_frame_skeleton_data[:,2], c='r')

    plt.show()

    set_axes_ranges(ax,this_frame_skeleton_data,1000)

    plt.show()

    f = 2


if __name__ == '__main__':
    
    this_computer_name = socket.gethostname()

    if this_computer_name == 'DESKTOP-V3D343U':
        freemocap_validation_data_path = Path(r"I:\My Drive\HuMoN_Research_Lab\FreeMoCap_Stuff\FreeMoCap_Balance_Validation\data")
    elif this_computer_name == 'DESKTOP-F5LCT4Q':
        #freemocap_validation_data_path = Path(r"C:\Users\aaron\Documents\HumonLab\Spring2022\ValidationStudy\FreeMocap_Data")
        #freemocap_validation_data_path = Path(r'D:\freemocap2022\FreeMocap_Data')
        freemocap_data_folder_path = Path(r'D:\ValidationStudy2022\FreeMocap_Data')
    else:
        #freemocap_validation_data_path = Path(r"C:\Users\kiley\Documents\HumonLab\SampleFMC_Data\FreeMocap_Data-20220216T173514Z-001\FreeMocap_Data")
        freemocap_validation_data_path = Path(r"C:\Users\Rontc\Documents\HumonLab\ValidationStudy")


    session_info = {'sessionID': 'qualisys_sesh_2022-05-24_16_02_53_JSM_T1_BOS', 'skeleton_type': 'qualisys'} #name of the sessionID folder
    sessionID = session_info['sessionID']
    skeleton_type_to_use = session_info['skeleton_type']

    this_freemocap_session_path = freemocap_data_folder_path / sessionID

    this_freemocap_data_path = this_freemocap_session_path/'DataArrays'
    if skeleton_type_to_use == 'qualisys':
        skeleton_data_path = this_freemocap_data_path/'qualisys_skel_3D.mat'
        qualysis_mat_file = sio.loadmat(skeleton_data_path)
        skeleton_data = qualysis_mat_file['mat_data_reshaped']
    
    plot_3D_skeleton(skeleton_data,40000)
