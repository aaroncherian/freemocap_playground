import numpy as np
import matplotlib.pyplot as plt
import socket

from pathlib import Path

import scipy.io as sio

from skeleton_data_holder import SkeletonDataHolder


def plot_all_skeletons(raw_skeleton_data,origin_translated_skeleton_data,y_aligned_skeleton_data,spine_aligned_skeleton_data, skeleton_indices, good_frame):
    def create_vector(point1,point2): 
        """Put two points in, make a vector"""
        vector = point2 - point1
        return vector

    def plot_origin_vectors(plot_ax,x_vector,y_vector,z_vector,origin):
        Zvector_X,Zvector_Y,Zvector_Z = zip(z_vector*800)
        Xvector_X,Xvector_Y,Xvector_Z = zip(x_vector*800)
        Yvector_X,Yvector_Y,Yvector_Z = zip(y_vector*800)

        Origin_X,Origin_Y,Origin_Z = zip(origin)

        plot_ax.quiver(Origin_X,Origin_Y,Origin_Z,Xvector_X,Xvector_Y,Xvector_Z,arrow_length_ratio=0.1,color='r', label = 'X-axis')
        plot_ax.quiver(Origin_X,Origin_Y,Origin_Z,Yvector_X,Yvector_Y,Yvector_Z,arrow_length_ratio=0.1,color='g', label = 'Y-axis')
        plot_ax.quiver(Origin_X,Origin_Y,Origin_Z,Zvector_X,Zvector_Y,Zvector_Z,arrow_length_ratio=0.1,color='b', label = 'Z-axis')            
    
    def set_axes_ranges(plot_ax,skeleton_data, ax_range):

        mx = np.nanmean(skeleton_data[:,0])
        my = np.nanmean(skeleton_data[:,1])
        mz = np.nanmean(skeleton_data[:,2])
    
        plot_ax.set_xlim(mx-ax_range,mx+ax_range)
        plot_ax.set_ylim(my-ax_range,my+ax_range)
        plot_ax.set_zlim(mz-ax_range,mz+ax_range)        

    def plot_spine_unit_vector(plot_ax,skeleton_mid_hip_XYZ,skeleton_spine_unit_vector):

        skeleton_spine_unit_x, skeleton_spine_unit_y, skeleton_spine_unit_z = zip(skeleton_spine_unit_vector*800)

        plot_ax.quiver(skeleton_mid_hip_XYZ[0],skeleton_mid_hip_XYZ[1],skeleton_mid_hip_XYZ[2], skeleton_spine_unit_x,skeleton_spine_unit_y,skeleton_spine_unit_z,arrow_length_ratio=0.1,color='pink')

    def plot_heel_unit_vector(plot_ax,skeleton_heel_unit_vector, heel_vector_origin_XYZ):

        origin_heel_x, origin_heel_y, origin_heel_z = zip(heel_vector_origin_XYZ)
        skeleton_heel_unit_x, skeleton_heel_unit_y, skeleton_heel_unit_z = zip(skeleton_heel_unit_vector*500)

        plot_ax.quiver(origin_heel_x,origin_heel_y,origin_heel_z, skeleton_heel_unit_x,skeleton_heel_unit_y,skeleton_heel_unit_z,arrow_length_ratio=0.1,color='orange')

    def plot_vectors_for_skeleton(plot_ax,skeleton_data_holder):
        heel_unit_vector = skeleton_data_holder.heel_unit_vector
        heel_vector_origin = skeleton_data_holder.heel_vector_origin
        mid_hip_XYZ = skeleton_data_holder.mid_hip_XYZ
        spine_unit_vector = skeleton_data_holder.spine_unit_vector

        plot_heel_unit_vector(plot_ax,heel_unit_vector,heel_vector_origin)
        plot_spine_unit_vector(plot_ax,mid_hip_XYZ,spine_unit_vector)

    origin = np.array([0, 0, 0])
    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])

    x_vector = create_vector(origin,x_axis)
    y_vector = create_vector(origin,y_axis)
    z_vector = create_vector(origin,z_axis)
    figure = plt.figure()

    ax1 = figure.add_subplot(221,projection = '3d')
    ax2 = figure.add_subplot(222,projection = '3d')
    ax3 = figure.add_subplot(223,projection = '3d')
    ax4 = figure.add_subplot(224,projection = '3d')

    axes_list = [ax1,ax2,ax3,ax4]

    ax1.set_title('Original Skeleton')
    ax2.set_title('Skeleton Translated to Origin')
    ax3.set_title('Skeleton Rotated to Make +Y Forwards')
    ax4.set_title('Skeleton Rotated to Make +Z Up')

    raw_skeleton_holder = SkeletonDataHolder(raw_skeleton_data, skeleton_indices,good_frame)
    origin_translated_skeleton_holder = SkeletonDataHolder(origin_translated_skeleton_data, skeleton_indices,good_frame)
    y_aligned_skeleton_holder = SkeletonDataHolder(y_aligned_skeleton_data, skeleton_indices,good_frame)
    spine_aligned_skeleton_holder = SkeletonDataHolder(spine_aligned_skeleton_data, skeleton_indices,good_frame)

    skeleton_holder_list = [raw_skeleton_holder, origin_translated_skeleton_holder, y_aligned_skeleton_holder,spine_aligned_skeleton_holder]

    ax_range = 1800
    for ax,skeleton_data_holder in zip(axes_list,skeleton_holder_list):
        set_axes_ranges(ax,skeleton_data_holder.good_frame_skeleton_data,ax_range)
        plot_origin_vectors(ax,x_vector,y_vector,z_vector,origin)
        ax.scatter(skeleton_data_holder.good_frame_skeleton_data[:,0], skeleton_data_holder.good_frame_skeleton_data[:,1], skeleton_data_holder.good_frame_skeleton_data[:,2], c = 'r')
        plot_vectors_for_skeleton(ax,skeleton_data_holder)

    ax3.legend()
    ax4.legend()

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
