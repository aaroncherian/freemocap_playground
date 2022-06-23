from distutils.log import debug
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
from scipy import optimize
from scipy.spatial.transform import Rotation
import scipy.io as sio

from fmc_validation_toolbox.mediapipe_skeleton_builder import mediapipe_indices, build_mediapipe_skeleton, slice_mediapipe_data
from anthropometry_data_tables import segments, joint_connections, segment_COM_lengths, segment_COM_percentages
from fmc_validation_toolbox.qualisys_skeleton_builder import qualisys_indices, build_qualisys_skeleton

from fmc_validation_toolbox import good_frame_finder

def get_translation_error_between_two_rotation_matrices(translation_guess,segments_list_A, segments_list_B):
    #convert euler angles to rotation matrix
    total_error_list = []
    for segmentA, segmentB in zip(segments_list_A, segments_list_B):

        segmentA_translated_by_guess = segmentA + translation_guess

        #mean_segmentB_point = np.mean(segmentB, axis=0)

        #mean_segmentA_point = np.mean(segmentA_translated_by_guess, axis=0)

        error_list = [abs(y-x) for x,y in zip(segmentA_translated_by_guess,segmentB)]

        this_segment_error = np.mean(error_list)

        total_error_list.append(this_segment_error)

    return total_error_list

def get_optimal_translation_matrix(segments_list_A, segments_list_B):
    translation_matrix = optimize.least_squares(get_translation_error_between_two_rotation_matrices,
                                    [0,0,0], args = (segments_list_A, segments_list_B),
                                    gtol = 1e-10,
                                    verbose = 2).x
    return translation_matrix


def translate_skeleton_frame(skeleton_data_frame, translation_distance):
    """Take in a frame of rotated skeleton data, and apply the translation distance to each point in the skeleton"""
    translated_skeleton_frame = skeleton_data_frame + translation_distance
    #translated_point = [x + y for x,y in zip(skeleton_data_frame, translation_distance)]
    return  translated_skeleton_frame


def get_optimal_rotation_matrix(segments_list_A,segments_list_B):
    euler_angles = optimize.least_squares(get_error_between_two_rotation_matrices,
                                    [0,0,0], args = (segments_list_A, segments_list_B),
                                    gtol = 1e-10,
                                    verbose = 2).x
    return Rotation.from_euler('XYZ',euler_angles).as_matrix()

def get_error_between_two_rotation_matrices(euler_angle_guess, segments_list_A, segments_list_B):

    rotation_matrix_guess = Rotation.from_euler('XYZ',euler_angle_guess).as_matrix()
    total_error_list = []
    for segmentA, segmentB in zip(segments_list_A, segments_list_B):
        #----Attempt 2 for rotation
        segmentA_rotated_by_guess = [rotation_matrix_guess @ x for x in segmentA]

        vectorA_rotated_by_guess = segmentA_rotated_by_guess[1] - segmentA_rotated_by_guess[0]
        vectorB = segmentB[1] - segmentB[0]

        error = abs(np.cross(vectorA_rotated_by_guess,vectorB))

        #error = np.linalg.norm(error)

        total_error_list.append(error)

    mean_error = np.mean(total_error_list, axis = 0)

    return mean_error 

def rotate_point(point,rotation_matrix):
    rotated_point = np.dot(rotation_matrix,point)
    return rotated_point

def rotate_skeleton_frame(this_frame_aligned_skeleton_data, rotation_matrix):
    """Take in a frame of skeleton data, and apply the rotation matrix to each point in the skeleton"""

    this_frame_rotated_skeleton = np.zeros(this_frame_aligned_skeleton_data.shape)  #initialize the array to hold the rotated skeleton data for this frame
    num_tracked_points = this_frame_aligned_skeleton_data.shape[0]

    for i in range(num_tracked_points):
        this_frame_rotated_skeleton[i,:] = rotate_point(this_frame_aligned_skeleton_data[i,:],rotation_matrix)

    return this_frame_rotated_skeleton


this_computer_name = socket.gethostname()

if this_computer_name == 'DESKTOP-V3D343U':
    freemocap_data_path = Path(r"I:\My Drive\HuMoN_Research_Lab\FreeMoCap_Stuff\FreeMoCap_Balance_Validation\data")
elif this_computer_name == 'DESKTOP-F5LCT4Q':
    #freemocap_data_path = Path(r"C:\Users\aaron\Documents\HumonLab\Spring2022\ValidationStudy\FreeMocap_Data")
    #freemocap_data_path = Path(r'D:\freemocap2022\FreeMocap_Data')
    freemocap_data_path = Path(r'D:\ValidationStudy2022\FreeMocap_Data')
else:
    #freemocap_validation_data_path = Path(r"C:\Users\kiley\Documents\HumonLab\SampleFMC_Data\FreeMocap_Data-20220216T173514Z-001\FreeMocap_Data")
    freemocap_data_path = Path(r"C:\Users\Rontc\Documents\HumonLab\ValidationStudy")



#freemocap_sessionID = 'gopro_sesh_2022-05-24_16_02_53_JSM_T1_BOS' #name of the sessionID folder
#qualisys_sessionID = 'qualisys_sesh_2022-05-24_16_02_53_JSM_T1_BOS'

#freemocap_sessionID = 'gopro_sesh_2022-05-24_16_02_53_JSM_T1_WalkRun'
#qualisys_sessionID = 'qualisys_sesh_2022-05-24_16_02_53_JSM_T1_WalkRun'

freemocap_sessionID = 'sesh_2022-05-24_15_55_40_JSM_T1_BOS'
qualisys_sessionID =  'qualisys_sesh_2022-05-24_16_02_53_JSM_T1_BOS'

qualisys_data_array_name = 'qualisys_origin_aligned_skeleton_3D.npy'

mediapipe_data_array_name = 'mediapipe_origin_aligned_skeleton_3D.npy'
#mediapipe_data_array_name = 'mediaPipeSkel_3d_smoothed.npy'

num_pose_joints = 33

freemocap_session_path = freemocap_data_path / freemocap_sessionID
freemocap_data_array_path = freemocap_session_path/'DataArrays'

qualisys_session_path = freemocap_data_path/qualisys_sessionID
qualisys_data_array_path = qualisys_session_path/'DataArrays'

qualisys_data_path = qualisys_data_array_path/qualisys_data_array_name
mediapipe_data_path = freemocap_data_array_path/mediapipe_data_array_name

#qualysis_mat_file = sio.loadmat(qualisys_data_path)
qualisys_pose_data = np.load(qualisys_data_path)
#qualisys_num_frame_range = range(qualisys_pose_data.shape[0])
qualisys_num_frame_range = range(qualisys_pose_data.shape[0])

mediapipeSkel_fr_mar_dim = np.load(mediapipe_data_path)
mediapipe_pose_data = slice_mediapipe_data(mediapipeSkel_fr_mar_dim, num_pose_joints)

#mediapipe_pose_data = mediapipe_pose_data[0:10000,:,:]

mediapipe_num_frame_range = range(len(mediapipe_pose_data))

mediapipe_skeleton_path = freemocap_data_array_path/'origin_aligned_mediapipe_Skelcoordinates_frame_segment_joint_XYZ.pkl'
qualisys_skeleton_path = qualisys_data_array_path/'origin_aligned_qualisys_Skelcoordinates_frame_segment_joint_XYZ.pkl'

df = pd.DataFrame(list(zip(segments,joint_connections,segment_COM_lengths,segment_COM_percentages)),columns = ['Segment Name','Joint Connection','Segment COM Length','Segment COM Percentage'])
segment_conn_len_perc_dataframe = df.set_index('Segment Name')

if mediapipe_skeleton_path.is_file():
    with open(mediapipe_skeleton_path, 'rb') as f:
        mediapipe_skeleton_data = pickle.load(f)
    f.close()
else:
    mediapipe_skeleton_data = build_mediapipe_skeleton(mediapipe_pose_data,segment_conn_len_perc_dataframe, mediapipe_indices, mediapipe_num_frame_range)

if qualisys_skeleton_path.is_file():
    with open(qualisys_skeleton_path, 'rb') as f:
        qualisys_skeleton_data = pickle.load(f)
    f.close()
else:
    qualisys_skeleton_data = build_qualisys_skeleton(qualisys_pose_data,segment_conn_len_perc_dataframe, qualisys_indices, qualisys_num_frame_range)




#qualisys_frame = 56686
qualisys_frame = 4730-750
#mediapipe_frame = 9499
mediapipe_frame = 473

this_frame_qualisys_joint_data = qualisys_pose_data[qualisys_frame,:,:]
this_frame_qualisys_skeleton = qualisys_skeleton_data[qualisys_frame]



this_frame_mediapipe_joint_data = mediapipeSkel_fr_mar_dim[mediapipe_frame,:,:]
this_frame_mediapipe_skeleton = mediapipe_skeleton_data[mediapipe_frame]

mediapipe_joint_data = mediapipeSkel_fr_mar_dim

segments_for_reference = ['trunk',
'right_upper_arm',
'left_upper_arm',
'right_forearm',
'left_forearm',
'right_hand',
'left_hand',
'right_thigh',
'left_thigh',
'right_shin',
'left_shin',
'right_foot',
'left_foot']

#segments_for_reference = ['left_thigh', 'left_foot', 'right_foot']

mediapipe_reference_segment_list = [this_frame_mediapipe_skeleton[segment] for segment in segments_for_reference]
qualisys_reference_segment_list = [this_frame_qualisys_skeleton[segment] for segment in segments_for_reference]



segmentA = mediapipe_reference_segment_list[0]
segmentB = qualisys_reference_segment_list[0]

rotation_matrix = get_optimal_rotation_matrix(mediapipe_reference_segment_list, qualisys_reference_segment_list)


def plot_final_rotated_segments(ax,segmentA, segmentB, rotation_matrix):
    segmentA_rotated_by_guess = [rotation_matrix @ x for x in segmentA]

    segmentA_xvalues, segmentA_yvalues, segmentA_zvalues = get_segment_values(segmentA)
    segmentA_rotated_by_guess_xvalues, segmentA_rotated_by_guess_yvalues, segmentA_rotated_by_guess_zvalues = get_segment_values(segmentA_rotated_by_guess)
    segmentB_xvalues, segmentB_yvalues, segmentB_zvalues = get_segment_values(segmentB)

    ax.plot(segmentA_xvalues, segmentA_yvalues, segmentA_zvalues, 'b',label='original segmentA', alpha = .5)
    ax.plot(segmentB_xvalues, segmentB_yvalues, segmentB_zvalues, 'r',label='original segmentB', alpha = .5)
    ax.plot(segmentA_rotated_by_guess_xvalues, segmentA_rotated_by_guess_yvalues, segmentA_rotated_by_guess_zvalues, 'g-o',label='segmentA_rotated_by_guess', alpha = .5, linestyle = 'dashed')


def get_segment_values(segment):

    segment_x_values = [segment[0][0],segment[1][0]]
    segment_y_values = [segment[0][1],segment[1][1]]
    segment_z_values = [segment[0][2],segment[1][2]]

    return segment_x_values, segment_y_values, segment_z_values

# figure = plt.figure()
# ax1 = figure.add_subplot(111,projection = '3d')


# ax1.legend()
# plt.show()

# this_frame_rotated_mediapipe_joint_data = rotate_skeleton_frame(this_frame_mediapipe_joint_data, rotation_matrix)


f = 2



# uncomment for later
mediapipe_rotated_joints = np.zeros(mediapipe_joint_data.shape)

num_frames = mediapipe_joint_data.shape[0]

for frame in track(range(num_frames), description='rotating'):
    mediapipe_rotated_joints[frame,:,:] = rotate_skeleton_frame(mediapipe_joint_data[frame,:,:], rotation_matrix)

rotated_mediapipe_skeleton = build_mediapipe_skeleton(mediapipe_rotated_joints,segment_conn_len_perc_dataframe, mediapipe_indices, mediapipe_num_frame_range)

this_frame_rotated_mediapipe_joint_data = mediapipe_rotated_joints[mediapipe_frame]
this_frame_rotated_mediapipe_skeleton = rotated_mediapipe_skeleton[mediapipe_frame]

rotated_mediapipe_reference_segment_list = [this_frame_rotated_mediapipe_skeleton[segment] for segment in segments_for_reference]

translation_matrix = get_optimal_translation_matrix(rotated_mediapipe_reference_segment_list, qualisys_reference_segment_list)


mediapipe_translated = np.zeros(mediapipe_joint_data.shape)

for frame in track(range(num_frames), description = 'translating'):
    mediapipe_translated[frame,:,:] = translate_skeleton_frame(mediapipe_rotated_joints[frame,:,:], translation_matrix)

this_frame_mediapipe_translated = mediapipe_translated[mediapipe_frame]
#mediapipe_translated[:,:] = translate_skeleton_frame(this_frame_rotated_mediapipe_joint_data, translation_matrix)

save_name = 'mediapipe_skel_data_aligned_to_qualisys.npy'
save_path = freemocap_data_array_path/save_name


np.save(save_path, mediapipe_translated)


f = 2


def set_axes_ranges(plot_ax,skeleton_data, ax_range):

    mx = np.nanmean(skeleton_data[:,0])
    my = np.nanmean(skeleton_data[:,1])
    mz = np.nanmean(skeleton_data[:,2])

    plot_ax.set_xlim(mx-ax_range,mx+ax_range)
    plot_ax.set_ylim(my-ax_range,my+ax_range)
    plot_ax.set_zlim(mz-ax_range,mz+ax_range)        



figure = plt.figure()
ax1 = figure.add_subplot(141,projection = '3d')
ax2 = figure.add_subplot(142,projection = '3d')
ax3 = figure.add_subplot(143,projection = '3d')
ax4 = figure.add_subplot(144,projection = '3d')




ax1.scatter(this_frame_qualisys_joint_data[:,0],this_frame_qualisys_joint_data[:,1],this_frame_qualisys_joint_data[:,2],c = 'r',marker = 'o')
ax1.scatter(this_frame_mediapipe_joint_data[:,0],this_frame_mediapipe_joint_data[:,1],this_frame_mediapipe_joint_data[:,2],c = 'b',marker = 'o')

ax2.scatter(this_frame_mediapipe_joint_data[:,0],this_frame_mediapipe_joint_data[:,1],this_frame_mediapipe_joint_data[:,2],c = 'b',marker = 'o')
ax2.scatter(this_frame_rotated_mediapipe_joint_data[:,0],this_frame_rotated_mediapipe_joint_data[:,1],this_frame_rotated_mediapipe_joint_data[:,2],c = 'b',marker = 'o')

ax3.scatter(this_frame_qualisys_joint_data[:,0],this_frame_qualisys_joint_data[:,1],this_frame_qualisys_joint_data[:,2],c = 'r',marker = 'o')
ax3.scatter(this_frame_rotated_mediapipe_joint_data[:,0],this_frame_rotated_mediapipe_joint_data[:,1],this_frame_rotated_mediapipe_joint_data[:,2],c = 'b',marker = 'o')

ax4.scatter(this_frame_qualisys_joint_data[:,0],this_frame_qualisys_joint_data[:,1],this_frame_qualisys_joint_data[:,2],c = 'r',marker = 'o')
ax4.scatter(this_frame_mediapipe_translated[:,0], this_frame_mediapipe_translated[:,1],this_frame_mediapipe_translated[:,2],c = 'b',marker = 'o')

for count,segment in enumerate(segments_for_reference):
    plot_final_rotated_segments(ax3,mediapipe_reference_segment_list[count], qualisys_reference_segment_list[count], rotation_matrix)


set_axes_ranges(ax1,this_frame_qualisys_joint_data,1000)
set_axes_ranges(ax2,this_frame_qualisys_joint_data,1000)
set_axes_ranges(ax3,this_frame_qualisys_joint_data,1000)
set_axes_ranges(ax4,this_frame_qualisys_joint_data,1000)

plt.show()

f=2