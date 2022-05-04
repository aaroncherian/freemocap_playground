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

import math
import scipy.spatial.transform as st

this_computer_name = socket.gethostname()
print(this_computer_name)


system_to_plot = 'mediapipe'

if this_computer_name == 'DESKTOP-V3D343U':
    freemocap_validation_data_path = Path(r"I:\My Drive\HuMoN_Research_Lab\FreeMoCap_Stuff\FreeMoCap_Balance_Validation\data")
elif this_computer_name == 'DESKTOP-F5LCT4Q':
    freemocap_validation_data_path = Path(r"C:\Users\aaron\Documents\HumonLab\Spring2022\ValidationStudy\FreeMocap_Data")
    #freemocap_validation_data_path = Path(r'D:\freemocap2022\FreeMocap_Data')
else:
    #freemocap_validation_data_path = Path(r"C:\Users\kiley\Documents\HumonLab\SampleFMC_Data\FreeMocap_Data-20220216T173514Z-001\FreeMocap_Data")
    freemocap_validation_data_path = Path(r"C:\Users\Rontc\Documents\HumonLab\ValidationStudy")

debug = True

sessionID = 'session_SER_1_20_22' #name of the sessionID folder
this_freemocap_session_path = freemocap_validation_data_path / sessionID
this_freemocap_data_path = this_freemocap_session_path/'DataArrays'

if system_to_plot == 'mediapipe':
    mediapipe_data_path = this_freemocap_data_path/'mediaPipeSkel_3d_smoothed.npy'

elif system_to_plot == 'openpose':
    mediapipe_data_path = this_freemocap_data_path/'openPoseSkel_3d_smoothed.npy'

mediapipe_data = np.load(mediapipe_data_path)

frame = 3000


this_mediapipe_test = mediapipe_data[frame,:,:]


this_frame_skel_x = this_mediapipe_test[:,0]
this_frame_skel_y = this_mediapipe_test[:,1]
this_frame_skel_z = this_mediapipe_test[:,2]


def create_vector(point1,point2):
    vector = point2 - point1
    return vector

def create_normal_vector(vector1,vector2):
    normal_vector = np.cross(vector1,vector2)
    return normal_vector

def create_unit_vector(vector):
    unit_vector = vector/np.linalg.norm(vector)
    return unit_vector


origin = np.array([0, 0, 0])
x_axis = np.array([1, 0, 0])
y_axis = np.array([0, 1, 0])

x_vector = create_vector(origin,x_axis)
y_vector = create_vector(origin,y_axis)

origin_normal = create_normal_vector(x_vector,y_vector)
origin_normal_unit = create_unit_vector(origin_normal)


# #for openpose
if system_to_plot == 'openpose':
    right_heel = 24
    right_foot_origin = this_mediapipe_test[24,:]
    left_foot_origin = this_mediapipe_test[21,:]
    right_foot_toe = np.mean([this_mediapipe_test[22,:],this_mediapipe_test[23,:]], axis = 0)
    right_foot_vector = create_vector(this_mediapipe_test[24,:],right_foot_toe)
    left_foot_vector = create_vector(this_mediapipe_test[21,:],this_mediapipe_test[19,:])
    heel_vector = create_vector(this_mediapipe_test[24,:],this_mediapipe_test[21,:])
    


#for mediapipe
if system_to_plot == 'mediapipe':
    right_heel = 30
    right_foot_origin = this_mediapipe_test[30,:]
    left_foot_origin = this_mediapipe_test[29,:]
    right_foot_vector = create_vector(this_mediapipe_test[30,:],this_mediapipe_test[32,:])
    left_foot_vector = create_vector(this_mediapipe_test[29,:],this_mediapipe_test[31,:])
    heel_vector = create_vector(this_mediapipe_test[30,:],this_mediapipe_test[29,:])

  




foot_normal =  create_normal_vector(right_foot_vector,heel_vector)

foot_normal_unit = create_unit_vector(foot_normal)

right_foot_unit_vector = create_unit_vector(right_foot_vector)

heel_unit_vector = create_unit_vector(heel_vector)

XXV,YXV,ZXV = zip(x_vector*1000)
XYV,YYV,ZYV = zip(y_vector*1000)

X0,Y0,Z0 = zip(origin)
U0,V0,W0 = zip(origin_normal_unit*1000)


UY,VY,WY = zip(y_vector*1000)

X, Y, Z = zip(right_foot_origin) 
U,V,W = zip(right_foot_vector)

X1, Y1, Z1 = zip(left_foot_origin)
U1,V1,W1 = zip(left_foot_vector)

U2, V2, W2 = zip(heel_vector)

U3,V3,W3 = zip(foot_normal_unit*1000)


#R = st.Rotation.align_vectors(foot_normal_unit,origin_normal_unit)

f = 2

def calculate_skewed_symmetric_cross_product(cross_product_vector):
    skew_symmetric_cross_product = np.array([[0, -cross_product_vector[2], cross_product_vector[1]],
                                             [cross_product_vector[2], 0, -cross_product_vector[0]],
                                             [-cross_product_vector[1], cross_product_vector[0], 0]])
    return skew_symmetric_cross_product


def calculate_rotation_matrix(vector1,vector2):

    identity_matrix = np.identity(3)
    
    vector_cross_product = np.cross(vector1,vector2)

    vector_dot_product = np.dot(vector1,vector2)

    skew_symmetric_cross_product = calculate_skewed_symmetric_cross_product(vector_cross_product)

    rotation_matrix  = identity_matrix + skew_symmetric_cross_product + (np.dot(skew_symmetric_cross_product,skew_symmetric_cross_product))*(1 - vector_dot_product)/(np.linalg.norm(vector_cross_product)**2)

    return rotation_matrix

rotation_matrix = calculate_rotation_matrix(foot_normal_unit,origin_normal_unit)


unit_heel = create_unit_vector(right_foot_origin)

rotated_right_heel = np.dot(rotation_matrix,foot_normal_unit)


A,B,C = zip(rotated_right_heel*1500)

def rotate_point(point,rotation_matrix):

    point_unit_vector = create_unit_vector(point)

    rotated_point = np.dot(rotation_matrix,point)

    return rotated_point

def rotate_skeleton(skeleton_data, rotation_matrix):

    rotated_skeleton = np.zeros((skeleton_data.shape))

    for i in range(skeleton_data.shape[0]):

        rotated_skeleton[i,:] = rotate_point(skeleton_data[i,:],rotation_matrix)

    return rotated_skeleton




mediapipe_rotated_skeleton = rotate_skeleton(this_mediapipe_test,rotation_matrix)

rotated_right_heel = mediapipe_rotated_skeleton[right_heel,:]

translation_distance = rotated_right_heel - [0,0,0]


def translate_skeleton(rotated_skeleton_data, translation_distance):

    translated_skeleton = rotated_skeleton_data - translation_distance

    return translated_skeleton

mediapipe_translated_skeleton = translate_skeleton(mediapipe_rotated_skeleton,translation_distance)


if system_to_plot == 'openpose':
    hip_center = mediapipe_translated_skeleton[8,:]
    shoulder_center = (mediapipe_translated_skeleton[2,:] + mediapipe_translated_skeleton[5,:])/2

elif system_to_plot == 'mediapipe':
    hip_center = (mediapipe_translated_skeleton[11,:] + mediapipe_translated_skeleton[12,:])/2
    shoulder_center = (mediapipe_translated_skeleton[23,:] + mediapipe_translated_skeleton[24,:])/2



translated_right_foot_vector = create_vector(mediapipe_translated_skeleton[30,:],mediapipe_translated_skeleton[32,:])

translated_right_foot_unit_vector = create_unit_vector(translated_right_foot_vector)

rotation_matrix_to_align_with_positive_y = calculate_rotation_matrix(translated_right_foot_unit_vector,y_vector)

origin_aligned_skeleton = rotate_skeleton(mediapipe_translated_skeleton,rotation_matrix_to_align_with_positive_y)

if debug:
    figure = plt.figure()
    ax = figure.add_subplot( projection = '3d')


    # ax_range = 800
    # mx = np.nanmean(mediapipe_translated_skeleton[0,:])
    # my = np.nanmean(mediapipe_translated_skeleton[1,:])
    # mz = np.nanmean(mediapipe_translated_skeleton[2,:])

    # mx = np.nanmean(mediapipe_data[0,:])
    # my = np.nanmean(mediapipe_data[1,:])
    # mz = np.nanmean(mediapipe_data[2,:])

    ax_range = 800
    mx = np.nanmean(origin_aligned_skeleton[0,:])
    my = np.nanmean(origin_aligned_skeleton[1,:])
    mz = np.nanmean(origin_aligned_skeleton[2,:])


  
    #ax.set_box_aspect([1,1,1])
    ax.set_xlim([mx-ax_range, mx+ax_range]) #maybe set ax limits before the function? if we're using cla() they probably don't need to be redefined every time 
    ax.set_ylim([my-ax_range, my+ax_range])
    ax.set_zlim([mz-ax_range, mz+ax_range])
    ax.scatter(origin_aligned_skeleton[:,0], origin_aligned_skeleton[:,1], origin_aligned_skeleton[:,2], c='orange', marker='.')
    # xx, yy = np.meshgrid(range(1000), range(1000))
    # zz = xx*0

    
    # ax.plot_surface(xx, yy, zz)

    # ax.scatter(this_frame_skel_x, this_frame_skel_y, this_frame_skel_z, c='r', marker='.')
    # #ax.scatter(rotated_right_heel[0], rotated_right_heel[1], rotated_right_heel[2], c='b', marker='.')

    # ax.scatter(mediapipe_rotated_skeleton[:,0], mediapipe_rotated_skeleton[:,1], mediapipe_rotated_skeleton[:,2], c='g', marker='.')
    ax.scatter(mediapipe_translated_skeleton[:,0], mediapipe_translated_skeleton[:,1], mediapipe_translated_skeleton[:,2], c='b', marker='.')
    # ax.scatter(origin_aligned_skeleton[:,0], mediapipe_translated_skeleton[:,1], mediapipe_translated_skeleton[:,2], c='orange', marker='.')

    # ax.plot([hip_center[0], shoulder_center[0]], [hip_center[1], shoulder_center[1]], [hip_center[2], shoulder_center[2]], c='k')
    # # ax.scatter(this_mediapipe_test[19,0], this_mediapipe_test[19,1],this_mediapipe_test[19,2], c='b', marker='o')
    # # ax.scatter(this_mediapipe_test[31,0], this_mediapipe_test[31,1],this_mediapipe_test[31,2], c='b', marker='.')

    # #ax.plot([0,this_frame_skel_x[32]], [0,this_frame_skel_y[32]], [0,this_frame_skel_z[32]], c='b')

    # ax.quiver(X,Y,Z,U,V,W,arrow_length_ratio=0.1)

    # ax.quiver(X1,Y1,Z1,U1,V1,W1,arrow_length_ratio=0.1)
    # ax.quiver(X,Y,Z,U2,V2,W2,arrow_length_ratio=0.1)

    # ax.quiver(X,Y,Z,U3,V3,W3,arrow_length_ratio=0.1, color = 'g')
    # ax.quiver(X0,Y0,Z0,U0,V0,W0,arrow_length_ratio=0.1, color = 'r')

    # ax.quiver(X0,Y0,Z0,A,B,C,arrow_length_ratio=0.1, alpha = .5, color = 'g')

    # #ax.quiver(X0,Y0,Z0,XXV,XYV,ZXV,arrow_length_ratio=0.1)
    # #ax.quiver(X0,Y0,Z0,YXV,YYV,ZYV,arrow_length_ratio=0.1)

    plt.show()


mediapipe_translated_skeleton[30,2]
mediapipe_translated_skeleton[31,2]


mediapipe_translated_skeleton[29,2]
mediapipe_translated_skeleton[31,2]


rotation_matrix_to_align_with_positive_y = calculate_rotation_matrix(right_foot_unit_vector,y_vector)

def calculate_distance(point1, point2):

    distance = np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2 + (point1[2]-point2[2])**2)

    return distance

for number in range(0,32):

    point1_og = this_mediapipe_test[0,:]
    point2_og = this_mediapipe_test[number,:]

    point1_translate = mediapipe_translated_skeleton[0,:]
    point2_translate = mediapipe_translated_skeleton[number,:]

    og_rotated_distance = calculate_distance(point1_og, point2_og)
    translate_rotated_distance = calculate_distance(point1_translate, point2_translate)

    if not np.round(og_rotated_distance,5) == np.round(translate_rotated_distance,5):
        print('number not equal:{}, og_distance:{}, rotated_distance: {} ', number, og_rotated_distance, translate_rotated_distance)
        f=2

    f = 2




f = 2

