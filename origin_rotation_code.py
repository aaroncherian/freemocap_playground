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
skeleton_to_plot = 'mediapipe'
rotation_base_frame = 3000

save_file = this_freemocap_data_path/'{}_origin_aligned_skeleton_3D.npy'.format(skeleton_to_plot)


if skeleton_to_plot == 'mediapipe':
    skeleton_data_path = this_freemocap_data_path/'mediaPipeSkel_3d_smoothed.npy'
    right_heel_index = 30
    right_toe_index = 32
    left_heel_index = 29
    left_toe_index = 31

elif skeleton_to_plot == 'openpose':
    skeleton_data_path = this_freemocap_data_path/'openPoseSkel_3d_smoothed.npy'

skeleton_data = np.load(skeleton_data_path)





##get rotation matrix 
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


rotation_frame_skeleton_data = skeleton_data[rotation_base_frame,:,:]
right_foot_origin = rotation_frame_skeleton_data[right_heel_index,:]
left_foot_origin = rotation_frame_skeleton_data[left_heel_index,:]
right_foot_vector = create_vector(rotation_frame_skeleton_data[right_heel_index,:],rotation_frame_skeleton_data[right_toe_index,:])
left_foot_vector = create_vector(rotation_frame_skeleton_data[left_heel_index,:],rotation_frame_skeleton_data[left_toe_index,:])
heel_vector = create_vector(rotation_frame_skeleton_data[right_heel_index,:],rotation_frame_skeleton_data[left_heel_index,:])


foot_normal =  create_normal_vector(right_foot_vector,heel_vector)
foot_normal_unit = create_unit_vector(foot_normal)
right_foot_unit_vector = create_unit_vector(right_foot_vector)
heel_unit_vector = create_unit_vector(heel_vector)


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


def rotate_point(point,rotation_matrix):
    rotated_point = np.dot(rotation_matrix,point)
    return rotated_point

def rotate_skeleton_frame(this_frame_skeleton_data, rotation_matrix):

    this_frame_rotated_skeleton = np.zeros(this_frame_skeleton_data.shape)

    num_tracked_points = this_frame_skeleton_data.shape[0]

    for i in range(num_tracked_points):
        this_frame_rotated_skeleton[i,:] = rotate_point(this_frame_skeleton_data[i,:],rotation_matrix)
    return this_frame_rotated_skeleton

rotated_skeleton_data = np.zeros(skeleton_data.shape)
num_frames = skeleton_data.shape[0]

for frame in track(range(num_frames)):
    rotated_skeleton_data[frame,:,:] = rotate_skeleton_frame(skeleton_data[frame,:,:],rotation_matrix)


def calculate_translation_distance(skeleton_right_heel):
    translation_distance = skeleton_right_heel - [0,0,0]
    return translation_distance 

translation_distance = calculate_translation_distance(rotated_skeleton_data[rotation_base_frame,right_heel_index,:])


def translate_skeleton_frame(rotated_skeleton_data_frame, translation_distance):

    translated_skeleton_frame = rotated_skeleton_data_frame - translation_distance

    return translated_skeleton_frame

translated_and_rotated_skeleton_data = np.zeros(rotated_skeleton_data.shape)


for frame in track(range(num_frames)):
   translated_and_rotated_skeleton_data[frame,:,:] = translate_skeleton_frame(rotated_skeleton_data[frame,:,:],translation_distance)



translated_and_rotated_right_foot_vector = create_vector(translated_and_rotated_skeleton_data[rotation_base_frame,left_heel_index,:],translated_and_rotated_skeleton_data[rotation_base_frame,right_heel_index,:])
translated_and_rotated_right_foot_unit_vector = create_unit_vector(translated_and_rotated_right_foot_vector)

rotation_matrix_to_align_skeleton_with_positive_y = calculate_rotation_matrix(translated_and_rotated_right_foot_unit_vector,x_vector)

origin_aligned_skeleton_data = np.zeros(skeleton_data.shape)

for frame in track(range(num_frames)):
   origin_aligned_skeleton_data[frame,:,:] = rotate_skeleton_frame(translated_and_rotated_skeleton_data[frame,:,:],rotation_matrix_to_align_skeleton_with_positive_y)

np.save(save_file,origin_aligned_skeleton_data)
f = 2