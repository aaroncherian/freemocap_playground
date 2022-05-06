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
skeleton_to_plot = 'mediapipe' #for a future situation where we want to rotate openpose/dlc skeletons 
rotation_base_frame = 3000
debug = True


this_freemocap_session_path = freemocap_validation_data_path / sessionID
this_freemocap_data_path = this_freemocap_session_path/'DataArrays'
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


## functions used in getting a matrix
def create_vector(point1,point2): 
    """Put two points in, make a vector"""
    vector = point2 - point1
    return vector

def create_normal_vector(vector1,vector2): 
    """Put two vectors in, make a normal vector"""
    normal_vector = np.cross(vector1,vector2)
    return normal_vector

def create_unit_vector(vector): 
    """Take in a vector, make it a unit vector"""
    unit_vector = vector/np.linalg.norm(vector)
    return unit_vector


def calculate_skewed_symmetric_cross_product(cross_product_vector):
    skew_symmetric_cross_product = np.array([[0, -cross_product_vector[2], cross_product_vector[1]],
                                             [cross_product_vector[2], 0, -cross_product_vector[0]],
                                             [-cross_product_vector[1], cross_product_vector[0], 0]])
    return skew_symmetric_cross_product


def calculate_rotation_matrix(vector1,vector2):
    """Put in two vectors to calculate the rotation matrix between those two vectors"""
    #based on the code found here: https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d"""
    
    identity_matrix = np.identity(3)
    vector_cross_product = np.cross(vector1,vector2)
    vector_dot_product = np.dot(vector1,vector2)
    skew_symmetric_cross_product = calculate_skewed_symmetric_cross_product(vector_cross_product)
    rotation_matrix  = identity_matrix + skew_symmetric_cross_product + (np.dot(skew_symmetric_cross_product,skew_symmetric_cross_product))*(1 - vector_dot_product)/(np.linalg.norm(vector_cross_product)**2)

    return rotation_matrix

def rotate_point(point,rotation_matrix):
    rotated_point = np.dot(rotation_matrix,point)
    return rotated_point

def rotate_skeleton_frame(this_frame_skeleton_data, rotation_matrix):
    """Take in a frame of skeleton data, and apply the rotation matrix to each point in the skeleton"""

    this_frame_rotated_skeleton = np.zeros(this_frame_skeleton_data.shape)  #initialize the array to hold the rotated skeleton data for this frame
    num_tracked_points = this_frame_skeleton_data.shape[0]

    for i in range(num_tracked_points):
        this_frame_rotated_skeleton[i,:] = rotate_point(this_frame_skeleton_data[i,:],rotation_matrix)

    return this_frame_rotated_skeleton



def calculate_translation_distance(skeleton_right_heel):
    """Take in the right heel point, and calculate the distance between the right heel and the origin"""

    translation_distance = skeleton_right_heel - [0,0,0]
    return translation_distance 


def translate_skeleton_frame(rotated_skeleton_data_frame, translation_distance):
    """Take in a frame of rotated skeleton data, and apply the translation distance to each point in the skeleton"""

    translated_skeleton_frame = rotated_skeleton_data_frame - translation_distance
    return translated_skeleton_frame

origin = np.array([0, 0, 0])
x_axis = np.array([1, 0, 0])
y_axis = np.array([0, 1, 0])

x_vector = create_vector(origin,x_axis)
y_vector = create_vector(origin,y_axis)

origin_normal = create_normal_vector(x_vector,y_vector) #create a normal vector to the origin (basically the z axis)
origin_normal_unit_vector = create_unit_vector(origin_normal) #note - this is kinda unncessary because the origin normal unit vector == original normal vector 


rotation_frame_skeleton_data = skeleton_data[rotation_base_frame,:,:] #get the skeleton data for the frame we want to calculate the rotation matrix from

right_foot_origin = rotation_frame_skeleton_data[right_heel_index,:]
left_foot_origin = rotation_frame_skeleton_data[left_heel_index,:]

#create vectors for the right foot and left foot, and between the two heels 
right_foot_vector = create_vector(rotation_frame_skeleton_data[right_heel_index,:],rotation_frame_skeleton_data[right_toe_index,:])
left_foot_vector = create_vector(rotation_frame_skeleton_data[left_heel_index,:],rotation_frame_skeleton_data[left_toe_index,:])
heel_vector = create_vector(rotation_frame_skeleton_data[right_heel_index,:],rotation_frame_skeleton_data[left_heel_index,:])

#create a normal unit vector from the right foot and heel vector
foot_normal =  create_normal_vector(right_foot_vector,heel_vector)
foot_normal_unit_vector = create_unit_vector(foot_normal)

#right_foot_unit_vector = create_unit_vector(right_foot_vector)
#heel_unit_vector = create_unit_vector(heel_vector)

#calculate the rotation matrix between the origin normal and the foot normal
rotation_matrix = calculate_rotation_matrix(foot_normal_unit_vector,origin_normal_unit_vector)




rotated_skeleton_data = np.zeros(skeleton_data.shape) #create an array to hold the rotated skeleton data
num_frames = skeleton_data.shape[0]

for frame in track(range(num_frames)): #rotate the skeleton on each frame 
    rotated_skeleton_data[frame,:,:] = rotate_skeleton_frame(skeleton_data[frame,:,:],rotation_matrix)

#get the translation distance between the right heel and the origin on the frame we used to build the rotation matrix
translation_distance = calculate_translation_distance(rotated_skeleton_data[rotation_base_frame,right_heel_index,:])

translated_and_rotated_skeleton_data = np.zeros(rotated_skeleton_data.shape)


for frame in track(range(num_frames)):
   translated_and_rotated_skeleton_data[frame,:,:] = translate_skeleton_frame(rotated_skeleton_data[frame,:,:],translation_distance) #translate the skeleton data for each frame  


#to get the final alignment (to face the person forward in +y), get the new heel unit vector 
translated_and_rotated_heel_vector = create_vector(translated_and_rotated_skeleton_data[rotation_base_frame,left_heel_index,:],translated_and_rotated_skeleton_data[rotation_base_frame,right_heel_index,:])
translated_and_rotated_heel_unit_vector = create_unit_vector(translated_and_rotated_heel_vector)

#find the rotation matrix between the new heel unit vector and the x-axis
rotation_matrix_to_align_skeleton_with_positive_y = calculate_rotation_matrix(translated_and_rotated_heel_unit_vector,x_vector)

origin_aligned_skeleton_data = np.zeros(skeleton_data.shape)

#rotate the skeleton in each frame with the new rotation matrix 
for frame in track(range(num_frames)):
   origin_aligned_skeleton_data[frame,:,:] = rotate_skeleton_frame(translated_and_rotated_skeleton_data[frame,:,:],rotation_matrix_to_align_skeleton_with_positive_y)


if debug:

    #debug plot shows the X,Y,Z vectors pointing out from the origin - the right heel should be at 0,0,0. 
    #the z-vector should align with the normal vector between the right foot vector and heel vector
    #the x-vector should align with the vector between the left heel and right heel 
    
    figure = plt.figure()
    ax = figure.add_subplot( projection = '3d')

    this_frame_skeleton_data = origin_aligned_skeleton_data[rotation_base_frame,:,:]

    ax_range = 800
    mx = np.nanmean(this_frame_skeleton_data[:,0])
    my = np.nanmean(this_frame_skeleton_data[:,1])
    mz = np.nanmean(this_frame_skeleton_data[:,2])

    ax.set_xlim([mx-ax_range, mx+ax_range]) #maybe set ax limits before the function? if we're using cla() they probably don't need to be redefined every time 
    ax.set_ylim([my-ax_range, my+ax_range])
    ax.set_zlim([mz-ax_range, mz+ax_range])


    ax.scatter(this_frame_skeleton_data[:,0],this_frame_skeleton_data[:,1],this_frame_skeleton_data[:,2],c='r')


    
    this_frame_right_foot_vector = create_vector(this_frame_skeleton_data[right_heel_index,:] ,this_frame_skeleton_data[right_toe_index,:])
    this_frame_heel_vector = create_vector(this_frame_skeleton_data[right_heel_index,:] ,this_frame_skeleton_data[left_heel_index,:]) #this is the heel vector for aligning the right heel with the origin
    this_frame_x_alignment_heel_vector = create_vector(this_frame_skeleton_data[left_heel_index,:] ,this_frame_skeleton_data[right_heel_index,:]) #this is the heel vector for aligning the skeleton to face positive y
#NOTE - maybe use the left heel as the origin for the rotation as well 

    this_frame_heel_unit_vector = create_unit_vector(this_frame_heel_vector)
    this_frame_heel_alignment_unit_vector = create_unit_vector(this_frame_x_alignment_heel_vector)

    this_frame_foot_normal_vector = create_normal_vector(this_frame_right_foot_vector,this_frame_heel_vector)
    this_frame_foot_normal_unit_vector = create_unit_vector(this_frame_foot_normal_vector)

    #W,U,V = zip(this_frame_foot_unit_vector)

    Zvector_X,Zvector_Y,Zvector_Z = zip(origin_normal_unit_vector*800)
    Xvector_X,Xvector_Y,Xvector_Z = zip(x_vector*800)
    Yvector_X,Yvector_Y,Yvector_Z = zip(y_vector*800)

    Origin_X,Origin_Y,Origin_Z = zip(origin)

    Rightheel_X, Rightheel_Y, Rightheel_Z = zip(this_frame_skeleton_data[right_heel_index,:]) 
    Footnormal_X,Footnormal_Y,Footnormal_Z = zip(this_frame_foot_normal_unit_vector*500)
    Heel_X, Heel_Y, Heel_Z = zip(this_frame_heel_unit_vector*500)
    Heel_alignment_X, Heel_alignment_Y, Heel_alignment_Z = zip(this_frame_heel_alignment_unit_vector*500)

    ax.quiver(Origin_X,Origin_Y,Origin_Z,Zvector_X,Zvector_Y,Zvector_Z,arrow_length_ratio=0.1,color='b', label = 'Z-axis')
    ax.quiver(Origin_X,Origin_Y,Origin_Z,Xvector_X,Xvector_Y,Xvector_Z,arrow_length_ratio=0.1,color='cyan', label = 'X-axis')
    ax.quiver(Origin_X,Origin_Y,Origin_Z,Yvector_X,Yvector_Y,Yvector_Z,arrow_length_ratio=0.1,color='purple', label = 'Y-axis')

    ax.quiver(Rightheel_X,Rightheel_Y,Rightheel_Z,Footnormal_X,Footnormal_Y,Footnormal_Z,arrow_length_ratio=0.1,color='r')
    ax.quiver(Rightheel_X,Rightheel_Y,Rightheel_Z,Heel_alignment_X,Heel_alignment_Y,Heel_alignment_Z,arrow_length_ratio=0.1,color='g') 

    ax.legend()

    plt.show()

#save the aligned skeleton data to a new file
np.save(save_file,origin_aligned_skeleton_data)
f = 2