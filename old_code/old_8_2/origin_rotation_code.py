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
    #freemocap_validation_data_path = Path(r"C:\Users\aaron\Documents\HumonLab\Spring2022\ValidationStudy\FreeMocap_Data")
    #freemocap_validation_data_path = Path(r'D:\freemocap2022\FreeMocap_Data')
    freemocap_validation_data_path = Path(r'D:\ValidationStudy2022\FreeMocap_Data')
else:
    #freemocap_validation_data_path = Path(r"C:\Users\kiley\Documents\HumonLab\SampleFMC_Data\FreeMocap_Data-20220216T173514Z-001\FreeMocap_Data")
    freemocap_validation_data_path = Path(r"C:\Users\Rontc\Documents\HumonLab\ValidationStudy")

    
#sessionID = 'sesh_2022-05-03_13_43_00_JSM_treadmill_day2_t0' #name of the sessionID folder
sessionID = 'sesh_2022-05-24_15_55_40_JSM_T1_BOS'

skeleton_to_plot = 'mediapipe' #for a future situation where we want to rotate openpose/dlc skeletons 
base_frame = 468
debug = True


this_freemocap_session_path = freemocap_validation_data_path / sessionID
this_freemocap_data_path = this_freemocap_session_path/'DataArrays'
save_file = this_freemocap_data_path/'{}_origin_aligned_skeleton_3D.npy'.format(skeleton_to_plot)


if skeleton_to_plot == 'mediapipe':
    #skeleton_data_path = this_freemocap_data_path/'mediapipe_origin_corrected_and_rotated.npy'
    skeleton_data_path = this_freemocap_data_path/'mediaPipeSkel_3d_smoothed.npy'
    right_heel_index = 30
    right_toe_index = 32
    left_heel_index = 29
    left_toe_index = 31

    num_pose_joints = 33
    

elif skeleton_to_plot == 'openpose':
    skeleton_data_path = this_freemocap_data_path/'openPoseSkel_3d_smoothed.npy'

# primary_foot_indices = [left_heel_index,left_toe_index]
# secondary_foot_index = [right_heel_index]

primary_foot_indices = [left_heel_index,left_toe_index]
secondary_foot_index = [right_heel_index]

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

def rotate_skeleton_frame(this_frame_aligned_skeleton_data, rotation_matrix):
    """Take in a frame of skeleton data, and apply the rotation matrix to each point in the skeleton"""

    this_frame_rotated_skeleton = np.zeros(this_frame_aligned_skeleton_data.shape)  #initialize the array to hold the rotated skeleton data for this frame
    num_tracked_points = this_frame_aligned_skeleton_data.shape[0]

    for i in range(num_tracked_points):
        this_frame_rotated_skeleton[i,:] = rotate_point(this_frame_aligned_skeleton_data[i,:],rotation_matrix)

    return this_frame_rotated_skeleton



def calculate_translation_distance(skeleton_right_heel):
    """Take in the right heel point, and calculate the distance between the right heel and the origin"""

    translation_distance = skeleton_right_heel - [0,0,0]
    return translation_distance 


def translate_skeleton_frame(rotated_skeleton_data_frame, translation_distance):
    """Take in a frame of rotated skeleton data, and apply the translation distance to each point in the skeleton"""

    translated_skeleton_frame = rotated_skeleton_data_frame - translation_distance
    return translated_skeleton_frame


def calculate_normal_vector_to_foot(heel_one_index, toe_one_index, heel_two_index, skeleton_data):
    foot_one_vector = create_vector(skeleton_data[heel_one_index,:],skeleton_data[toe_one_index,:])
    heel_vector = create_vector(skeleton_data[heel_one_index,:],skeleton_data[heel_two_index,:])

    foot_normal_vector =  create_normal_vector(heel_vector,foot_one_vector)

    return foot_normal_vector, foot_one_vector, heel_vector

origin = np.array([0, 0, 0])
x_axis = np.array([1, 0, 0])
y_axis = np.array([0, 1, 0])
z_axis = np.array([0, 0, 1])

x_vector = create_vector(origin,x_axis)
y_vector = create_vector(origin,y_axis)
z_vector = create_vector(origin,z_axis)

#origin_normal = create_normal_vector(x_vector,y_vector) #create a normal vector to the origin (basically the z axis)
origin_normal_unit_vector = z_vector  #note - this is kinda unncessary because the origin normal unit vector == original normal vector 


num_frames = skeleton_data.shape[0]

base_frame_skeleton_data = skeleton_data[base_frame,:,:] #get the skeleton data for the frame we want to calculate the rotation matrix from

base_frame_normal_vector_to_foot,base_frame_left_foot_vector,base_frame_heel_vector = calculate_normal_vector_to_foot(left_heel_index,left_toe_index,right_heel_index,base_frame_skeleton_data)

#get the translation distance between the right heel and the origin on the frame we used to build the rotation matrix
translation_distance = calculate_translation_distance(base_frame_skeleton_data[primary_foot_indices[0],:])
translated_skeleton_data = np.zeros(skeleton_data.shape)


for frame in track(range(num_frames)):
   translated_skeleton_data[frame,:,:] = translate_skeleton_frame(skeleton_data[frame,:,:],translation_distance) #translate the skeleton data for each frame  


translated_base_frame_skeleton_data = translated_skeleton_data[base_frame,:,:] 
#translated_normal_vector_to_left_foot, translated_left_foot_vector, translated_heel_vector = calculate_normal_vector_to_foot(left_heel_index,left_toe_index,right_heel_index,translated_base_frame_skeleton_data)
translated_normal_vector_to_left_foot, translated_left_foot_vector, translated_heel_vector = calculate_normal_vector_to_foot(primary_foot_indices[0],primary_foot_indices[1], secondary_foot_index[0],translated_base_frame_skeleton_data)

unit_translated_normal_vector_to_left_foot = create_unit_vector(translated_normal_vector_to_left_foot)

#calculate the rotation matrix between the origin normal and the foot normal
rotation_matrix = calculate_rotation_matrix(unit_translated_normal_vector_to_left_foot,origin_normal_unit_vector)

translated_and_rotated_skeleton_data = np.zeros(skeleton_data.shape) #create an array to hold the rotated skeleton data
num_frames = translated_skeleton_data.shape[0]



for frame in track(range(num_frames)): #rotate the skeleton on each frame 
    translated_and_rotated_skeleton_data[frame,:,:] = rotate_skeleton_frame(translated_skeleton_data[frame,:,:],rotation_matrix)

translated_and_rotated_base_frame_skeleton_data = translated_and_rotated_skeleton_data[base_frame,:,:]

#translated_and_rotated_normal_vector_to_left_foot, translated_and_rotated_left_foot_vector, translated_and_rotated_heel_vector = calculate_normal_vector_to_foot(left_heel_index,left_toe_index,right_heel_index,translated_and_rotated_base_frame_skeleton_data)
translated_and_rotated_normal_vector_to_left_foot, translated_and_rotated_left_foot_vector, translated_and_rotated_heel_vector = calculate_normal_vector_to_foot(primary_foot_indices[0],primary_foot_indices[1], secondary_foot_index[0],translated_and_rotated_base_frame_skeleton_data)
unit_translated_and_rotated_normal_vector_to_left_foot = create_unit_vector(translated_and_rotated_normal_vector_to_left_foot)
unit_translated_and_rotated_heel_vector = create_unit_vector(translated_and_rotated_heel_vector)

#rotation_matrix_to_align_skeleton_with_positive_y = calculate_rotation_matrix(translated_and_rotated_heel_unit_vector,x_vector*-1)

translated_and_rotated_heel_vector = create_vector(translated_and_rotated_skeleton_data[base_frame,right_heel_index,:],translated_and_rotated_skeleton_data[base_frame,left_heel_index,:])
unit_translated_and_rotated_heel_vector = create_unit_vector(translated_and_rotated_heel_vector)


rotation_matrix_to_align_skeleton_with_positive_y = calculate_rotation_matrix(unit_translated_and_rotated_heel_vector,-1*x_vector)

origin_aligned_skeleton_data = np.zeros(skeleton_data.shape)

for frame in track(range(num_frames)):
   origin_aligned_skeleton_data[frame,:,:] = rotate_skeleton_frame(translated_and_rotated_skeleton_data[frame,:,:],rotation_matrix_to_align_skeleton_with_positive_y)

origin_aligned_base_frame_skeleton_data = origin_aligned_skeleton_data[base_frame,:,:]

#origin_aligned_normal_vector_to_left_foot, origin_aligned_left_foot_vector, origin_aligned_heel_vector = calculate_normal_vector_to_foot(left_heel_index,left_toe_index,right_heel_index,origin_aligned_base_frame_skeleton_data)
origin_aligned_normal_vector_to_left_foot, origin_aligned_left_foot_vector, origin_aligned_heel_vector = calculate_normal_vector_to_foot(primary_foot_indices[0],primary_foot_indices[1], secondary_foot_index[0],origin_aligned_base_frame_skeleton_data)

f = 2


if debug:

    #debug plot shows the X,Y,Z vectors pointing out from the origin - the right heel should be at 0,0,0. 
    #the z-vector should align with the normal vector between the right foot vector and heel vector
    #the x-vector should align with the vector between the left heel and right heel 
    
    def plot_origin_vectors(plot_ax,x_vector,y_vector,z_vector,origin):
        Zvector_X,Zvector_Y,Zvector_Z = zip(origin_normal_unit_vector*800)
        Xvector_X,Xvector_Y,Xvector_Z = zip(x_vector*800)
        Yvector_X,Yvector_Y,Yvector_Z = zip(y_vector*800)

        Origin_X,Origin_Y,Origin_Z = zip(origin)

        plot_ax.quiver(Origin_X,Origin_Y,Origin_Z,Zvector_X,Zvector_Y,Zvector_Z,arrow_length_ratio=0.1,color='b', label = 'Z-axis')
        plot_ax.quiver(Origin_X,Origin_Y,Origin_Z,Xvector_X,Xvector_Y,Xvector_Z,arrow_length_ratio=0.1,color='r', label = 'X-axis')
        plot_ax.quiver(Origin_X,Origin_Y,Origin_Z,Yvector_X,Yvector_Y,Yvector_Z,arrow_length_ratio=0.1,color='g', label = 'Y-axis')

    def calculate_COM(skeleton_data):
        COM_XYZ = np.nanmean(skeleton_data,axis=0)
        return COM_XYZ

    def calculate_COM_ground_projection_y(center_of_mass_XYZ, skeleton_data):
        projection_distance = center_of_mass_XYZ[1] - skeleton_data[right_heel_index,1]
        COM_ground_projected_XYZ = center_of_mass_XYZ - projection_distance*np.array([0,1,0])

        return COM_ground_projected_XYZ

    def calculate_COM_ground_projection_z(center_of_mass_XYZ, skeleton_data):
        projection_distance = center_of_mass_XYZ[2] - skeleton_data[right_heel_index,2]
        COM_ground_projected_XYZ = center_of_mass_XYZ - projection_distance*np.array([0,0,1])
        
        return COM_ground_projected_XYZ


    def plot_normal_unit_vector_to_foot(normal_vector_to_foot, origin_foot_index, skeleton_data,plot_ax):

        normal_vector_to_foot_unit_vector = create_unit_vector(normal_vector_to_foot)

        normal_vector_to_foot_X,normal_vector_to_foot_Y,normal_vector_to_foot_Z = zip(normal_vector_to_foot_unit_vector*800)
        plot_ax.quiver(skeleton_data[origin_foot_index,0],skeleton_data[origin_foot_index,1],skeleton_data[origin_foot_index,2],normal_vector_to_foot_X,normal_vector_to_foot_Y,normal_vector_to_foot_Z,arrow_length_ratio=0.1,color='pink')

    def set_axes_ranges(plot_ax,skeleton_data, ax_range):

        mx = np.nanmean(skeleton_data[:,0])
        my = np.nanmean(skeleton_data[:,1])
        mz = np.nanmean(skeleton_data[:,2])
    
        plot_ax.set_xlim(mx-ax_range,mx+ax_range)
        plot_ax.set_ylim(my-ax_range,my+ax_range)
        plot_ax.set_zlim(mz-ax_range,mz+ax_range)

    def plot_COM_point_and_projection(plot_ax,COM_XYZ,COM_ground_projected_XYZ):

        plot_ax.scatter(COM_XYZ[0],COM_XYZ[1],COM_XYZ[2],color='b')
        plot_ax.scatter(COM_ground_projected_XYZ[0],COM_ground_projected_XYZ[1],COM_ground_projected_XYZ[2], color = 'b')
        plot_ax.plot([COM_XYZ[0],COM_ground_projected_XYZ[0]],[COM_XYZ[1],COM_ground_projected_XYZ[1]],[COM_XYZ[2],COM_ground_projected_XYZ[2]],color='b', alpha = .5)




    figure = plt.figure()
    ax1 = figure.add_subplot(221,projection = '3d')
    ax2 = figure.add_subplot(222,projection = '3d')
    ax3 = figure.add_subplot(223,projection = '3d')
    ax4 = figure.add_subplot(224,projection = '3d')

    ax1.set_title('Original Skeleton')
    ax2.set_title('Skeleton Translated to Origin')
    ax3.set_title('Skeleton Rotated to Make +Z Up')
    ax4.set_title('Skeleton Rotated to Make +Y Forwards')

    rotation_base_frame = 349

    ax_range = 1800

    set_axes_ranges(ax1,base_frame_skeleton_data,ax_range)
    set_axes_ranges(ax2,translated_base_frame_skeleton_data,ax_range)
    set_axes_ranges(ax3,translated_and_rotated_base_frame_skeleton_data,ax_range)
    set_axes_ranges(ax4,origin_aligned_base_frame_skeleton_data,ax_range)

    ax1.scatter(base_frame_skeleton_data[:,0],base_frame_skeleton_data[:,1],base_frame_skeleton_data[:,2],c='r')
    plot_origin_vectors(ax1,x_vector,y_vector,z_vector,origin)

    original_COM_XYZ = calculate_COM(base_frame_skeleton_data[0:num_pose_joints,:])
    original_COM_XYZ_ground_projection = calculate_COM_ground_projection_y(original_COM_XYZ,base_frame_skeleton_data)

    plot_COM_point_and_projection(ax1,original_COM_XYZ,original_COM_XYZ_ground_projection)
    plot_normal_unit_vector_to_foot(base_frame_normal_vector_to_foot,primary_foot_indices[0],base_frame_skeleton_data,ax1)

    ax2.scatter(translated_base_frame_skeleton_data[:,0],translated_base_frame_skeleton_data[:,1],translated_base_frame_skeleton_data[:,2],c='g')
    plot_origin_vectors(ax2,x_vector,y_vector,z_vector,origin)

    translated_COM_XYZ = calculate_COM(translated_base_frame_skeleton_data[0:num_pose_joints,:])
    translated_COM_XYZ_ground_projection = calculate_COM_ground_projection_y(translated_COM_XYZ,translated_base_frame_skeleton_data)

    plot_COM_point_and_projection(ax2,translated_COM_XYZ,translated_COM_XYZ_ground_projection)
    plot_normal_unit_vector_to_foot(translated_normal_vector_to_left_foot,primary_foot_indices[0],translated_base_frame_skeleton_data,ax2)
    # ax2.quiver(translated_right_heel_x,translated_right_heel_y,translated_right_heel_z,translated_foot_x,translated_foot_y,translated_foot_z,arrow_length_ratio=0.1,color='pink')

    ax3.scatter(translated_and_rotated_base_frame_skeleton_data[:,0],translated_and_rotated_base_frame_skeleton_data[:,1],translated_and_rotated_base_frame_skeleton_data[:,2],c='orange')
    plot_origin_vectors(ax3,x_vector,y_vector,z_vector,origin)

    z_rotated_COM_XYZ = calculate_COM(translated_and_rotated_base_frame_skeleton_data[0:num_pose_joints,:])
    z_rotated_COM_XYZ_ground_projection = calculate_COM_ground_projection_z(z_rotated_COM_XYZ,translated_and_rotated_base_frame_skeleton_data)

    plot_COM_point_and_projection(ax3,z_rotated_COM_XYZ,z_rotated_COM_XYZ_ground_projection)
    plot_normal_unit_vector_to_foot(translated_and_rotated_normal_vector_to_left_foot,primary_foot_indices[0],translated_and_rotated_base_frame_skeleton_data,ax3)
    # ax3.quiver(rotated_right_heel_x,rotated_right_heel_y,rotated_right_heel_z,rotated_foot_x,rotated_foot_y,rotated_foot_z,arrow_length_ratio=0.1,color='pink')
    # ax3.quiver(rotated_right_heel_x,rotated_right_heel_y,rotated_right_heel_z,rotated_left_heel_x,rotated_left_heel_y,rotated_left_heel_z,arrow_length_ratio=0.1,color='pink')

    ax4.scatter(origin_aligned_base_frame_skeleton_data[:,0],origin_aligned_base_frame_skeleton_data[:,1],origin_aligned_base_frame_skeleton_data[:,2],c='purple')
    plot_origin_vectors(ax4,x_vector,y_vector,z_vector,origin)

    origin_aligned_COM_XYZ = calculate_COM(origin_aligned_base_frame_skeleton_data[0:num_pose_joints,:])
    origin_aligned_COM_XYZ_ground_projection = calculate_COM_ground_projection_z(origin_aligned_COM_XYZ,origin_aligned_base_frame_skeleton_data)
    
    plot_COM_point_and_projection(ax4,origin_aligned_COM_XYZ,origin_aligned_COM_XYZ_ground_projection)
    plot_normal_unit_vector_to_foot(origin_aligned_normal_vector_to_left_foot,left_heel_index,origin_aligned_base_frame_skeleton_data,ax4)
     # ax4.quiver(origin_aligned_right_heel_x,origin_aligned_right_heel_y,origin_aligned_right_heel_z,origin_aligned_left_heel_x,origin_aligned_left_heel_y,origin_aligned_left_heel_z,arrow_length_ratio=0.1,color='pink')
  


    ax1.legend()
    ax2.legend()
    ax3.legend()
    plt.show()

#save the aligned skeleton data to a new file
np.save(save_file,origin_aligned_skeleton_data)
f = 2