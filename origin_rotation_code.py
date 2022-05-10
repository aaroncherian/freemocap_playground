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
    freemocap_validation_data_path = Path(r'D:\freemocap2022\FreeMocap_Data')
else:
    #freemocap_validation_data_path = Path(r"C:\Users\kiley\Documents\HumonLab\SampleFMC_Data\FreeMocap_Data-20220216T173514Z-001\FreeMocap_Data")
    freemocap_validation_data_path = Path(r"C:\Users\Rontc\Documents\HumonLab\ValidationStudy")

    
sessionID = 'sesh_2022-05-09_15_40_59' #name of the sessionID folder
skeleton_to_plot = 'mediapipe' #for a future situation where we want to rotate openpose/dlc skeletons 
rotation_base_frame = 349
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

    num_pose_joints = 33

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

origin = np.array([0, 0, 0])
x_axis = np.array([1, 0, 0])
y_axis = np.array([0, 1, 0])
z_axis = np.array([0, 0, 1])

x_vector = create_vector(origin,x_axis)
y_vector = create_vector(origin,y_axis)
z_vector = create_vector(origin,z_axis)

#origin_normal = create_normal_vector(x_vector,y_vector) #create a normal vector to the origin (basically the z axis)
origin_normal_unit_vector = z_vector  #note - this is kinda unncessary because the origin normal unit vector == original normal vector 


rotation_frame_skeleton_data = skeleton_data[rotation_base_frame,:,:] #get the skeleton data for the frame we want to calculate the rotation matrix from

num_frames = skeleton_data.shape[0]

#get the translation distance between the right heel and the origin on the frame we used to build the rotation matrix
translation_distance = calculate_translation_distance(rotation_frame_skeleton_data[right_heel_index,:])

translated_skeleton_data = np.zeros(skeleton_data.shape)


for frame in track(range(num_frames)):
   translated_skeleton_data[frame,:,:] = translate_skeleton_frame(skeleton_data[frame,:,:],translation_distance) #translate the skeleton data for each frame  






rotation_frame_skeleton_data = translated_skeleton_data[rotation_base_frame,:,:] 

right_foot_origin = rotation_frame_skeleton_data[right_heel_index,:]
left_foot_origin = rotation_frame_skeleton_data[left_heel_index,:]

#create vectors for the right foot and left foot, and between the two heels 
right_foot_vector = create_vector(rotation_frame_skeleton_data[right_heel_index,:],rotation_frame_skeleton_data[right_toe_index,:])
left_foot_vector = create_vector(rotation_frame_skeleton_data[left_heel_index,:],rotation_frame_skeleton_data[left_toe_index,:])
heel_vector = create_vector(rotation_frame_skeleton_data[right_heel_index,:],rotation_frame_skeleton_data[left_heel_index,:])


#create a normal unit vector from the right foot and heel vector
foot_normal =  create_normal_vector(right_foot_vector,heel_vector)
foot_normal_unit_vector = create_unit_vector(foot_normal)

#calculate the rotation matrix between the origin normal and the foot normal
rotation_matrix = calculate_rotation_matrix(foot_normal_unit_vector,origin_normal_unit_vector)

translated_and_rotated_skeleton_data = np.zeros(skeleton_data.shape) #create an array to hold the rotated skeleton data
num_frames = translated_skeleton_data.shape[0]

for frame in track(range(num_frames)): #rotate the skeleton on each frame 
    translated_and_rotated_skeleton_data[frame,:,:] = rotate_skeleton_frame(translated_skeleton_data[frame,:,:],rotation_matrix)

translated_and_rotated_heel_vector = create_vector(translated_and_rotated_skeleton_data[rotation_base_frame,right_heel_index,:],translated_and_rotated_skeleton_data[rotation_base_frame,left_heel_index,:])
translated_and_rotated_heel_unit_vector = create_unit_vector(translated_and_rotated_heel_vector)

rotation_matrix_to_align_skeleton_with_positive_y = calculate_rotation_matrix(translated_and_rotated_heel_unit_vector,x_vector*-1)

origin_aligned_skeleton_data = np.zeros(skeleton_data.shape)

for frame in track(range(num_frames)):
   origin_aligned_skeleton_data[frame,:,:] = rotate_skeleton_frame(translated_and_rotated_skeleton_data[frame,:,:],rotation_matrix_to_align_skeleton_with_positive_y)


# right_foot_origin = rotation_frame_skeleton_data[right_heel_index,:]
# left_foot_origin = rotation_frame_skeleton_data[left_heel_index,:]

# #create vectors for the right foot and left foot, and between the two heels 
# right_foot_vector = create_vector(rotation_frame_skeleton_data[right_heel_index,:],rotation_frame_skeleton_data[right_toe_index,:])
# left_foot_vector = create_vector(rotation_frame_skeleton_data[left_heel_index,:],rotation_frame_skeleton_data[left_toe_index,:])
# heel_vector = create_vector(rotation_frame_skeleton_data[right_heel_index,:],rotation_frame_skeleton_data[left_heel_index,:])

# #create a normal unit vector from the right foot and heel vector
# foot_normal =  create_normal_vector(right_foot_vector,heel_vector)
# foot_normal_unit_vector = create_unit_vector(foot_normal)

# #right_foot_unit_vector = create_unit_vector(right_foot_vector)
# #heel_unit_vector = create_unit_vector(heel_vector)

# #calculate the rotation matrix between the origin normal and the foot normal
# rotation_matrix = calculate_rotation_matrix(foot_normal_unit_vector,origin_normal_unit_vector)




# rotated_skeleton_data = np.zeros(skeleton_data.shape) #create an array to hold the rotated skeleton data
# num_frames = skeleton_data.shape[0]

# for frame in track(range(num_frames)): #rotate the skeleton on each frame 
#     rotated_skeleton_data[frame,:,:] = rotate_skeleton_frame(skeleton_data[frame,:,:],rotation_matrix)



# #origin_aligned_skeleton_data = translated_and_rotated_skeleton_data

# # #to get the final alignment (to face the person forward in +y), get the new heel unit vector 
# # translated_and_rotated_heel_vector = create_vector(translated_and_rotated_skeleton_data[rotation_base_frame,left_heel_index,:],translated_and_rotated_skeleton_data[rotation_base_frame,right_heel_index,:])
# # translated_and_rotated_heel_unit_vector = create_unit_vector(translated_and_rotated_heel_vector)

# # #find the rotation matrix between the new heel unit vector and the x-axis
# # rotation_matrix_to_align_skeleton_with_positive_y = calculate_rotation_matrix(translated_and_rotated_heel_unit_vector,x_vector)

# # origin_aligned_skeleton_data = np.zeros(skeleton_data.shape)

# # #rotate the skeleton in each frame with the new rotation matrix 
# # for frame in track(range(num_frames)):
# #    origin_aligned_skeleton_data[frame,:,:] = rotate_skeleton_frame(translated_and_rotated_skeleton_data[frame,:,:],rotation_matrix_to_align_skeleton_with_positive_y)


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

    def get_foot_normal_vector(skeleton_data,right_heel_index,left_heel_index,right_toe_index):
        right_foot_vector = create_vector(skeleton_data[right_heel_index,:],skeleton_data[right_toe_index,:])
        heel_vector = create_vector(skeleton_data[right_heel_index,:],skeleton_data[left_heel_index,:])
        right_foot_normal_vector =  create_normal_vector(right_foot_vector,heel_vector)

        return right_foot_normal_vector

    def set_axes_ranges(plot_ax,skeleton_data, ax_range):

        mx = np.nanmean(skeleton_data[:,0])
        my = np.nanmean(skeleton_data[:,1])
        mz = np.nanmean(skeleton_data[:,2])
    
        plot_ax.set_xlim(mx-ax_range,mx+ax_range)
        plot_ax.set_ylim(my-ax_range,my+ax_range)
        plot_ax.set_zlim(mz-ax_range,mz+ax_range)

    figure = plt.figure()
    ax1 = figure.add_subplot(221,projection = '3d')
    ax2 = figure.add_subplot(222,projection = '3d')
    ax3 = figure.add_subplot(223,projection = '3d')
    ax4 = figure.add_subplot(224,projection = '3d')

    this_frame_original_skeleton_data = skeleton_data[rotation_base_frame,:,:]
    this_frame_translated_skeleton_data = translated_skeleton_data[rotation_base_frame,:,:]
    this_frame_z_rotated_skeleton_data = translated_and_rotated_skeleton_data[rotation_base_frame,:,:]
    this_frame_origin_aligned_skeleton_data = origin_aligned_skeleton_data[rotation_base_frame,:,:]

    translated_right_foot_normal_vector = get_foot_normal_vector(this_frame_translated_skeleton_data,right_heel_index,left_heel_index,right_toe_index)
    translated_right_foot_normal_unit_vector = create_unit_vector(translated_right_foot_normal_vector)

    translated_right_heel_x, translated_right_heel_y, translated_right_heel_z = zip(this_frame_translated_skeleton_data[right_heel_index,:])
    translated_foot_x, translated_foot_y, translated_foot_z = zip(translated_right_foot_normal_unit_vector*500)

    rotated_right_foot_normal_vector = get_foot_normal_vector(this_frame_z_rotated_skeleton_data,right_heel_index,left_heel_index,right_toe_index)
    rotated_right_foot_normal_unit_vector = create_unit_vector(rotated_right_foot_normal_vector)

    rotated_right_heel_x, rotated_right_heel_y, rotated_right_heel_z = zip(this_frame_z_rotated_skeleton_data[right_heel_index,:])
    rotated_left_heel_x, rotated_left_heel_y, rotated_left_heel_z = zip(this_frame_z_rotated_skeleton_data[left_heel_index,:])
    rotated_foot_x, rotated_foot_y, rotated_foot_z = zip(rotated_right_foot_normal_unit_vector*500)

    origin_aligned_right_foot_normal_vector = get_foot_normal_vector(this_frame_origin_aligned_skeleton_data,right_heel_index,left_heel_index,right_toe_index)
    origin_aligned_right_foot_normal_unit_vector = create_unit_vector(origin_aligned_right_foot_normal_vector)

    origin_aligned_right_heel_x, origin_aligned_right_heel_y, origin_aligned_right_heel_z = zip(this_frame_origin_aligned_skeleton_data[right_heel_index,:])
    origin_aligned_left_heel_x, origin_aligned_left_heel_y, origin_aligned_left_heel_z = zip(this_frame_origin_aligned_skeleton_data[left_heel_index,:])
    origin_aligned_foot_x, origin_aligned_foot_y, origin_aligned_foot_z = zip(origin_aligned_right_foot_normal_unit_vector*500)
    #plot the origin vectors



#     Origin_X,Origin_Y,Origin_Z = zip(origin)


    ax_range = 1800

    set_axes_ranges(ax1,this_frame_original_skeleton_data,ax_range)
    set_axes_ranges(ax2,this_frame_translated_skeleton_data,ax_range)
    set_axes_ranges(ax3,this_frame_z_rotated_skeleton_data,ax_range)
    set_axes_ranges(ax4,this_frame_origin_aligned_skeleton_data,ax_range)

    ax1.scatter(this_frame_original_skeleton_data[:,0],this_frame_original_skeleton_data[:,1],this_frame_original_skeleton_data[:,2],c='r')
    plot_origin_vectors(ax1,x_vector,y_vector,z_vector,origin)

    original_COM_XYZ = calculate_COM(this_frame_original_skeleton_data[0:num_pose_joints,:])
    original_COM_XYZ_ground_projection = calculate_COM_ground_projection_y(original_COM_XYZ,this_frame_original_skeleton_data)

    ax1.scatter(original_COM_XYZ[0],original_COM_XYZ[1],original_COM_XYZ[2],c='b')
    ax1.scatter(original_COM_XYZ_ground_projection[0],original_COM_XYZ_ground_projection[1],original_COM_XYZ_ground_projection[2],c='b')
    ax1.plot([original_COM_XYZ[0],original_COM_XYZ_ground_projection[0]],[original_COM_XYZ[1],original_COM_XYZ_ground_projection[1]],[original_COM_XYZ[2],original_COM_XYZ_ground_projection[2]],c='b', alpha = .5)

    ax2.scatter(this_frame_translated_skeleton_data[:,0],this_frame_translated_skeleton_data[:,1],this_frame_translated_skeleton_data[:,2],c='g')
    plot_origin_vectors(ax2,x_vector,y_vector,z_vector,origin)

    translated_COM_XYZ = calculate_COM(this_frame_translated_skeleton_data[0:num_pose_joints,:])
    translated_COM_XYZ_ground_projection = calculate_COM_ground_projection_y(translated_COM_XYZ,this_frame_translated_skeleton_data)

    ax2.scatter(translated_COM_XYZ[0],translated_COM_XYZ[1],translated_COM_XYZ[2],c='b')
    ax2.scatter(translated_COM_XYZ_ground_projection[0],translated_COM_XYZ_ground_projection[1],translated_COM_XYZ_ground_projection[2],c='b')
    ax2.plot([translated_COM_XYZ[0],translated_COM_XYZ_ground_projection[0]],[translated_COM_XYZ[1],translated_COM_XYZ_ground_projection[1]],[translated_COM_XYZ[2],translated_COM_XYZ_ground_projection[2]],c='b', alpha = .5)

    ax2.quiver(translated_right_heel_x,translated_right_heel_y,translated_right_heel_z,translated_foot_x,translated_foot_y,translated_foot_z,arrow_length_ratio=0.1,color='pink')

    ax3.scatter(this_frame_z_rotated_skeleton_data[:,0],this_frame_z_rotated_skeleton_data[:,1],this_frame_z_rotated_skeleton_data[:,2],c='orange')
    plot_origin_vectors(ax3,x_vector,y_vector,z_vector,origin)

    z_rotated_COM_XYZ = calculate_COM(this_frame_z_rotated_skeleton_data[0:num_pose_joints,:])
    z_rotated_COM_XYZ_ground_projection = calculate_COM_ground_projection_z(z_rotated_COM_XYZ,this_frame_z_rotated_skeleton_data)

    ax3.scatter(z_rotated_COM_XYZ[0],z_rotated_COM_XYZ[1],z_rotated_COM_XYZ[2],c='b', alpha = .5)
    ax3.scatter(z_rotated_COM_XYZ_ground_projection[0],z_rotated_COM_XYZ_ground_projection[1],z_rotated_COM_XYZ_ground_projection[2],c='b',alpha = .5)
    ax3.plot([z_rotated_COM_XYZ[0],z_rotated_COM_XYZ_ground_projection[0]],[z_rotated_COM_XYZ[1],z_rotated_COM_XYZ_ground_projection[1]],[z_rotated_COM_XYZ[2],z_rotated_COM_XYZ_ground_projection[2]],c='b', alpha = .5)
    
    ax3.quiver(rotated_right_heel_x,rotated_right_heel_y,rotated_right_heel_z,rotated_foot_x,rotated_foot_y,rotated_foot_z,arrow_length_ratio=0.1,color='pink')
    ax3.quiver(rotated_right_heel_x,rotated_right_heel_y,rotated_right_heel_z,rotated_left_heel_x,rotated_left_heel_y,rotated_left_heel_z,arrow_length_ratio=0.1,color='pink')

    ax4.scatter(this_frame_origin_aligned_skeleton_data[:,0],this_frame_origin_aligned_skeleton_data[:,1],this_frame_origin_aligned_skeleton_data[:,2],c='purple')
    plot_origin_vectors(ax4,x_vector,y_vector,z_vector,origin)

    origin_aligned_COM_XYZ = calculate_COM(this_frame_origin_aligned_skeleton_data[0:num_pose_joints,:])
    origin_aligned_COM_XYZ_ground_projection = calculate_COM_ground_projection_z(origin_aligned_COM_XYZ,this_frame_origin_aligned_skeleton_data)
    
    ax4.scatter(origin_aligned_COM_XYZ[0],origin_aligned_COM_XYZ[1],origin_aligned_COM_XYZ[2],c='b')
    ax4.scatter(origin_aligned_COM_XYZ_ground_projection[0],origin_aligned_COM_XYZ_ground_projection[1],origin_aligned_COM_XYZ_ground_projection[2],c='b')
    ax4.plot([origin_aligned_COM_XYZ[0],origin_aligned_COM_XYZ_ground_projection[0]],[origin_aligned_COM_XYZ[1],origin_aligned_COM_XYZ_ground_projection[1]],[origin_aligned_COM_XYZ[2],origin_aligned_COM_XYZ_ground_projection[2]],c='b', alpha = .5)
    ax4.quiver(origin_aligned_right_heel_x,origin_aligned_right_heel_y,origin_aligned_right_heel_z,origin_aligned_left_heel_x,origin_aligned_left_heel_y,origin_aligned_left_heel_z,arrow_length_ratio=0.1,color='pink')
    ax4.quiver(origin_aligned_right_heel_x,origin_aligned_right_heel_y,origin_aligned_right_heel_z,origin_aligned_foot_x,origin_aligned_foot_y,origin_aligned_foot_z,arrow_length_ratio=0.1,color='pink')

    ax1.legend()
    ax2.legend()
    ax3.legend()
    plt.show()

#     this_frame_aligned_skeleton_data = origin_aligned_skeleton_data[rotation_base_frame,:,:]
#     this_frame_original_skeleton_data = skeleton_data[rotation_base_frame,:,:]

#     ax_range = 1800
#     mx = np.nanmean(this_frame_aligned_skeleton_data[:,0])
#     my = np.nanmean(this_frame_aligned_skeleton_data[:,1])
#     mz = np.nanmean(this_frame_aligned_skeleton_data[:,2])

#     ax.set_xlim([mx-ax_range, mx+ax_range]) #maybe set ax limits before the function? if we're using cla() they probably don't need to be redefined every time 
#     ax.set_ylim([my-ax_range, my+ax_range])
#     ax.set_zlim([mz-ax_range, mz+ax_range])

#     mx2 = np.nanmean(this_frame_original_skeleton_data[:,0])
#     my2 = np.nanmean(this_frame_original_skeleton_data[:,1])
#     mz2 = np.nanmean(this_frame_original_skeleton_data[:,2])

#     ax2.set_xlim([mx2-ax_range, mx2+ax_range]) #maybe set ax limits before the function? if we're using cla() they probably don't need to be redefined every time 
#     ax2.set_ylim([my2-ax_range, my2+ax_range])
#     ax2.set_zlim([mz2-ax_range, mz2+ax_range])
#     ax2.view_init(elev = -90, azim = 180)

#     ax.scatter(this_frame_aligned_skeleton_data[:,0],this_frame_aligned_skeleton_data[:,1],this_frame_aligned_skeleton_data[:,2],c='r')

#     ax2.scatter(this_frame_original_skeleton_data[:,0],this_frame_original_skeleton_data[:,1],this_frame_original_skeleton_data[:,2],c='b')
    
#     this_frame_right_foot_vector = create_vector(this_frame_aligned_skeleton_data[right_heel_index,:] ,this_frame_aligned_skeleton_data[right_toe_index,:])
    
#     this_frame_heel_vector = create_vector(this_frame_aligned_skeleton_data[right_heel_index,:] ,this_frame_aligned_skeleton_data[left_heel_index,:]) #this is the heel vector for aligning the right heel with the origin
#     this_frame_x_alignment_heel_vector = create_vector(this_frame_aligned_skeleton_data[left_heel_index,:] ,this_frame_aligned_skeleton_data[right_heel_index,:]) #this is the heel vector for aligning the skeleton to face positive y
# #NOTE - maybe use the left heel as the origin for the rotation as well 

#     this_frame_heel_unit_vector = create_unit_vector(this_frame_heel_vector)
#     this_frame_heel_alignment_unit_vector = create_unit_vector(this_frame_x_alignment_heel_vector)

#     this_frame_foot_normal_vector = create_normal_vector(this_frame_right_foot_vector,this_frame_heel_vector)
#     this_frame_foot_normal_unit_vector = create_unit_vector(this_frame_foot_normal_vector)

#     #W,U,V = zip(this_frame_foot_unit_vector)

#     Zvector_X,Zvector_Y,Zvector_Z = zip(origin_normal_unit_vector*800)
#     Xvector_X,Xvector_Y,Xvector_Z = zip(x_vector*800)
#     Yvector_X,Yvector_Y,Yvector_Z = zip(y_vector*800)

#     Origin_X,Origin_Y,Origin_Z = zip(origin)

#     Rightheel_X, Rightheel_Y, Rightheel_Z = zip(this_frame_aligned_skeleton_data[right_heel_index,:]) 
#     Footnormal_X,Footnormal_Y,Footnormal_Z = zip(this_frame_foot_normal_unit_vector*500)
#     Heel_X, Heel_Y, Heel_Z = zip(this_frame_heel_unit_vector*500)
#     Heel_alignment_X, Heel_alignment_Y, Heel_alignment_Z = zip(this_frame_heel_alignment_unit_vector*500)

#     ax.quiver(Origin_X,Origin_Y,Origin_Z,Zvector_X,Zvector_Y,Zvector_Z,arrow_length_ratio=0.1,color='b', label = 'Z-axis')
#     ax.quiver(Origin_X,Origin_Y,Origin_Z,Xvector_X,Xvector_Y,Xvector_Z,arrow_length_ratio=0.1,color='cyan', label = 'X-axis')
#     ax.quiver(Origin_X,Origin_Y,Origin_Z,Yvector_X,Yvector_Y,Yvector_Z,arrow_length_ratio=0.1,color='purple', label = 'Y-axis')

#     ax.quiver(Rightheel_X,Rightheel_Y,Rightheel_Z,Footnormal_X,Footnormal_Y,Footnormal_Z,arrow_length_ratio=0.1,color='r')
#     ax.quiver(Rightheel_X,Rightheel_Y,Rightheel_Z,Heel_alignment_X,Heel_alignment_Y,Heel_alignment_Z,arrow_length_ratio=0.1,color='g') 

#     ax.legend()

#     plt.show()

#save the aligned skeleton data to a new file
np.save(save_file,origin_aligned_skeleton_data)
f = 2