
from pathlib import Path 

import numpy as np
import matplotlib.pyplot as plt
from rich.progress import track

from skeleton_data_holder import SkeletonDataHolder



def create_vector(point1,point2): 
    """Put two points in, make a vector"""
    vector = point2 - point1
    return vector

def calculate_translation_distance(skeleton_point_coordinate):
    """Take a skeleton point coordinate and calculate its distance to the origin"""

    translation_distance = skeleton_point_coordinate - [0,0,0]
    return translation_distance 


def translate_skeleton_frame(rotated_skeleton_data_frame, translation_distance):
    """Take in a frame of rotated skeleton data, and apply the translation distance to each point in the skeleton"""

    translated_skeleton_frame = rotated_skeleton_data_frame - translation_distance
    return translated_skeleton_frame

def calculate_skewed_symmetric_cross_product(cross_product_vector):
    #needed in the calculate_rotation_matrix function 
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


def align_skeleton_with_origin( skeleton_data, skeleton_indices, good_frame, debug = False):

    """
    Takes in freemocap skeleton data and translates the skeleton to the origin, and then rotates the data 
    so that the skeleton is facing the +y direction and standing in the +z direction

    Input:
        skeleton data: a 3D numpy array of skeleton data in freemocap format
        skeleton indices: a list of joints being tracked by mediapipe/your 2d pose estimator
        good frame: the frame that you want to base the rotation on (can be entered manually, 
                    or use the 'good_frame_finder.py' to calculate it)
        debug: If 'True', display a plot of the raw data and the 3 main alignment stages

    Output:
        spine aligned skeleton data: a 3d numpy array of the origin aligned data in freemocap format 
    """

    origin = np.array([0, 0, 0])
    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])

    x_vector = create_vector(origin,x_axis)
    y_vector = create_vector(origin,y_axis)
    z_vector = create_vector(origin,z_axis)

    origin_normal_unit_vector = z_vector  #note - this is kinda unncessary because the origin normal unit vector == original normal vector 

    num_frames = skeleton_data.shape[0]

    #Put the raw, unaligned skeleton data in the SkeletonDataHolder class 
    raw_skeleton_holder = SkeletonDataHolder(skeleton_data, skeleton_indices, good_frame)

    ## Translate the raw data such that the midpoint of the hips is over the origin 
    raw_mid_hip_XYZ = raw_skeleton_holder.mid_hip_XYZ #hip midpoint of the raw data 
    mid_hip_translation_distance = calculate_translation_distance(raw_mid_hip_XYZ) #get distance between hip midpoint and origin

    hip_translated_skeleton_data = np.zeros(skeleton_data.shape)
    for frame in track(range(num_frames)):
        hip_translated_skeleton_data[frame,:,:] = translate_skeleton_frame(skeleton_data[frame,:,:],mid_hip_translation_distance) #translate the skeleton data for each frame  
    hip_translated_skeleton_holder = SkeletonDataHolder(hip_translated_skeleton_data, skeleton_indices, good_frame)
    
    ## Now translate the data upwards, such that the midpoint between the two feet is at the origin 
    hip_translated_mid_foot_XYZ = hip_translated_skeleton_holder.mid_foot_XYZ
    mid_foot_translated_distance = calculate_translation_distance(hip_translated_mid_foot_XYZ)

    foot_translated_skeleton_data = np.zeros(skeleton_data.shape)
    for frame in track (range(num_frames)):
        foot_translated_skeleton_data[frame,:,:] = translate_skeleton_frame(hip_translated_skeleton_data[frame,:,:],mid_foot_translated_distance)
    foot_translated_skeleton_holder = SkeletonDataHolder(foot_translated_skeleton_data, skeleton_indices, good_frame)

    # Rotate the skeleton to face the +y direction
    foot_translated_heel_unit_vector = foot_translated_skeleton_holder.heel_unit_vector
    rotation_matrix_to_align_skeleton_with_positive_y = calculate_rotation_matrix(foot_translated_heel_unit_vector,-1*x_vector)

    y_aligned_skeleton_data = np.zeros(skeleton_data.shape)
    for frame in track(range(num_frames)):
        y_aligned_skeleton_data [frame,:,:] = rotate_skeleton_frame(foot_translated_skeleton_data[frame,:,:],rotation_matrix_to_align_skeleton_with_positive_y)
    y_aligned_skeleton_holder = SkeletonDataHolder(y_aligned_skeleton_data, skeleton_indices, good_frame)
  
    #Rotating the skeleton so that the spine is aligned with +z
    y_aligned_spine_unit_vector = y_aligned_skeleton_holder.spine_unit_vector
    rotation_matrix_to_align_spine = calculate_rotation_matrix(y_aligned_spine_unit_vector,origin_normal_unit_vector)

    spine_aligned_skeleton_data = np.zeros(skeleton_data.shape)
    for frame in track(range(num_frames)):
        spine_aligned_skeleton_data [frame,:,:] = rotate_skeleton_frame(y_aligned_skeleton_data[frame,:,:],rotation_matrix_to_align_spine)
    spine_aligned_skeleton_holder = SkeletonDataHolder(spine_aligned_skeleton_data, skeleton_indices, good_frame)


    if debug:
            def plot_origin_vectors(plot_ax,x_vector,y_vector,z_vector,origin):
                Zvector_X,Zvector_Y,Zvector_Z = zip(origin_normal_unit_vector*800)
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

            def plot_spine_unit_vector(plot_ax,skeleton_data,skeleton_mid_hip_XYZ,skeleton_spine_unit_vector):

                skeleton_spine_unit_x, skeleton_spine_unit_y, skeleton_spine_unit_z = zip(skeleton_spine_unit_vector*800)

                plot_ax.quiver(skeleton_mid_hip_XYZ[0],skeleton_mid_hip_XYZ[1],skeleton_mid_hip_XYZ[2], skeleton_spine_unit_x,skeleton_spine_unit_y,skeleton_spine_unit_z,arrow_length_ratio=0.1,color='pink')

            def plot_heel_unit_vector(plot_ax,skeleton_heel_unit_vector, heel_vector_origin_XYZ):

                origin_heel_x, origin_heel_y, origin_heel_z = zip(heel_vector_origin_XYZ)
                skeleton_heel_unit_x, skeleton_heel_unit_y, skeleton_heel_unit_z = zip(skeleton_heel_unit_vector*500)

                plot_ax.quiver(origin_heel_x,origin_heel_y,origin_heel_z, skeleton_heel_unit_x,skeleton_heel_unit_y,skeleton_heel_unit_z,arrow_length_ratio=0.1,color='orange')

            
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

      

            #grab the skeleton data for each plot for the frame we are plotting
            raw_good_frame_skeleton_data = raw_skeleton_holder.good_frame_skeleton_data
            foot_translated_good_frame_skeleton_data = foot_translated_skeleton_holder.good_frame_skeleton_data
            y_aligned_good_frame_skeleton_data = y_aligned_skeleton_holder.good_frame_skeleton_data
            spine_aligned_good_frame_skeleton_data = spine_aligned_skeleton_holder.good_frame_skeleton_data 
            good_frame_data_arrays_list = [raw_good_frame_skeleton_data,foot_translated_good_frame_skeleton_data, y_aligned_good_frame_skeleton_data, spine_aligned_good_frame_skeleton_data]
            
            ax_range = 1800
            for ax,skeleton_data_array in zip(axes_list,good_frame_data_arrays_list):
                set_axes_ranges(ax,skeleton_data_array,ax_range)
                plot_origin_vectors(ax,x_vector,y_vector,z_vector,origin)
                ax.scatter(skeleton_data_array[:,0], skeleton_data_array[:,1], skeleton_data_array[:,2], c = 'r')

            ##Plot the spine vector on ax1
            # raw_spine_unit_vector = raw_skeleton_holder.spine_unit_vector
            # plot_spine_unit_vector(ax1,raw_good_frame_skeleton_data,raw_mid_hip_XYZ,raw_spine_unit_vector)

            #Plot the heel vector on ax2    
            foot_translated_heel_vector_origin = foot_translated_skeleton_holder.heel_vector_origin
            plot_heel_unit_vector(ax2,foot_translated_heel_unit_vector,foot_translated_heel_vector_origin)

            #Plot the spine and heel vectors on ax3
            y_aligned_heel_unit_vector = y_aligned_skeleton_holder.heel_unit_vector
            y_aligned_heel_vector_origin = y_aligned_skeleton_holder.heel_vector_origin
            y_aligned_mid_hip_XYZ = y_aligned_skeleton_holder.mid_hip_XYZ
            plot_heel_unit_vector(ax3,y_aligned_heel_unit_vector,y_aligned_heel_vector_origin)
            plot_spine_unit_vector(ax3,y_aligned_good_frame_skeleton_data,y_aligned_mid_hip_XYZ,y_aligned_spine_unit_vector)


            #Plot the spine and heel vectors on ax4
            spine_aligned_mid_hip_XYZ = spine_aligned_skeleton_holder.mid_hip_XYZ
            spine_aligned_spine_unit_vector = spine_aligned_skeleton_holder.spine_unit_vector
            spine_aligned_heel_unit_vector = spine_aligned_skeleton_holder.heel_unit_vector
            spine_aligned_heel_vector_origin = spine_aligned_skeleton_holder.heel_vector_origin
            plot_heel_unit_vector(ax4,spine_aligned_heel_unit_vector,spine_aligned_heel_vector_origin)
            plot_spine_unit_vector(ax4,spine_aligned_good_frame_skeleton_data,spine_aligned_mid_hip_XYZ,spine_aligned_spine_unit_vector)

            ax3.legend()
            ax4.legend()

            plt.show()
    
    return spine_aligned_skeleton_data



if __name__ == '__main__':

    from mediapipe_skeleton_builder import mediapipe_indices

    freemocap_data_folder_path = Path(r'D:\freemocap2022\FreeMocap_Data')
    sessionID = 'sesh_2022-05-09_12_20_23'

    freemocap_data_array_folder_path = freemocap_data_folder_path/sessionID/'DataArrays'
    skeleton_data_path = freemocap_data_array_folder_path/'mediaPipeSkel_3d_smoothed.npy' 

    skeleton_data = np.load(skeleton_data_path)
    skeleton_indices = mediapipe_indices


    good_frame = 81 #a random number

    origin_aligned_skeleton_data = align_skeleton_with_origin(skeleton_data,skeleton_indices, good_frame, debug = True)

    f = 2