
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



def plot_optimization_error(error,gaze_xyz, gaze_rotated_by_guess_then_head_rotation_xyz, mean_fixation_point_xyz, skel_during_vor_fr_mar_dim):
        figure_number=13451

        if not plt.fignum_exists(figure_number):
            fig = plt.figure(figure_number)
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = plt.gcf()
            ax = plt.gca()


        fig.suptitle(f'error: {error}')
        ax_range = 1e3
        ax.clear() 

        ax.plot(0,0,0, 'mo',label='origin')

        ax.plot(gaze_xyz[:,0],
                gaze_xyz[:,1], 
                gaze_xyz[:,2],  'k-o',label='original gaze xyz')

        gaze_rot_xyz  =np.array(gaze_rotated_by_guess_then_head_rotation_xyz)
        ax.plot(gaze_rot_xyz[:,0],
                gaze_rot_xyz[:,1], 
                gaze_rot_xyz[:,2],  'r-o',label='gaze_rotated_by_guess_then_head_rotation_xyz')

        ax.plot(mean_fixation_point_xyz[0],
                mean_fixation_point_xyz[1], 
                mean_fixation_point_xyz[2],  'b-o',label='mean_fixation_point_xyz')

        ax.set_xlim([-ax_range, ax_range])
        ax.set_ylim([-ax_range, ax_range])
        ax.set_zlim([-ax_range, ax_range])
        ax.legend()
        plt.pause(.01)


def get_optimal_rotation_matrix_to_align_gaze_with_target(gaze_xyz,
                                                          fixation_point_in_eye_coordinates_xyz,
                                                          head_rotation_matricies_fr_row_col,
                                                          skel_during_vor_fr_mar_dim):
    euler_angles = optimize.least_squares(get_error_between_two_rotation_matricies,
                                    [0,0,0],
                                    args=(gaze_xyz,fixation_point_in_eye_coordinates_xyz, head_rotation_matricies_fr_row_col, skel_during_vor_fr_mar_dim),
                                    gtol=1e-10,
                                    verbose=2).x
    return Rotation.from_euler('XYZ',euler_angles).as_matrix()



def get_error_between_two_rotation_matricies(euler_angle_guess,
                                             gaze_xyz,
                                             fixation_point_in_eye_coordinates_xyz,
                                             head_rotation_maticies_fr_row_col,
                                             skel_during_vor_fr_mar_dim):
    #convert euler angles to rotation matrix
    rotation_matrix_guess = Rotation.from_euler('XYZ',euler_angle_guess).as_matrix()

    #rotate gaze by rotation guess
    gaze_rotated_by_guess = [rotation_matrix_guess @ gaze_xyz[this_frame_number,:] for this_frame_number in range(gaze_xyz.shape[0])]

    #...then rotate THAT by head_rotation matrix
    gaze_rotated_by_guess_then_head_rotation_xyz =[head_rotation_maticies_fr_row_col[this_frame_number,:,:] @ gaze_rotated_by_guess[this_frame_number]
                                                     for this_frame_number in range(gaze_xyz.shape[0])]

    mean_fixation_point_xyz = np.mean(fixation_point_in_eye_coordinates_xyz, axis=0)
    #define error as difference between the fixation point and that rotated gaze estimate (these are both normalized, I think)
    error_per_frame = gaze_rotated_by_guess_then_head_rotation_xyz - mean_fixation_point_xyz
    error = np.nanmean(np.nansum(error_per_frame**2)/error_per_frame.shape[0])
    
    plot_optimization_error(error,gaze_xyz, gaze_rotated_by_guess_then_head_rotation_xyz, mean_fixation_point_xyz, skel_during_vor_fr_mar_dim)

    return error


def get_translation_error_between_two_rotation_matrices(translation_guess,segmentA, segmentB):
    #convert euler angles to rotation matrix
    
    segmentA_translated_by_guess = segmentA + translation_guess

    #mean_segmentB_point = np.mean(segmentB, axis=0)

    #mean_segmentA_point = np.mean(segmentA_translated_by_guess, axis=0)

    error_list = [abs(y-x) for x,y in zip(segmentA_translated_by_guess,segmentB)]

    error = np.mean(error_list)

    return error

def get_optimal_translation_matrix(segmentA, segmentB):
    translation_matrix = optimize.least_squares(get_translation_error_between_two_rotation_matrices,
                                    [0,0,0], args = (segmentA, segmentB),
                                    gtol = 1e-10,
                                    verbose = 2).x
    return translation_matrix
                    
def get_error_between_two_rotation_matrices(euler_angle_guess, segmentA, segmentB):

    rotation_matrix_guess = Rotation.from_euler('XYZ',euler_angle_guess).as_matrix()

    # vectorA = segmentA[1] - segmentA[0]

    # vectorA_rotated_by_guess = rotation_matrix_guess @ vectorA
    

    # segmentA_rotated_by_guess = segmentA[0] + vectorA_rotated_by_guess

    # #----Attempt 1 for rotation
    # segmentA_rotated_by_guess = [rotation_matrix_guess @ x for x in segmentA]

    # error_list = [abs(y-x) for x,y in zip(segmentA_rotated_by_guess,segmentB)]
    # error = np.mean(error_list)
    # #----

    #----Attempt 2 for rotation
    segmentA_rotated_by_guess = [rotation_matrix_guess @ x for x in segmentA]

    vectorA_rotated_by_guess = segmentA_rotated_by_guess[1] - segmentA_rotated_by_guess[0]
    vectorB = segmentB[1] - segmentB[0]

    error = abs(np.cross(vectorA_rotated_by_guess,vectorB))

    error = np.linalg.norm(error)
    # figure = plt.figure()
    # ax1 = figure.add_subplot(projection = '3d')
    # plot_final_rotated_segments(ax1,segmentA, segmentB, rotation_matrix_guess)
    # plt.show()

    # mean_segmentB_point = np.mean(segmentB, axis=0)

    # mean_segmentA_point = np.mean(segmentA_rotated_by_guess, axis=0)

    # error = abs(mean_segmentB_point-mean_segmentA_point)

    return error 

def get_optimal_rotation_matrix(segmentA,segmentB):
    euler_angles = optimize.least_squares(get_error_between_two_rotation_matrices,
                                    [0,0,0], args = (segmentA, segmentB),
                                    gtol = 1e-10,
                                    verbose = 2).x
    return Rotation.from_euler('XYZ',euler_angles).as_matrix()

def get_segment_values(segment):

    segment_x_values = [segment[0][0],segment[1][0]]
    segment_y_values = [segment[0][1],segment[1][1]]
    segment_z_values = [segment[0][2],segment[1][2]]

    return segment_x_values, segment_y_values, segment_z_values
def plot_final_segments(ax,segmentA, segmentB, new_segment):
    #segmentA_translated_by_guess = [x + translation_matrix for x in segmentA]
    
    segmentA_xvalues, segmentA_yvalues, segmentA_zvalues = get_segment_values(segmentA)
    #segmentA_translated_by_guess_xvalues, segmentA_translated_by_guess_yvalues, segmentA_translated_by_guess_zvalues = get_segment_values(segmentA_translated_by_guess)
    segmentB_xvalues, segmentB_yvalues, segmentB_zvalues = get_segment_values(segmentB)
    new_segment_xvalues, new_segment_yvalues, new_segment_zvalues = get_segment_values(new_segment)



    ax.plot(segmentA_xvalues, segmentA_yvalues, segmentA_zvalues, 'b',label='original segmentA', alpha = .5)

    ax.plot(segmentB_xvalues, segmentB_yvalues, segmentB_zvalues, 'r',label='original segmentB', alpha = .5)

    #ax.plot(segmentA_translated_by_guess_xvalues, segmentA_translated_by_guess_yvalues, segmentA_translated_by_guess_zvalues, 'g',label='segmentA_translated_by_guess', alpha = .5, linestyle = 'dashed')
    ax.plot(new_segment_xvalues, new_segment_yvalues, new_segment_zvalues, 'g',label='new_segment', alpha = .5)
    f =2 

def plot_final_rotated_segments(ax,segmentA, segmentB, rotation_matrix):
    segmentA_rotated_by_guess = [rotation_matrix @ x for x in segmentA]

    segmentA_xvalues, segmentA_yvalues, segmentA_zvalues = get_segment_values(segmentA)
    segmentA_rotated_by_guess_xvalues, segmentA_rotated_by_guess_yvalues, segmentA_rotated_by_guess_zvalues = get_segment_values(segmentA_rotated_by_guess)
    segmentB_xvalues, segmentB_yvalues, segmentB_zvalues = get_segment_values(segmentB)

    ax.plot(segmentA_xvalues, segmentA_yvalues, segmentA_zvalues, 'b',label='original segmentA', alpha = .5)
    ax.plot(segmentB_xvalues, segmentB_yvalues, segmentB_zvalues, 'r',label='original segmentB', alpha = .5)
    ax.plot(segmentA_rotated_by_guess_xvalues, segmentA_rotated_by_guess_yvalues, segmentA_rotated_by_guess_zvalues, 'g-o',label='segmentA_rotated_by_guess', alpha = .5, linestyle = 'dashed')


def translate_skeleton_frame(skeleton_data_frame, translation_distance):
    """Take in a frame of rotated skeleton data, and apply the translation distance to each point in the skeleton"""

    translated_point = [x + y for x,y in zip(skeleton_data_frame, translation_distance)]
    return translated_point


# #----TRANSLATION
# # pointA1 = np.array([5,13,4])
# # pointA2 = np.array([10,18,9])

# # pointB1 = np.array([8,9,7])
# # pointB2 = np.array([13,14,12])

# pointA1 = np.array([12,4,3])
# pointA2 = np.array([6,1,9])

# pointB1 = np.array([5,13,4])
# pointB2 = np.array([10,12,12])

# segmentA = [pointA1, pointA2]

# segmentB = [pointB1, pointB2]

# translation_matrix = get_optimal_translation_matrix(segmentA, segmentB)

# figure = plt.figure()
# ax1 = figure.add_subplot(projection = '3d')
# plot_final_segments(ax1,segmentA, segmentB, translation_matrix)
# plt.show()
# #-------



# pointA1 = np.array([12,4,3])
# pointA2 = np.array([6,1,9])

# pointB1 = np.array([5,13,4])
# pointB2 = np.array([10,12,12])


pointA1 = np.array([4,12,0])
pointA2 = np.array([10,7,2])

pointB1 = np.array([15,3,24])
pointB2 = np.array([19,2,12])

# pointA1 = np.array([5,13,4])
# pointA2 = np.array([10,18,9])

# pointB1 = np.array([8,9,7])
# pointB2 = np.array([13,14,12])

pointA1 = np.array([-311.95498264,  107.9060753 , 1308.04556225])
pointA2 = np.array([-358.87439602,  104.9854728 , 1035.59036417])

pointB1 = np.array([ -70.52998081,   22.85845106, 1319.20843919])
pointB2 = np.array([ -65.79256761,   41.31004819, 1093.13235156])
segmentA = [pointA1, pointA2]

segmentB = [pointB1, pointB2]

# segmentA = np.vstack((pointA1,pointA2))

# csegmentB = np.vstack((pointB1,pointB2))

debug = True


figure = plt.figure()
ax1 = figure.add_subplot(projection = '3d')

rotation_matrix  = get_optimal_rotation_matrix(segmentA,segmentB)

segmentA_rotated_by_guess = [rotation_matrix @ x for x in segmentA]

plot_final_rotated_segments(ax1,segmentA, segmentB, rotation_matrix)
ax1.legend()
if debug:
    plt.show()



translation_matrix = get_optimal_translation_matrix(segmentA_rotated_by_guess, segmentB)

translation_distance = segmentB[0] - segmentA_rotated_by_guess[0]

#segment_A_translated = [x + y for x,y in zip(segmentA_rotated_by_guess[0], translation_distance)]
segmentA_translated = []

for x in range(len(segmentA)):
    point_translated = translate_skeleton_frame(segmentA_rotated_by_guess[x], translation_distance)
    segmentA_translated.append(point_translated)

figure = plt.figure()
ax1 = figure.add_subplot(projection = '3d')
plot_final_segments(ax1,segmentA_rotated_by_guess, segmentB, segmentA_translated)


if debug:
    plt.show()
f = 2

bot_row_transformation_matrix = np.array([0,0,0,1])

translation_matrix_transposed = np.array([[x] for x in translation_matrix])

rotation_and_translation_stacked = np.hstack((rotation_matrix,translation_matrix_transposed))

transformation_matrix = np.vstack((rotation_and_translation_stacked,bot_row_transformation_matrix))

segmentA_one = np.hstack((segmentA[0],np.array([1])))

segmentA_two = np.hstack((segmentA[1],np.array([1])))

segmentA_one_T = transformation_matrix @ segmentA_one
segmentA_two_T = transformation_matrix @ segmentA_two

segmentA_t = [segmentA_one_T[0:3],segmentA_two_T[0:3]]

figure = plt.figure()
ax1 = figure.add_subplot(projection = '3d')
#plot_final_segments(ax1,segmentA_t, segmentB, [0,0,0])

#ax1.legend()

if debug:
    plt.show()

f = 2