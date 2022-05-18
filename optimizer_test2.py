from distutils.log import debug, error
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

def get_optimal_rotation_matrix(segmentA,segmentB):
    transformation_matrix = optimize.least_squares(get_error_between_two_transformation_matrices,
                                    [0,0,0,0,0,0], args = (segmentA, segmentB),
                                    gtol = 1e-10,
                                    verbose = 2).x
    return transformation_matrix

# def get_error_between_two_transformation_matrices(transformation_matrix_guess, segmentA, segmentB):

#     rotation_matrix = transformation_matrix_guess[0:3]
#     translation_vector_guess = transformation_matrix_guess[4:6]

#     translation_vector_guess_transposed = np.array([[x] for x in translation_vector_guess])

#     rotation_matrix_guess = Rotation.from_euler('XYZ',rotation_matrix).as_matrix()

#     transformation_bottom_row = np.array([0,0,0,1])

#     translation_and_rotation_matrix_guess = np.hstack((rotation_matrix_guess, translation_vector_guess_transposed))

#     transformation_matrix_guess = np.vstack((translation_and_rotation_matrix_guess, transformation_bottom_row))

#     segmentA_transformed = [np.dot(transformation_matrix_guess, x.T).T for x in segmentA]

def get_error_between_two_transformation_matrices(transformation_matrix_guess, segmentA, segmentB):

    rotation_matrix = transformation_matrix_guess[0:3]

    translation_vector_guess = transformation_matrix_guess[3:6]

    rotation_matrix_guess = Rotation.from_euler('XYZ',rotation_matrix).as_matrix()

    segmentA_rotated_by_guess = [rotation_matrix_guess @ x for x in segmentA]

    vectorA_rotated_by_guess = segmentA_rotated_by_guess[1] - segmentA_rotated_by_guess[0]
    vectorB = segmentB[1] - segmentB[0]

    rotation_error = abs(np.cross(vectorA_rotated_by_guess,vectorB))

    #segmentA_translated_by_guess = segmentA_rotated_by_guess + translation_vector_guess
    #translation_error_list = [abs(y-x) for x,y in zip(segmentA_translated_by_guess,segmentB)]
    translation_error_list = [abs(y-x) for x,y in zip(vectorA_rotated_by_guess,segmentB)]

    #translation_error = np.mean(translation_error_list)

    total_error = np.mean(translation_error_list)
    #print(total_error)

    #figure = plt.figure()
    #ax1 = figure.add_subplot(projection = '3d') 

    return total_error

def get_segment_values(segment):

    segment_x_values = [segment[0][0],segment[1][0]]
    segment_y_values = [segment[0][1],segment[1][1]]
    segment_z_values = [segment[0][2],segment[1][2]]

    return segment_x_values, segment_y_values, segment_z_values

def plot_final_segments(ax,segmentA,segmentB, transformed_segmentA):

    segment_Ax, segment_Ay, segment_Az = get_segment_values(segmentA)
    segment_Bx, segment_By, segment_Bz = get_segment_values(segmentB)
    transformed_segmentAx, transformed_segmentAy, transformed_segmentAz = get_segment_values(transformed_segmentA)
    
    ax.plot(segment_Ax, segment_Ay, segment_Az, 'b', label = 'Segment A')
    ax.plot(segment_Bx, segment_By, segment_Bz, 'r', label = 'Segment B')
    ax.plot(transformed_segmentAx, transformed_segmentAy, transformed_segmentAz, 'g', label = 'Transformed Segment A')

def build_full_transformation_matrix(transformation_matrix):

    rotation_matrix_euler = transformation_matrix[0:3]

    rotation_matrix = Rotation.from_euler('XYZ',rotation_matrix_euler).as_matrix()

    translation_vector_guess = transformation_matrix[3:6]

    translation_vector_transposed = np.array([[x] for x in translation_vector_guess])

    rotation_and_translation_stacked = np.hstack((rotation_matrix,translation_vector_transposed))

    full_transformation_matrix = np.vstack((rotation_and_translation_stacked, np.array([0,0,0,1])))

    return full_transformation_matrix

def add_one_to_matrix(matrix):
    return np.hstack((matrix, np.array([1])))

pointA1 = np.array([12,4,3])
pointA2 = np.array([6,1,9])

pointB1 = np.array([5,13,4])
pointB2 = np.array([10,12,12])

segmentA = [pointA1, pointA2]

segmentB = [pointB1, pointB2]

transformation_matrix = get_optimal_rotation_matrix(segmentA, segmentB)



figure = plt.figure()
ax1 = figure.add_subplot(projection = '3d')

full_transformation_matrix = build_full_transformation_matrix(transformation_matrix)
segmentA_1 = add_one_to_matrix(segmentA[0])
segmentA_2 = add_one_to_matrix(segmentA[1])

new_segmentA = [segmentA_1, segmentA_2]

segmentA_T = [full_transformation_matrix @ x for x in new_segmentA]

plot_final_segments(ax1,segmentA, segmentB, segmentA_T)

ax1.legend()

plt.show()


f = 2