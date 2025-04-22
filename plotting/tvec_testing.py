# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D

# def rot_z(theta):
#     """Rotation about Z by theta radians."""
#     return np.array([
#         [np.cos(theta), -np.sin(theta), 0],
#         [np.sin(theta),  np.cos(theta), 0],
#         [0,              0,             1]
#     ])

# # def demo_shift_origin_to_cam0():
# #     # --- Step 0: Define extrinsic params for two cameras ---
# #     theta0, theta1 = np.deg2rad(30), np.deg2rad(-45)
# #     R0, R1 = rot_z(theta0), rot_z(theta1)
# #     t0, t1 = np.array([1.0, 2.0, 0.0]), np.array([4.0, 1.0, 0.0])
# #     Rs = [R0, R1]
# #     ts = [t0, t1]

# #     # --- Step 1: Compute true camera centers in world coords ---
# #     Cs_orig = [(-R.T @ t) for R, t in zip(Rs, ts)]

# #     # --- Step 2: Naive recenter: subtract t0 directly on t_i ---
# #     ts_naive   = [t - t0 for t in ts]
# #     Cs_naive   = [(-R.T @ tn) for R, tn in zip(Rs, ts_naive)]

# #     # --- Step 3: Proper recenter: shift origin in world, then express in each camera's axes ---
# #     delta_w    = -R0.T @ t0             # = world→cam0-origin shift in world axes
# #     ts_proper  = [t + (R @ delta_w) for R, t in zip(Rs, ts)]
# #     Cs_proper  = [(-R.T @ tp) for R, tp in zip(Rs, ts_proper)]

# #     # # --- Step 4: Table of results ---
# #     # df = pd.DataFrame({
# #     #     'camera':      ['cam0', 'cam1'],
# #     #     'C_orig':      [list(np.round(c,3)) for c in Cs_orig],
# #     #     'C_naive':     [list(np.round(c,3)) for c in Cs_naive],
# #     #     'C_proper':    [list(np.round(c,3)) for c in Cs_proper],
# #     # })
# #     # print(df)

# #     # --- Step 5: Plot in world space ---
# #     fig = plt.figure(figsize=(6,5))
# #     ax = fig.add_subplot(111, projection='3d')
# #     cam_colors = ['blue', 'orange']

# #     # Original centers
# #     for c, col in zip(Cs_orig, cam_colors):
# #         ax.scatter(*c, color=col, marker='o', s=80, alpha=1.0)
# #     # Naive centers
# #     for c, col in zip(Cs_naive, cam_colors):
# #         ax.scatter(*c, color=col, marker='s', s=60, alpha=0.6)
# #     # Proper centers
# #     for c, col in zip(Cs_proper, cam_colors):
# #         ax.scatter(*c, color=col, marker='^', s=60, alpha=0.6)

# #     # Custom legend
# #     legend_elems = [
# #         Line2D([0],[0], marker='o', color='k', linestyle='', label='Original', markersize=8),
# #         Line2D([0],[0], marker='s', color='k', linestyle='', label='Naive', markersize=8, alpha=0.6),
# #         Line2D([0],[0], marker='^', color='k', linestyle='', label='Proper', markersize=8, alpha=0.6),
# #         Line2D([0],[0], marker='o', color='blue', linestyle='', label='cam0', markersize=8),
# #         Line2D([0],[0], marker='o', color='orange', linestyle='', label='cam1', markersize=8),
# #     ]
# #     ax.legend(handles=legend_elems, bbox_to_anchor=(1.05,1), loc='upper left')
# #     ax.set_title("Camera Centers in World Space\nOriginal vs Naive vs Proper")
# #     ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
# #     plt.tight_layout()
# #     plt.show()

# # # Run the demo
# # demo_shift_origin_to_cam0()

# def demo_shift_origin_to_cam0():
#     # --- Step 0: Define extrinsic params for two cameras ---
#     theta0, theta1 = np.deg2rad(30), np.deg2rad(-45)
#     R0, R1 = rot_z(theta0), rot_z(theta1)
#     t0, t1 = np.array([1.0, 2.0, 0.0]), np.array([4.0, 1.0, 0.0])
#     Rs = [R0, R1]
#     ts = [t0, t1]

#     # --- Step 1: Compute true camera centers in world coords ---
#     Cs_orig = [t0, t1]

#     # --- Step 2: Naive recenter: subtract t0 directly on t_i ---
#     ts_naive   = [t - t0 for t in ts]
#     Cs_naive   = [ts_naive[0], ts_naive[1]]


#     Cs_world = [(-R.T @ t) for R, t in zip(Rs, ts)]
#     Cs_naive_world = [(-R.T @ t) for R, t in zip(Rs, ts_naive)]


#     # --- Step 3: Proper recenter: shift origin in world, then express in each camera's axes ---
#     delta_w    = -R0.T @ t0             # = world→cam0-origin shift in world axes
#     ts_proper  = [t + (R @ delta_w) for R, t in zip(Rs, ts)]
#     Cs_proper  = [(-R.T @ tp) for R, tp in zip(Rs, ts_proper)]

#     # # --- Step 4: Table of results ---
#     # df = pd.DataFrame({
#     #     'camera':      ['cam0', 'cam1'],
#     #     'C_orig':      [list(np.round(c,3)) for c in Cs_orig],
#     #     'C_naive':     [list(np.round(c,3)) for c in Cs_naive],
#     #     'C_proper':    [list(np.round(c,3)) for c in Cs_proper],
#     # })
#     # print(df)

#     # --- Step 5: Plot in world space ---
#     fig = plt.figure(figsize=(6,5))
#     ax = fig.add_subplot(111, projection='3d')
#     cam_colors = ['blue', 'orange']

#     # Original centers
#     for c, col in zip(Cs_orig, cam_colors):
#         ax.scatter(*c, color=col, marker='o', s=80, alpha=1.0)
#     # Naive centers
#     for c, col in zip(Cs_naive, cam_colors):
#         ax.scatter(*c, color=col, marker='s', s=60, alpha=0.6)
#     # # Proper centers
#     # for c, col in zip(Cs_proper, cam_colors):
#     #     ax.scatter(*c, color=col, marker='^', s=60, alpha=0.6)
#     for c, col in zip(Cs_world, cam_colors):
#         ax.scatter(*c, color=col, marker='x', s=60, alpha=0.6)
#     for c, col in zip(Cs_naive_world, cam_colors):
#         ax.scatter(*c, color=col, marker='^', s=60, alpha=0.6)

#     # Custom legend
#     legend_elems = [
#         Line2D([0],[0], marker='o', color='k', linestyle='', label='Original', markersize=8),
#         Line2D([0],[0], marker='s', color='k', linestyle='', label='Naive', markersize=8, alpha=0.6),
#         Line2D([0],[0], marker='^', color='k', linestyle='', label='Proper', markersize=8, alpha=0.6),
#         Line2D([0],[0], marker='o', color='blue', linestyle='', label='cam0', markersize=8),
#         Line2D([0],[0], marker='o', color='orange', linestyle='', label='cam1', markersize=8),
#     ]
#     ax.legend(handles=legend_elems, bbox_to_anchor=(1.05,1), loc='upper left')
#     ax.set_title("Camera Centers in World Space\nOriginal vs Naive vs Proper")
#     ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
#     plt.tight_layout()
#     plt.show()

# # Run the demo
# demo_shift_origin_to_cam0()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cv2

## ------- Make extrinsic parameters for two cameras -------- ##

def create_plot(steps):
    cam_colors = ['blue', 'orange', 'green']
    
    # Create just one figure outside the loop
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*[0,0,0], color='k', marker='o')
    
    # Plot all steps on the same figure
    for i, (label, pts, marker, border) in enumerate(steps):
        for cam_idx, p in enumerate(pts):
            ax.scatter(*p, color=cam_colors[cam_idx],
                      marker=marker, s=80,
                      alpha=1.0 if i == 0 else 0.6,
                      label=f'{label}_{cam_idx}',
                      edgecolors='black' if border else None,
                      linewidths=3 if border else 0)
    # Set title and show plot only once after all steps are plotted
    ax.set_title(f'Step {i+1}: {label}')
    ax.legend()
    plt.show()

steps = []

rvec_0 = np.array([ 0.040760069318579034, 0.017580512046857167, 0.0701610289420285,])
R0,_ = cv2.Rodrigues(rvec_0)
tvec_0 = np.array([ -35.75397305454985, 118.52574646385185, -300.31247883673956,])

rvec_1 = np.array([ 0.0988147420656383, 1.2324020613174713, 0.6356256491944287,])
R1, _ = cv2.Rodrigues(rvec_1)
tvec_1 = np.array([ -2555.1777089626594, -525.511267670988, 441.57434690970086,])

rvec_2 = np.array([0.1156237913096535, 0.5902861320315332, 0.5150858109942985,])
R2,_ = cv2.Rodrigues(rvec_2)
tvec_2 = np.array([ -1444.4499534004683, -234.75249030410643, -436.471133983029,])

original_tvecs = [tvec_0, tvec_1, tvec_2]
rmatrices = [R0, R1, R2]
steps.append(('cam_space', original_tvecs, 'o', False))




# create_plot(steps)


#Pc = RPw + t [t is translation shift from world origin to camera origin in camera space]
#Pc = 0 = RPw + t
#t = -RPw = -RC (where C is the coordinates of the camera in world space)
# C = -R.T*t
def get_camera_loc_in_world_space(rmatrix,tvec):
    return -rmatrix.T@tvec

original_cs = [get_camera_loc_in_world_space(rmatrix,tvec) for rmatrix, tvec in zip(rmatrices,original_tvecs)]
steps.append(('world_space', original_cs, 's', False))
# create_plot(steps)


##shifting translation with just tvecs
tvecs_shifted = [tvec-original_tvecs[0] for tvec in original_tvecs]
steps.append(('shifted_tvecs', tvecs_shifted, 'o', True))
# create_plot(steps,)

shifted_cs = [get_camera_loc_in_world_space(rmatrix,tvec) for rmatrix, tvec in zip(rmatrices,tvecs_shifted)]
steps.append(('shifted_world_space', shifted_cs, 's', True))
# create_plot(steps)


remapped_cs = [c - original_cs[0] for c in original_cs]
steps.append(('remapped_cs', remapped_cs, 'x', True))
# create_plot(steps)


steps = []
steps.append(('cam_space', original_tvecs, 'o', False))
steps.append(('world_space', original_cs, 's', False))

##now shifting by transforming into world space first
cam_0_world_space_pos = get_camera_loc_in_world_space(rmatrix=R0, tvec=tvec_0) 

transformed_tvecs = [tvec + rmatrix@cam_0_world_space_pos for tvec, rmatrix in zip(original_tvecs, rmatrices)]
steps.append(('transformed_tvecs', transformed_tvecs, 'o', True))

transformed_cs = [get_camera_loc_in_world_space(rmatrix,tvec) for rmatrix, tvec in zip(rmatrices,transformed_tvecs)]
steps.append(('transformed_world_space', transformed_cs, 's', True))

steps.append(('shifted_world_space', shifted_cs, 'x', True))



import numpy as np
from itertools import combinations

def compute_pairwise_distances(name, points):
    print(f"\n{name} pairwise distances:")
    for i, j in combinations(range(len(points)), 2):
        dist = np.linalg.norm(points[i] - points[j])
        print(f"Distance between cam {i} and cam {j}: {dist:.4f}")

# Ensure your inputs are NumPy arrays
original_cs = np.array(original_cs)
shifted_cs = np.array(shifted_cs)
transformed_cs = np.array(transformed_cs)

compute_pairwise_distances("Original", original_cs)
compute_pairwise_distances("Improperly shifted (tvec-based)", shifted_cs)
compute_pairwise_distances("Properly shifted (world-space)", transformed_cs)


# remapped_cs = [c - original_cs[0] for c in original_cs]
# steps.append(('remapped_cs', remapped_cs, 'x', True))

create_plot(steps)

f = 2
# def rot_z(theta):
#     """Rotation about Z by theta radians."""
#     return np.array([
#         [np.cos(theta), -np.sin(theta), 0],
#         [np.sin(theta),  np.cos(theta), 0],
#         [0,              0,             1]
#     ])

# theta0, theta1 = np.deg2rad(30), np.deg2rad(-45)
# R0, R1 = rot_z(theta0), rot_z(theta1)
# t0, t1 = np.array([1.0, 2.0, 0.0]), np.array([4.0, 1.0, 0.0])
# rvecs = [R0, R1]
# tvecs = [t0, t1]

# steps = []
## ------- Plot positions in camera coordinates  -------- ##
# fig = plt.figure(figsize=(6,5))
# ax = fig.add_subplot(111, projection='3d')
# cam_colors = ['blue', 'orange']
# ax.scatter(*[0,0,0], color = 'k', marker = 'o')
# steps.append(('original tvec', tvecs, 'o'))

# create_plot(steps)

## ------- Shift positions based on tvecs  -------- ##

# tvecs_shifted = [t - t0 for t in tvecs]
# steps.append(('shifted tvec', tvecs_shifted, 's'))
# create_plot(steps)


# for i, (c, col) in enumerate(zip(tvecs_shifted, cam_colors)):
#     ax.scatter(*c, color=col, marker='s', s=80, alpha=1.0, label=f'Shifted tvec {i}')

# ax.legend()
# plt.show()
