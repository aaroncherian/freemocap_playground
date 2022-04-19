import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path 
import socket
import pickle
import scipy.io as sio

this_computer_name = socket.gethostname()
print(this_computer_name)

if this_computer_name == 'DESKTOP-V3D343U':
    freemocap_data_path = Path(r"I:\My Drive\HuMoN_Research_Lab\FreeMoCap_Stuff\FreeMoCap_Balance_Validation\data")
elif this_computer_name == 'DESKTOP-F5LCT4Q':
    freemocap_data_path = Path(r"C:\Users\aaron\Documents\HumonLab\Spring2022\ValidationStudy\FreeMocap_Data")
else:
    freemocap_data_path = Path(r"C:\Users\kiley\Documents\HumonLab\SampleFMC_Data\FreeMocap_Data-20220216T173514Z-001\FreeMocap_Data")

sessionID = 'session_SER_1_20_22' #name of the sessionID folder
data_array_name = 'mediaPipeSkel_3d_smoothed.npy'

this_freemocap_session_path = freemocap_data_path / sessionID
this_freemocap_data_path = this_freemocap_session_path/'DataArrays'
mediapipe_data_path = this_freemocap_data_path/data_array_name

mediapipeSkel_fr_mar_dim = np.load(mediapipe_data_path)

num_pose_joints = 33 #number of pose joints tracked by mediapipe 


pose_joint_range = range(num_pose_joints)

mediapipe_pose_data = mediapipeSkel_fr_mar_dim[:,0:num_pose_joints,:] #load just the pose joints into a data array, removing hands and face data 

num_frames = len(mediapipe_pose_data)

num_frame_range = range(num_frames)

skel_x = mediapipe_pose_data[:,:,0]
skel_y = mediapipe_pose_data[:,:,1]
skel_z = mediapipe_pose_data[:,:,2]

segments = [
'head',
'trunk',
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
'left_foot'
]

joint_connections = [
['left_ear','right_ear'],
['mid_chest_marker', 'mid_hip_marker'], #do these joint_connections have to correspond to mediapipe? yes
['right_shoulder','right_elbow'],
['left_shoulder','left_elbow'],
['right_elbow', 'right_wrist'],
['left_elbow', 'left_wrist'],
['right_wrist', 'right_hand_marker'], #need to spend some time on the hands,can there be more than two elements in these lists?
['left_wrist', 'left_hand_marker'],
['right_hip', 'right_knee'],
['left_hip', 'left_knee'],
['right_knee', 'right_ankle'],
['left_knee', 'left_ankle'],
['right_back_of_foot_marker', 'right_foot_index'], #will need to figure out these naming conventions later
['left_back_of_foot_marker', 'left_foot_index']
]

segment_COM_lengths = [
.5,
.5,
.436,
.436,
.430,
.430,
.506, #check on the hand, did you actually use this in the com approx? Yes I did
.506, #check on the hand, did you actually use this in the com approx? Yes I did
.433,
.433,
.433,
.433,
.5, #check on the foot, did you actually use this in the com approx? No, I didn't
.5  #check on the foot, did you actually use this in the com approx? No, I didn't
]

segment_COM_percentages = [
.081,
.497,
.028,
.028,
.016,
.016,
.006,
.006,
.1,
.1,
.0465,
.0465,
.0145,
.0145
]

num_segments = len(segments)

df = pd.DataFrame(list(zip(segments,joint_connections,segment_COM_lengths,segment_COM_percentages)),columns = ['Segment Name','Joint Connection','Segment COM Length','Segment COM Percentage'])
segment_conn_len_perc_dataframe = df.set_index('Segment Name')

def build_mediapipe_skeleton(mediapipe_pose_data,segment_dataframe):
    """ This function takes in the mediapipe pose data array and the segment_conn_len_perc_dataframe. 
        For each frame of data, it loops through each segment we want to find and identifies the names
        of the proximal and distal joints of that segment. Then it searches the mediapipe_indices list
        to find the index that corresponds to the name of that segment. We plug the index into the 
        mediapipe_pose_data array to find the proximal/distal joints' XYZ coordinates at that frame. 
        The segment, its proximal joint and its distal joint gets thrown into a dictionary. 
        And then that dictionary gets saved to a list for each frame. By the end of the function, you 
        have a list that contains the skeleton segment XYZ coordinates for each frame."""

    def build_mediapipe_virtual_markers(mediapipe_indices,frame):
        def mediapipe_index_finder(list_of_joint_names):

            indices = []
            for name in list_of_joint_names:
                this_name_index = mediapipe_indices.index(name)
                indices.append(this_name_index)
            
            return indices
        
        def mediapipe_XYZ_finder(indices_list):

                XYZ_coordinates = []
                for index in indices_list:
                    this_joint_coordinate = mediapipe_pose_data[frame,index,:]
                    XYZ_coordinates.append(this_joint_coordinate)

                return XYZ_coordinates

            
        def build_virtual_trunk_marker(trunk_joint_connection,mediapipe_indices):
                trunk_indices = mediapipe_index_finder(trunk_joint_connection)

                trunk_XYZ_coordinates = mediapipe_XYZ_finder(trunk_indices)

                trunk_proximal = (trunk_XYZ_coordinates[0] + trunk_XYZ_coordinates[1])/2
                trunk_distal = (trunk_XYZ_coordinates[2] + trunk_XYZ_coordinates[3])/2

                return trunk_proximal, trunk_distal
        
        def build_virtual_foot_marker(foot_joint_connection, mediapipe_indices):
                foot_indices = mediapipe_index_finder(foot_joint_connection)

                foot_XYZ_coordinates = mediapipe_XYZ_finder(foot_indices)

                foot_virtual_marker = (foot_XYZ_coordinates[0] + foot_XYZ_coordinates[1])/2

                return foot_virtual_marker

        trunk_joint_connection = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']

        left_foot_joint_connection = ['left_heel','left_ankle']
        right_foot_joint_connection = ['right_heel','right_ankle']

        trunk_virtual_markers = build_virtual_trunk_marker(trunk_joint_connection,mediapipe_indices)
        left_foot_virtual_marker = build_virtual_foot_marker(left_foot_joint_connection,mediapipe_indices)
        right_foot_virtual_marker = build_virtual_foot_marker(right_foot_joint_connection,mediapipe_indices)
    
        return trunk_virtual_markers, left_foot_virtual_marker, right_foot_virtual_marker

    #this is a list of the 33 pose joints that mediapipe tracks
    mediapipe_indices = [
    'nose',
    'left_eye_inner',
    'left_eye',
    'left_eye_outer',
    'right_eye_inner',
    'right_eye',
    'right_eye_outer',
    'left_ear',
    'right_ear',
    'mouth_left',
    'mouth_right',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_pinky',
    'right_pinky',
    'left_index',
    'right_index',
    'left_thumb',
    'right_thumb',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle',
    'left_heel',
    'right_heel',
    'left_foot_index',
    'right_foot_index'
    ]
    

    mediapipe_frame_segment_joint_XYZ = [] #empty list to hold all the skeleton XYZ coordinates/frame
    for frame in num_frame_range: #NOTE: need to change frame_range to numFrames 2/09/2022 - AC

        trunk_virtual_markers, left_foot_virtual_marker, right_foot_virtual_marker = build_mediapipe_virtual_markers(mediapipe_indices,frame)

        mediapipe_pose_skeleton_coordinates = {}
        for segment,segment_info in segment_conn_len_perc_dataframe.iterrows(): #iterate through the data frame by the segment name and all the info for that segment
            if segment == 'trunk':

                #based on index, excract coordinate data from fmc mediapipe data
                mediapipe_pose_skeleton_coordinates[segment] = [trunk_virtual_markers[0], 
                                                                trunk_virtual_markers[1]
                                                                ]
            elif segment == 'left_hand' or segment == 'right_hand':
                proximal_joint_hand = segment_info['Joint Connection'][0]
                if segment == 'left_hand':
                    distal_joint_hand = 'left_index'
                else:
                    distal_joint_hand = 'right_index'

                proximal_joint_hand_index = mediapipe_indices.index(proximal_joint_hand)
                distal_joint_hand_index = mediapipe_indices.index(distal_joint_hand)

                mediapipe_pose_skeleton_coordinates[segment] = [mediapipe_pose_data[frame,proximal_joint_hand_index, :],
                                                                mediapipe_pose_data[frame,distal_joint_hand_index,:]]

            elif segment == 'left_foot' or segment == 'right_foot':
                if segment == 'left_foot':
                    proximal_joint_foot_name = 'left_ankle'
                else:
                    proximal_joint_foot_name = 'right_ankle'
                
                proximal_joint_foot_index = mediapipe_indices.index(proximal_joint_foot_name)

                distal_joint_foot = segment_info['Joint Connection'][1]
                distal_joint_foot_index = mediapipe_indices.index(distal_joint_foot)
                mediapipe_pose_skeleton_coordinates[segment] = [mediapipe_pose_data[frame,proximal_joint_foot_index, :],
                                                                mediapipe_pose_data[frame, distal_joint_foot_index,:]]            

            else:
                proximal_joint_name = segment_info['Joint Connection'][0] 
                distal_joint_name = segment_info['Joint Connection'][1]

            #get the mediapipe index for the proximal and distal joint for this segment
                proximal_joint_index = mediapipe_indices.index(proximal_joint_name)
                distal_joint_index = mediapipe_indices.index(distal_joint_name)

            #use the mediapipe indices to get the XYZ coordinates for the prox/distal joints and throw it in a dictionary
            #mediapipe_pose_skeleton_coordinates[segment] = {'proximal':mediapipe_pose_data[frame,proximal_joint_index,:],'distal':mediapipe_pose_data[frame,distal_joint_index,:]}
                mediapipe_pose_skeleton_coordinates[segment] = [mediapipe_pose_data[frame,proximal_joint_index,:],mediapipe_pose_data[frame,distal_joint_index,:]]
                
        mediapipe_frame_segment_joint_XYZ.append(mediapipe_pose_skeleton_coordinates)
        f = 2 
    
    return mediapipe_frame_segment_joint_XYZ

    f= 2
skelcoordinates_frame_segment_joint_XYZ = build_mediapipe_skeleton(mediapipe_pose_data,segment_conn_len_perc_dataframe)

def calculate_segment_COM(segment_conn_len_perc_dataframe,skelcoordinates_frame_segment_joint_XYZ, num_frame_range):
    segment_COM_frame_dict = []
    for frame in num_frame_range:
        segment_COM_dict = {}
        for segment,segment_info in segment_conn_len_perc_dataframe.iterrows():
            this_segment_XYZ = skelcoordinates_frame_segment_joint_XYZ[frame][segment]

            #for mediapipe
            this_segment_proximal = this_segment_XYZ[0]
            this_segment_distal = this_segment_XYZ[1]
            this_segment_COM_length = segment_info['Segment COM Length']

            this_segment_COM = this_segment_proximal + this_segment_COM_length*(this_segment_distal-this_segment_proximal)
            segment_COM_dict[segment] = this_segment_COM
        segment_COM_frame_dict.append(segment_COM_dict)
    return segment_COM_frame_dict
    f = 2


def reformat_segment_COM(segment_COM_frame_dict, num_frame_range):
    
    segment_COM_frame_imgPoint_XYZ = np.empty([int(len(num_frame_range)),int(num_segments),3])
    for frame in num_frame_range:
        this_frame_skeleton = segment_COM_frame_dict[frame]
        for joint_count,segment in enumerate(this_frame_skeleton.keys()):
            segment_COM_frame_imgPoint_XYZ[frame,joint_count,:] = this_frame_skeleton[segment]
    return segment_COM_frame_imgPoint_XYZ


segment_COM_frame_dict = calculate_segment_COM(segment_conn_len_perc_dataframe, skelcoordinates_frame_segment_joint_XYZ, num_frame_range)
segment_COM_frame_imgPoint_XYZ = reformat_segment_COM(segment_COM_frame_dict,num_frame_range)


def calculate_total_body_COM(segment_conn_len_perc_dataframe,segment_COM_frame_dict,num_frame_range):
    totalBodyCOM_frame_XYZ = np.empty([int(len(num_frame_range)),3])

    for frame in num_frame_range:

        this_frame_total_body_percentages = []
        this_frame_skeleton = segment_COM_frame_dict[frame]

        for segment, segment_info in segment_conn_len_perc_dataframe.iterrows():

            this_segment_COM = this_frame_skeleton[segment]
            this_segment_COM_percentage = segment_info['Segment COM Percentage']

            this_segment_total_body_percentage = this_segment_COM * this_segment_COM_percentage
            this_frame_total_body_percentages.append(this_segment_total_body_percentage)

        this_frame_total_body_COM = np.nansum(this_frame_total_body_percentages,axis = 0)
       
        totalBodyCOM_frame_XYZ[frame,:] = this_frame_total_body_COM

    f=2
    return totalBodyCOM_frame_XYZ        
    
totalBodyCOM_frame_XYZ = calculate_total_body_COM(segment_conn_len_perc_dataframe,segment_COM_frame_dict,num_frame_range)

f = 2