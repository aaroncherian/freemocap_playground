from rich.progress import track 


qualisys_indices = [
'head',
'left_ear',
'right_ear',
'cspine',
'left_shoulder',
'right_shoulder',
'left_elbow',
'right_elbow',
'left_wrist',
'right_wrist',
'left_index',
'right_index',
'left_hip',
'right_hip',
'left_knee',
'right_knee',
'left_ankle',
'right_ankle',
'left_heel',
'right_heel',
'left_foot_index',
'right_foot_index',
]


def build_qualisys_skeleton(mediapipe_pose_data,segment_dataframe, qualisys_indices, num_frame_range):
    """ This function takes in the mediapipe pose data array and the segment_conn_len_perc_dataframe. 
        For each frame of data, it loops through each segment we want to find and identifies the names
        of the proximal and distal joints of that segment. Then it searches the qualisys_indices list
        to find the index that corresponds to the name of that segment. We plug the index into the 
        mediapipe_pose_data array to find the proximal/distal joints' XYZ coordinates at that frame. 
        The segment, its proximal joint and its distal joint gets thrown into a dictionary. 
        And then that dictionary gets saved to a list for each frame. By the end of the function, you 
        have a list that contains the skeleton segment XYZ coordinates for each frame."""

    def build_mediapipe_virtual_markers(qualisys_indices,frame):
        def mediapipe_index_finder(list_of_joint_names):

            indices = []
            for name in list_of_joint_names:
                this_name_index = qualisys_indices.index(name)
                indices.append(this_name_index)
            
            return indices
        
        def mediapipe_XYZ_finder(indices_list):

                XYZ_coordinates = []
                for index in indices_list:
                    this_joint_coordinate = mediapipe_pose_data[frame,index,:]
                    XYZ_coordinates.append(this_joint_coordinate)

                return XYZ_coordinates

            
        def build_virtual_trunk_marker(trunk_joint_connection,qualisys_indices):
                trunk_indices = mediapipe_index_finder(trunk_joint_connection)

                trunk_XYZ_coordinates = mediapipe_XYZ_finder(trunk_indices)

                trunk_proximal = (trunk_XYZ_coordinates[0] + trunk_XYZ_coordinates[1])/2
                trunk_distal = (trunk_XYZ_coordinates[2] + trunk_XYZ_coordinates[3])/2

                return trunk_proximal, trunk_distal
        
        def build_virtual_foot_marker(foot_joint_connection, qualisys_indices):
                foot_indices = mediapipe_index_finder(foot_joint_connection)

                foot_XYZ_coordinates = mediapipe_XYZ_finder(foot_indices)

                foot_virtual_marker = (foot_XYZ_coordinates[0] + foot_XYZ_coordinates[1])/2

                return foot_virtual_marker

        trunk_joint_connection = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']

        left_foot_joint_connection = ['left_heel','left_ankle']
        right_foot_joint_connection = ['right_heel','right_ankle']

        trunk_virtual_markers = build_virtual_trunk_marker(trunk_joint_connection,qualisys_indices)
        left_foot_virtual_marker = build_virtual_foot_marker(left_foot_joint_connection,qualisys_indices)
        right_foot_virtual_marker = build_virtual_foot_marker(right_foot_joint_connection,qualisys_indices)
    
        return trunk_virtual_markers, left_foot_virtual_marker, right_foot_virtual_marker


    mediapipe_frame_segment_joint_XYZ = [] #empty list to hold all the skeleton XYZ coordinates/frame
    for frame in track(num_frame_range, description= 'Building a Qualisys Skeleton'): #NOTE: need to change frame_range to numFrames 2/09/2022 - AC

        trunk_virtual_markers, left_foot_virtual_marker, right_foot_virtual_marker = build_mediapipe_virtual_markers(qualisys_indices,frame)

        mediapipe_pose_skeleton_coordinates = {}
        for segment,segment_info in segment_dataframe.iterrows(): #iterate through the data frame by the segment name and all the info for that segment
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

                proximal_joint_hand_index = qualisys_indices.index(proximal_joint_hand)
                distal_joint_hand_index = qualisys_indices.index(distal_joint_hand)

                mediapipe_pose_skeleton_coordinates[segment] = [mediapipe_pose_data[frame,proximal_joint_hand_index, :],
                                                                mediapipe_pose_data[frame,distal_joint_hand_index,:]]

            elif segment == 'left_foot' or segment == 'right_foot':
                if segment == 'left_foot':
                    proximal_joint_foot_name = 'left_ankle'
                else:
                    proximal_joint_foot_name = 'right_ankle'
                
                proximal_joint_foot_index = qualisys_indices.index(proximal_joint_foot_name)

                distal_joint_foot = segment_info['Joint Connection'][1]
                distal_joint_foot_index = qualisys_indices.index(distal_joint_foot)
                mediapipe_pose_skeleton_coordinates[segment] = [mediapipe_pose_data[frame,proximal_joint_foot_index, :],
                                                                mediapipe_pose_data[frame, distal_joint_foot_index,:]]            

            else:
                proximal_joint_name = segment_info['Joint Connection'][0] 
                distal_joint_name = segment_info['Joint Connection'][1]

            #get the mediapipe index for the proximal and distal joint for this segment
                proximal_joint_index = qualisys_indices.index(proximal_joint_name)
                distal_joint_index = qualisys_indices.index(distal_joint_name)

            #use the mediapipe indices to get the XYZ coordinates for the prox/distal joints and throw it in a dictionary
            #mediapipe_pose_skeleton_coordinates[segment] = {'proximal':mediapipe_pose_data[frame,proximal_joint_index,:],'distal':mediapipe_pose_data[frame,distal_joint_index,:]}
                mediapipe_pose_skeleton_coordinates[segment] = [mediapipe_pose_data[frame,proximal_joint_index,:],mediapipe_pose_data[frame,distal_joint_index,:]]
                
        mediapipe_frame_segment_joint_XYZ.append(mediapipe_pose_skeleton_coordinates)
        f = 2 
    
    return mediapipe_frame_segment_joint_XYZ

# qualisys_indices = [ #these indices have been renamed to match the joint connections list above
#     'mid_hip_marker',
#     'Spine',
#     'Spine1',
#     'Spine2',
#     'Neck',
#     'head',
#     'LeftShoulder_false',
#     'left_shoulder',
#     'left_elbow',
#     'left_wrist',
#     'left_hand_marker',
#     'RightShoulder_false',
#     'right_shoulder',
#     'right_elbow',
#     'right_wrist',
#     'right_hand_marker',
#     'left_hip',
#     'left_knee',
#     'left_heel',
#     'left_foot_index',
#     'right_hip',
#     'right_knee',
#     'right_heel',
#     'right_foot_index'
# ]

# def build_qualisys_skeleton(qualisys_skeleton_data,segment_dataframe, qualisys_indices, num_frame_range):
#     def qualisys_index_finder(list_of_joint_names, qualisys_indices):

#         indices = []
#         for name in list_of_joint_names:
#             this_name_index = qualisys_indices.index(name)
#             indices.append(this_name_index)
#         return indices

#     def qualisys_XYZ_finder(indices_list,qualisys_skeleton_data):

#             XYZ_coordinates = []
#             for index in indices_list:
#                 this_joint_coordinate = qualisys_skeleton_data[frame,index,:]
#                 XYZ_coordinates.append(this_joint_coordinate)

#             return XYZ_coordinates

#     def build_mid_chest_marker(mid_chest_joints,qualisys_indices,segment_dataframe):

#         chest_indices = qualisys_index_finder(mid_chest_joints, qualisys_indices)
#         chest_coordinates = qualisys_XYZ_finder(chest_indices,qualisys_skeleton_data)
#         mid_chest_marker = (chest_coordinates[0] + chest_coordinates[1])/2

#         return mid_chest_marker

#     mid_chest_joints = ['left_shoulder','right_shoulder']

    
#     qualisys_frame_segment_joint_XYZ = [] #empty list to hold all the skeleton XYZ coordinates/frame
#     for frame in track(num_frame_range, description= 'Building a Qualisys Skeleton'): 
#         qualisys_pose_skeleton_coordinates = {}
#         for segment,segment_info in segment_dataframe.iterrows():
#             if segment == 'head':
#                 head_index = qualisys_indices.index(segment)
#                 qualisys_pose_skeleton_coordinates[segment] = [qualisys_skeleton_data[frame,head_index,:]]
            
#             elif segment == 'trunk':

#                 mid_chest_marker_XYZ = build_mid_chest_marker(mid_chest_joints,qualisys_indices,segment_dataframe)
                
#                 mid_hip_marker_name = segment_info['Joint Connection'][1]
#                 mid_hip_marker_index = qualisys_indices.index(mid_hip_marker_name)
#                 mid_hip_marker_XYZ = qualisys_skeleton_data[frame,mid_hip_marker_index,:]

#                 qualisys_pose_skeleton_coordinates[segment] = [mid_chest_marker_XYZ,mid_hip_marker_XYZ]


#             else:
#                 proximal_joint_name = segment_info['Joint Connection'][0] 
#                 distal_joint_name = segment_info['Joint Connection'][1]

#                 if 'ankle' in distal_joint_name: #qualysis doesn't have an ankle joint like mediapipe does, so need to replace the ankle joint with the foot markers they do track 
#                     if segment == 'right_shin':
#                         distal_joint_name = 'right_heel'
#                     else:
#                         distal_joint_name = 'left_heel'

#             #get the mediapipe index for the proximal and distal joint for this segment
#                 proximal_joint_index = qualisys_indices.index(proximal_joint_name)
#                 distal_joint_index = qualisys_indices.index(distal_joint_name)
#                 qualisys_pose_skeleton_coordinates[segment] = [qualisys_skeleton_data[frame,proximal_joint_index,:],qualisys_skeleton_data[frame,distal_joint_index,:]]
#         qualisys_frame_segment_joint_XYZ.append(qualisys_pose_skeleton_coordinates)

#     return qualisys_frame_segment_joint_XYZ
