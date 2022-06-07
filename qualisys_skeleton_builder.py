from rich.progress import track 

qualisys_indices = [ #these indices have been renamed to match the joint connections list above
    'mid_hip_marker',
    'Spine',
    'Spine1',
    'Spine2',
    'Neck',
    'head',
    'LeftShoulder_false',
    'left_shoulder',
    'left_elbow',
    'left_wrist',
    'left_hand_marker',
    'RightShoulder_false',
    'right_shoulder',
    'right_elbow',
    'right_wrist',
    'right_hand_marker',
    'left_hip',
    'left_knee',
    'left_back_of_foot_marker',
    'left_foot_index',
    'right_hip',
    'right_knee',
    'right_back_of_foot_marker',
    'right_foot_index'
]

def build_qualisys_skeleton(qualisys_skeleton_data,segment_dataframe, qualisys_indices, num_frame_range):
    def qualisys_index_finder(list_of_joint_names, qualisys_indices):

        indices = []
        for name in list_of_joint_names:
            this_name_index = qualisys_indices.index(name)
            indices.append(this_name_index)
        return indices

    def qualisys_XYZ_finder(indices_list,qualisys_skeleton_data):

            XYZ_coordinates = []
            for index in indices_list:
                this_joint_coordinate = qualisys_skeleton_data[frame,index,:]
                XYZ_coordinates.append(this_joint_coordinate)

            return XYZ_coordinates

    def build_mid_chest_marker(mid_chest_joints,qualisys_indices,segment_dataframe):

        chest_indices = qualisys_index_finder(mid_chest_joints, qualisys_indices)
        chest_coordinates = qualisys_XYZ_finder(chest_indices,qualisys_skeleton_data)
        mid_chest_marker = (chest_coordinates[0] + chest_coordinates[1])/2

        return mid_chest_marker

    mid_chest_joints = ['left_shoulder','right_shoulder']

    
    qualisys_frame_segment_joint_XYZ = [] #empty list to hold all the skeleton XYZ coordinates/frame
    for frame in track(num_frame_range, description= 'Building a Qualisys Skeleton'): 
        qualisys_pose_skeleton_coordinates = {}
        for segment,segment_info in segment_dataframe.iterrows():
            if segment == 'head':
                head_index = qualisys_indices.index(segment)
                qualisys_pose_skeleton_coordinates[segment] = [qualisys_skeleton_data[frame,head_index,:]]
            
            elif segment == 'trunk':

                mid_chest_marker_XYZ = build_mid_chest_marker(mid_chest_joints,qualisys_indices,segment_dataframe)
                
                mid_hip_marker_name = segment_info['Joint Connection'][1]
                mid_hip_marker_index = qualisys_indices.index(mid_hip_marker_name)
                mid_hip_marker_XYZ = qualisys_skeleton_data[frame,mid_hip_marker_index,:]

                qualisys_pose_skeleton_coordinates[segment] = [mid_chest_marker_XYZ,mid_hip_marker_XYZ]


            else:
                proximal_joint_name = segment_info['Joint Connection'][0] 
                distal_joint_name = segment_info['Joint Connection'][1]

                if 'ankle' in distal_joint_name: #qualysis doesn't have an ankle joint like mediapipe does, so need to replace the ankle joint with the foot markers they do track 
                    if segment == 'right_shin':
                        distal_joint_name = 'right_back_of_foot_marker'
                    else:
                        distal_joint_name = 'left_back_of_foot_marker'

            #get the mediapipe index for the proximal and distal joint for this segment
                proximal_joint_index = qualisys_indices.index(proximal_joint_name)
                distal_joint_index = qualisys_indices.index(distal_joint_name)
                qualisys_pose_skeleton_coordinates[segment] = [qualisys_skeleton_data[frame,proximal_joint_index,:],qualisys_skeleton_data[frame,distal_joint_index,:]]
        qualisys_frame_segment_joint_XYZ.append(qualisys_pose_skeleton_coordinates)

    return qualisys_frame_segment_joint_XYZ
