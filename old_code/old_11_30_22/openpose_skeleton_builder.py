from rich.progress import track 
import numpy as np 

openpose_indices = [
    'nose',
    'neck',
    'right_shoulder',
    'right_elbow',
    'right_wrist',
    'left_shoulder',
    'left_elbow',
    'left_wrist',
    'mid_hip_marker',
    'right_hip',
    'right_knee',
    'right_ankle',
    'left_hip',
    'left_knee',
    'left_ankle',
    'right_eye',
    'light_eye',
    'right_ear',
    'left_ear',
    'left_foot_index',
    'left_bigtoe',
    'left_smalltoe',
    'left_heel',
    'left_back_of_foot_marker',
    'right_bigtoe',
    'right_foot_index',
    'right_smalltoe',
    'right_heel',
    'right_back_of_foot_marker',
    'right_hand_marker',
    'left_hand_marker',
]


def slice_openpose_data(openpose_full_skeleton_data, num_pose_joints):
            
    indices_to_grab = [i for i in range(num_pose_joints)]
    indices_to_grab.append(32) #right_index_finger
    indices_to_grab.append(54) #left_index_finger

    openpose_pose_data = openpose_full_skeleton_data[:,indices_to_grab,:] #load just the pose joints into a data array, removing hands and face data 


    return openpose_pose_data

def build_openpose_skeleton(openpose_pose_data,segment_dataframe, openpose_indices, num_frame_range):
    

        
    def openpose_index_finder(list_of_joint_names, openpose_indices):
        indices_list = []
        for name in list_of_joint_names:
            this_name_index = openpose_indices.index(name)
            indices_list.append(this_name_index)

        return indices_list

    def openpose_XYZ_finder(indices_list,openpose_pose_data,frame):

        XYZ_coordinates = []
        for index in indices_list:
            this_joint_coordinate = openpose_pose_data[frame,index,:]
            XYZ_coordinates.append(this_joint_coordinate)

        return XYZ_coordinates

    def build_virtual_chest_marker(midchest_joint_connection, openpose_indices):

        shoulder_indices = openpose_index_finder(midchest_joint_connection,openpose_indices)

        shoulder_coordinates = openpose_XYZ_finder(shoulder_indices, openpose_pose_data,frame)

        midchest_coordinates = (shoulder_coordinates[0] + shoulder_coordinates[1])/2

        return midchest_coordinates

    openpose_frame_segment_joint_XYZ = [] #empty list to hold all the skeleton XYZ coordinates/frame
    
    midchest_joint_connection = ['left_shoulder','right_shoulder']

    for frame in track(num_frame_range, description= 'Building an OpenPose Skeleton'): #NOTE: need to change frame_range to numFrames 2/09/2022 - AC
        openpose_pose_skeleton_coordinates = {}
        for segment,segment_info in segment_dataframe.iterrows(): #iterate through the data frame by the segment name and all the info for that segment

            if segment == 'trunk':

                midchest_coordinates = build_virtual_chest_marker(midchest_joint_connection, openpose_indices)

                midhip_index = openpose_index_finder(['mid_hip_marker'], openpose_indices)

                midhip_coordinates = openpose_XYZ_finder(midhip_index, openpose_pose_data, frame)

                openpose_pose_skeleton_coordinates[segment] = [midchest_coordinates, midhip_coordinates[0]] #need to make it 0 because of reasons I need to fix later


            else:

                proximal_joint_name = segment_info['Joint Connection'][0]
                distal_joint_name = segment_info['Joint Connection'][1]

                proximal_joint_index = openpose_index_finder([proximal_joint_name], openpose_indices)
                distal_joint_index = openpose_index_finder([distal_joint_name], openpose_indices)

                proximal_joint_coordinates = openpose_XYZ_finder(proximal_joint_index, openpose_pose_data, frame)
                distal_joint_coordinates = openpose_XYZ_finder(distal_joint_index, openpose_pose_data, frame)

                #we index these at 0 because the coordinates get returned as a list with a single value, will fix later
                openpose_pose_skeleton_coordinates[segment] = [proximal_joint_coordinates[0], distal_joint_coordinates[0]]

        openpose_frame_segment_joint_XYZ.append(openpose_pose_skeleton_coordinates)

    return openpose_frame_segment_joint_XYZ