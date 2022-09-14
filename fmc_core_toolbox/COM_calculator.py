import numpy as np
from rich.progress import track 


def calculate_segment_COM(segment_conn_len_perc_dataframe,skelcoordinates_frame_segment_joint_XYZ, num_frame_range):
    segment_COM_frame_dict = []
    for frame in track(num_frame_range, description = 'Calculating Segment Center of Mass'):
        segment_COM_dict = {}
        for segment,segment_info in segment_conn_len_perc_dataframe.iterrows():
            this_segment_XYZ = skelcoordinates_frame_segment_joint_XYZ[frame][segment]

            #for mediapipe
            this_segment_proximal = this_segment_XYZ[0]
            this_segment_distal = this_segment_XYZ[1]
            this_segment_COM_length = segment_info['Segment_COM_Length']

            this_segment_COM = this_segment_proximal + this_segment_COM_length*(this_segment_distal-this_segment_proximal)
            segment_COM_dict[segment] = this_segment_COM
        segment_COM_frame_dict.append(segment_COM_dict)
    return segment_COM_frame_dict


def reformat_segment_COM(segment_COM_frame_dict, num_frame_range,num_segments):
    
    segment_COM_frame_imgPoint_XYZ = np.empty([int(len(num_frame_range)),int(num_segments),3])
    for frame in num_frame_range:
        this_frame_skeleton = segment_COM_frame_dict[frame]
        for joint_count,segment in enumerate(this_frame_skeleton.keys()):
            segment_COM_frame_imgPoint_XYZ[frame,joint_count,:] = this_frame_skeleton[segment]
    return segment_COM_frame_imgPoint_XYZ



def calculate_total_body_COM(segment_conn_len_perc_dataframe,segment_COM_frame_dict,num_frame_range):
    totalBodyCOM_frame_XYZ = np.empty([int(len(num_frame_range)),3])

    for frame in track(num_frame_range, description = 'Calculating Total Body Center of Mass'):

        this_frame_total_body_percentages = []
        this_frame_skeleton = segment_COM_frame_dict[frame]

        for segment, segment_info in segment_conn_len_perc_dataframe.iterrows():

            this_segment_COM = this_frame_skeleton[segment]
            this_segment_COM_percentage = segment_info['Segment_COM_Percentage']

            this_segment_total_body_percentage = this_segment_COM * this_segment_COM_percentage
            this_frame_total_body_percentages.append(this_segment_total_body_percentage)

        this_frame_total_body_COM = np.nansum(this_frame_total_body_percentages,axis = 0)
       
        totalBodyCOM_frame_XYZ[frame,:] = this_frame_total_body_COM

    f=2
    return totalBodyCOM_frame_XYZ        