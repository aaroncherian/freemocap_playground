from freemocap_utils.mediapipe_skeleton_builder import build_skeleton, mediapipe_indices, mediapipe_connections
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def calculate_distance(point1, point2):
    return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2 + (point1[2]-point2[2])**2)

def calculate_leg_length(skeleton_for_this_frame, leg_segment_name:str):
    lower_leg_length = calculate_distance(skeleton_for_this_frame[f'{leg_segment_name}_lower_leg'][0],skeleton_for_this_frame[f'{leg_segment_name}_lower_leg'][1])
    upper_leg_length = calculate_distance(skeleton_for_this_frame[f'{leg_segment_name}_upper_leg'][0],skeleton_for_this_frame[f'{leg_segment_name}_upper_leg'][1])
    leg_length = lower_leg_length + upper_leg_length
    return leg_length



def create_leg_length_dict(skeleton_connections_dict:dict):    
    leg_length_dict = {
        label:{
            'left_leg_length': [calculate_leg_length(skeleton_for_this_frame=skeleton[frame], leg_segment_name='left') for frame in range(len(skeleton))],
            'right_leg_length': [calculate_leg_length(skeleton_for_this_frame=skeleton[frame], leg_segment_name='right') for frame in range(len(skeleton))]
        }
        for label, skeleton in skeleton_connections_dict.items()
    }
    return leg_length_dict

def calculate_mean_leg_lengths(leg_length_dict):
    mean_leg_lengths = {}
    for label, data in leg_length_dict.items():
        mean_leg_lengths[label] = {
            'left_leg_mean_length': np.mean(data['left_leg_length']),
            'right_leg_mean_length': np.mean(data['right_leg_length'])
        }
    return mean_leg_lengths


path_to_recording_folder = Path(r'C:\Users\aaron\FreeMocap_Data\recording_sessions')

session_id_list = ['recording_15_19_00_gmt-4__brit_baseline','recording_15_20_51_gmt-4__brit_half_inch', 'recording_15_22_56_gmt-4__brit_one_inch','recording_15_24_58_gmt-4__brit_two_inch']
label_list = ['baseline', 'half inch lift', 'one inch lift', 'two inch lift']

skeleton_dict = {}
for session_id, label in zip(session_id_list, label_list):
    path_to_data = path_to_recording_folder/session_id/'output_data'/'mediapipe_body_3d_xyz_transformed.npy'

    marker_data_3d = np.load(path_to_data)
    mediapipe_skeleton = build_skeleton(skel_3d_data=marker_data_3d, pose_estimation_markers=mediapipe_indices, pose_estimation_connections=mediapipe_connections)
    
    skeleton_dict[label] = mediapipe_skeleton


leg_length_dict = create_leg_length_dict(skeleton_connections_dict=skeleton_dict)
f = 2
mean_leg_lengths = calculate_mean_leg_lengths(leg_length_dict)


figure = plt.figure()

left_leg_ax = figure.add_subplot(211)
right_leg_ax = figure.add_subplot(212)

# for label in label_list:
#     left_leg_ax.plot(leg_length_dict[label]['left_leg_length'], label = label)
#     right_leg_ax.plot(leg_length_dict[label]['right_leg_length'], label = label)


# left_leg_ax.set_title('Left Leg Length')
# right_leg_ax.set_title('Right Leg Length')
# left_leg_ax.legend()
# right_leg_ax.legend()

# plt.show()

# def plot_mean_leg_lengths(mean_leg_lengths):
#     labels = []
#     left_leg_means = []
#     right_leg_means = []

#     for label, data in mean_leg_lengths.items():
#         labels.append(label)
#         left_leg_means.append(data['left_leg_mean_length'])
#         right_leg_means.append(data['right_leg_mean_length'])

#     x = np.arange(len(labels))
#     width = 0.35

#     fig, ax = plt.subplots()
#     rects1 = ax.bar(x - width/2, left_leg_means, width, label='Left Leg')
#     rects2 = ax.bar(x + width/2, right_leg_means, width, label='Right Leg')

#     ax.set_ylabel('Mean Leg Length')
#     ax.set_title('Mean Leg Lengths by Shoe Lift Condition')
#     ax.set_xticks(x)
#     ax.set_xticklabels(labels)
#     ax.legend()

#     fig.tight_layout()
#     plt.show()

# plot_mean_leg_lengths(mean_leg_lengths)

# def plot_mean_leg_lengths_grouped(mean_leg_lengths):
#     labels = []
#     left_leg_means = []
#     right_leg_means = []

#     for label, data in mean_leg_lengths.items():
#         labels.append(label)
#         left_leg_means.append(data['left_leg_mean_length'])
#         right_leg_means.append(data['right_leg_mean_length'])

#     x = np.arange(len(labels))
#     width = 0.35

#     fig, ax = plt.subplots()
#     rects1 = ax.bar(x - width, left_leg_means, width, label='Left Leg')
#     rects2 = ax.bar(x, right_leg_means, width, label='Right Leg')

#     ax.set_ylabel('Mean Leg Length')
#     ax.set_title('Mean Leg Lengths by Shoe Lift Condition')
#     ax.set_xticks(x - width/2)
#     ax.set_xticklabels(labels)
#     ax.legend()

#     fig.tight_layout()
#     plt.show()

# plot_mean_leg_lengths_grouped(mean_leg_lengths)

def plot_mean_leg_lengths_separated(mean_leg_lengths):
    labels = ['Left Leg', 'Right Leg']
    left_leg_means = []
    right_leg_means = []

    for _, data in mean_leg_lengths.items():
        left_leg_means.append(data['left_leg_mean_length'])
        right_leg_means.append(data['right_leg_mean_length'])

    x = np.arange(len(labels))
    width = 0.15

    colors = ['b', 'g', 'r', 'c']  # You can define more colors if you have more conditions
    fig, ax = plt.subplots()
    for i, (left_leg_mean, right_leg_mean) in enumerate(zip(left_leg_means, right_leg_means)):
        rects1 = ax.bar(x[0] + width * i, left_leg_mean, width, label=f'{list(mean_leg_lengths.keys())[i]}', color=colors[i])
        rects2 = ax.bar(x[1] + width * i, right_leg_mean, width, color=colors[i])

        for rects in [rects1, rects2]:
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')

    ax.set_ylabel('Mean Leg Length')
    ax.set_title('Mean Leg Lengths by Shoe Lift Condition')
    ax.set_xticks(x + width * (len(left_leg_means) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()
    plt.show()

plot_mean_leg_lengths_separated(mean_leg_lengths)