
# joint_center_weights = {
#     'right_hip_xyz': {
#         'x': {
#             'right_upper_hip': .25,
#             'right_back_upper_hip': .25,
#             'right_front_hip': .25,
#             'right_upper_leg': .25,
#         },
#         'y': {
#             'right_upper_hip': .25,
#             'right_back_upper_hip': .25,
#             'right_front_hip': .25,
#             'right_upper_leg': .25,
#         },
#         'z': {
#             'right_upper_hip': .25,
#             'right_back_upper_hip': .25,
#             'right_front_hip': .25,
#             'right_upper_leg': .25,
#         },
#     },
#     'left_hip_xyz': {
#         'x': {
#             'left_upper_hip': .25,
#             'left_back_upper_hip': .25,
#             'left_front_hip': .25,
#             'left_upper_leg': .25,
#         },
#         'y': {
#             'left_upper_hip': .25,
#             'left_back_upper_hip': .25,
#             'left_front_hip': .25,
#             'left_upper_leg': .25,
#         },
#         'z': {
#             'left_upper_hip': .25,
#             'left_back_upper_hip': .25,
#             'left_front_hip': .25,
#             'left_upper_leg': .25,
#         },
#     },
#     'right_knee_xyz': {
#         'x': {
#             'right_outside_upper_knee': .25,
#             'right_inside_upper_knee': .25,
#             'right_outside_lower_knee': .25,
#             'right_inside_lower_knee': .25,
#         },
#         'y': {
#             'right_outside_upper_knee': .25,
#             'right_inside_upper_knee': .25,
#             'right_outside_lower_knee': .25,
#             'right_inside_lower_knee': .25,
#         },
#         'z': {
#             'right_outside_upper_knee': .25,
#             'right_inside_upper_knee': .25,
#             'right_outside_lower_knee': .25,
#             'right_inside_lower_knee': .25,
#         },
#     },
#     'left_knee_xyz': {
#         'x': {
#             'left_outside_upper_knee': .25,
#             'left_inside_upper_knee': .25,
#             'left_outside_lower_knee': .25,
#             'left_inside_lower_knee': .25,
#         },
#         'y': {
#             'left_outside_upper_knee': .25,
#             'left_inside_upper_knee': .25,
#             'left_outside_lower_knee': .25,
#             'left_inside_lower_knee': .25,
#         },
#         'z': {
#             'left_outside_upper_knee': .25,
#             'left_inside_upper_knee': .25,
#             'left_outside_lower_knee': .25,
#             'left_inside_lower_knee': .25,
#         },
#     },
#     'right_ankle_xyz': {
#         'x': {
#             'right_outside_ankle': .5,
#             'right_inside_ankle': .5,
#         },
#         'y': {
#             'right_outside_ankle': .5,
#             'right_inside_ankle': .5,
#         },
#         'z': {
#             'right_outside_ankle': .5,
#             'right_inside_ankle': .5,
#         },
#     },
#     'left_ankle_xyz': {
#         'x': {
#             'left_outside_ankle': .5,
#             'left_inside_ankle': .5,
#         },
#         'y': {
#             'left_outside_ankle': .5,
#             'left_inside_ankle': .5,
#         },
#         'z': {
#             'left_outside_ankle': .5,
#             'left_inside_ankle': .5,
#         },
#     },
#     'right_heel_xyz': {
#         'x': {
#             'right_upper_heel': .34,
#             'right_lower_heel': .33,
#             'right_lateral_heel': .33,
#         },
#         'y': {
#             'right_upper_heel': .34,
#             'right_lower_heel': .33,
#             'right_lateral_heel': .33,
#         },
#         'z': {
#             'right_upper_heel': .34,
#             'right_lower_heel': .33,
#             'right_lateral_heel': .33,
#         },
#     },
#     'left_heel_xyz': {
#         'x': {
#             'left_upper_heel': .34,
#             'left_lower_heel': .33,
#             'left_lateral_heel': .33,
#         },
#         'y': {
#             'left_upper_heel': .34,
#             'left_lower_heel': .33,
#             'left_lateral_heel': .33,
#         },
#         'z': {
#             'left_upper_heel': .34,
#             'left_lower_heel': .33,
#             'left_lateral_heel': .33,
#         },
#     },
#     'right_foot_index_xyz': {
#         'x': {
#             'right_inside_foot': .25,
#             'right_top_foot': .25,
#             'right_outside_foot': .25,
#             'right_toe': .25,
#         },
#         'y': {
#             'right_inside_foot': .25,
#             'right_top_foot': .25,
#             'right_outside_foot': .25,
#             'right_toe': .25,
#         },
#         'z': {
#             'right_inside_foot': .25,
#             'right_top_foot': .25,
#             'right_outside_foot': .25,
#             'right_toe': .25,
#         },
#     },
#     'left_foot_index_xyz': {
#         'x': {
#             'left_inside_foot': .25,
#             'left_top_foot': .25,
#             'left_outside_foot': .25,
#             'left_toe': .25,
#         },
#         'y': {
#             'left_inside_foot': .25,
#             'left_top_foot': .25,
#             'left_outside_foot': .25,
#             'left_toe': .25,
#         },
#         'z': {
#             'left_inside_foot': .25,
#             'left_top_foot': .25,
#             'left_outside_foot': .25,
#             'left_toe': .25,
#         },
#     },
# }


joint_center_weights = {
    'right_hip': {
        'right_upper_hip': [.25, .25, .25],
        'right_back_upper_hip': [.25, .25, .25],
        'right_front_hip': [.25, .25, .25],
        'right_upper_leg': [.25, .25, .25],
    },
    'left_hip': {
        'left_upper_hip': [.25, .25, .25],
        'left_back_upper_hip': [.25, .25, .25],
        'left_front_hip': [.25, .25, .25],
        'left_upper_leg': [.25, .25, .25],
    },
    'right_knee': {
        'right_outside_upper_knee': [.25, .25, .25],
        'right_inside_upper_knee': [.25, .25, .25],
        'right_outside_lower_knee': [.25, .25, .25],
        'right_inside_lower_knee': [.25, .25, .25],
    },
    'left_knee': {
        'left_outside_upper_knee': [.25, .25, .25],
        'left_inside_upper_knee': [.25, .25, .25],
        'left_outside_lower_knee': [.25, .25, .25],
        'left_inside_lower_knee': [.25, .25, .25],
    },
    'right_ankle': {
        'right_outside_ankle': [.5, .5, .5],
        'right_inside_ankle': [.5, .5, .5],
    },
    'left_ankle': {
        'left_outside_ankle': [.5, .5, .5],
        'left_inside_ankle': [.5, .5, .5],
    },
    'right_heel': {
        'right_upper_heel': [.34, .34, .34],
        'right_lower_heel': [.33, .33, .33],
        'right_lateral_heel': [.33, .33, .33],
    },
    'left_heel': {
        'left_upper_heel': [.34, .34, .34],
        'left_lower_heel': [.33, .33, .33],
        'left_lateral_heel': [.33, .33, .33],
    },
    'right_foot_index': {
        'right_inside_foot': [.25, .25, .25],
        'right_top_foot': [.25, .25, .25],
        'right_outside_foot': [.25, .25, .25],
        'right_toe': [.25, .25, .25],
    },
    'left_foot_index': {
        'left_inside_foot': [.25, .25, .25],
        'left_top_foot': [.25, .25, .25],
        'left_outside_foot': [.25, .25, .25],
        'left_toe': [.25, .25, .25],
    },
}