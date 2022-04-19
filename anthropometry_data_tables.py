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
