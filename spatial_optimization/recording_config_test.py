from skellyalign.models.recording_config import RecordingConfig
from mdn_validation_marker_set import markers_to_extract, qualisys_nih_markers, mediapipe_markers
from pathlib import Path
import numpy as np
from skellymodels.create_skeleton import create_skeleton_model
from skellymodels.model_info.mediapipe_model_info import MediapipeModelInfo


path_to_recording = Path(r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_14_53_48_MDN_NIH_Trial3")


sample_recording_config = RecordingConfig(
    path_to_recording=path_to_recording,
    path_to_freemocap_output_data = path_to_recording/'output_data'/'mediapipe_body_3d_xyz.npy',
    path_to_qualisys_output_data = path_to_recording/'qualisys'/ 'clipped_qualisys_skel_3d.npy',
    freemocap_markers=mediapipe_markers,
    qualisys_markers=qualisys_nih_markers,
    markers_for_alignment=markers_to_extract,
    frames_to_sample=20,
    max_iterations=20,
    inlier_threshold=50

)

freemocap_data = np.load(sample_recording_config.path_to_freemocap_output_data)
mediapipe_model_info = MediapipeModelInfo()

freemocap_model = create_skeleton_model(actual_markers=mediapipe_model_info.landmark_names, 
                      num_tracked_points=mediapipe_model_info.num_tracked_points,
                      segment_connections=mediapipe_model_info.segment_connections,
                     virtual_markers=mediapipe_model_info.virtual_markers_definitions,
                     joint_hierarchy= mediapipe_model_info.joint_hierarchy,
                     center_of_mass_info=mediapipe_model_info.center_of_mass_definitions
)


freemocap_model.integrate_freemocap_3d_data(freemocap_data)
# from skellyalign.run_alignment import run_ransac_alignment

# aligned_freemocap_data = run_ransac_alignment(sample_recording_config)

f =2 